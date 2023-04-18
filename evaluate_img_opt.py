import torch
from torchvision.transforms.functional import to_pil_image

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline
import glob
import os
from PIL import Image
from tqdm import tqdm
import argparse
import json
import random

from datasets_loading import get_dataset
from torch.utils.data import DataLoader
from utils import evaluate_scores
import csv
from accelerate import Accelerator

import cProfile

class Scorer:
    def __init__(self, args, clip_model=None, preprocess=None):

        # vae = AutoencoderKL.from_pretrained('./stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
        # vae.to(device='cuda', dtype=torch.bfloat16)
        # self.vae_model = StableDiffusionImg2LatentPipeline(vae).to('cuda')
        self.cache_dir = args.cache_dir if args.cache else None


    def score_batch(self, i, args, batch, model):
        """
        Takes a batch of images and captions and returns a score for each image-caption pair.
        """

        imgs, texts = batch[0], batch[1]
        imgs, imgs_resize = imgs[0], imgs[1]
        imgs, imgs_resize = [img.cuda() for img in imgs], [img.cuda() for img in imgs_resize]

        scores = []
        for txt_idx, text in enumerate(texts):
            for img_idx, resized_img in enumerate(imgs_resize):
                if len(resized_img.shape) == 3:
                    resized_img = resized_img.unsqueeze(0)
                
                print(f'Batch {i}, Text {txt_idx}, Image {img_idx}')
                dists = model.optimize_input(prompt=list(text), image=resized_img, scoring=True, guidance_scale=0.0, sampling_steps=args.sampling_steps, unconditional=args.img_retrieval, optim_steps=args.optim_steps, optim_lr=args.optim_lr, distance=args.distance)
                dists = dists.to(torch.float32)
                dists = dists.mean(dim=-1)
                dists = -dists
                scores.append(dists)
        model.reset_sampling()

        scores = torch.stack(scores).permute(1, 0) if args.batchsize > 1 else torch.stack(scores).unsqueeze(0)
        return scores
        

def main(args):

    accelerator = Accelerator()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
    model = model.to(accelerator.device)

    scorer = Scorer(args)
    dataset = get_dataset(args.task, f'datasets/{args.task}', transform=None)

    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=0)

    model, dataloader = accelerator.prepare(model, dataloader)

    r1s = []
    r5s = []
    max_more_than_onces = 0
    metrics = []
    ids = []
    clevr_dict = {}
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if args.subset and i % 10 != 0:
            continue
        scores = scorer.score_batch(i, args, batch, model)
        scores = scores.contiguous()
        accelerator.wait_for_everyone()
        scores = accelerator.gather(scores)
        print(scores)
        batch[-1] = accelerator.gather(batch[-1])
        if accelerator.is_main_process:
            if args.task == 'winoground':
                text_scores, img_scores, group_scores = evaluate_scores(args, scores, batch)
                metrics += list(zip(text_scores, img_scores, group_scores))
                text_score = sum([m[0] for m in metrics]) / len(metrics)
                img_score = sum([m[1] for m in metrics]) / len(metrics)
                group_score = sum([m[2] for m in metrics]) / len(metrics)
                print(f'Text score: {text_score}')
                print(f'Image score: {img_score}')
                print(f'Group score: {group_score}')
                print(len(metrics))
                with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
                    f.write(f'Text score: {text_score}\n')
                    f.write(f'Image score: {img_score}\n')
                    f.write(f'Group score: {group_score}\n')
            elif args.task in ['flickr30k', 'imagecode', 'imagenet', 'flickr30k_text']:
                r1,r5, max_more_than_once = evaluate_scores(args, scores, batch)
                r1s += r1
                r5s += r5
                max_more_than_onces += max_more_than_once
                r1 = sum(r1s) / len(r1s)
                r5 = sum(r5s) / len(r5s)
                print(f'R@1: {r1}')
                print(f'R@5: {r5}')
                print(f'Max more than once: {max_more_than_onces}')
                with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
                    f.write(f'R@1: {r1}\n')
                    f.write(f'R@5: {r5}\n')
                    f.write(f'Max more than once: {max_more_than_onces}\n')
                    f.write(f"Sample size {len(metrics)}\n")
            elif args.task == 'clevr':
                acc_list, max_more_than_once = evaluate_scores(args, scores, batch)
                print(acc_list)
                metrics += acc_list
                acc = sum(metrics) / len(metrics)
                max_more_than_onces += max_more_than_once
                print(f'Accuracy: {acc}')
                print(f'Max more than once: {max_more_than_onces}')
                with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
                    f.write(f'Accuracy: {acc}\n')
                    f.write(f'Max more than once: {max_more_than_onces}\n')
                    f.write(f"Sample size {len(metrics)}\n")

                # now do the same but for every subtask of CLEVR
                subtasks = batch[-2]
                for i, subtask in enumerate(subtasks):
                    if subtask not in clevr_dict:
                        clevr_dict[subtask] = []
                    clevr_dict[subtask].append(acc_list[i])
                for subtask in clevr_dict:
                    print(f'{subtask} accuracy: {sum(clevr_dict[subtask]) / len(clevr_dict[subtask])}')
                    with open(f'./paper_results/{args.run_id}_results.txt', 'a') as f:
                        f.write(f'{subtask} accuracy: {sum(clevr_dict[subtask]) / len(clevr_dict[subtask])}\n')
            else:
                acc, max_more_than_once = evaluate_scores(args, scores, batch)
                metrics += acc
                acc = sum(metrics) / len(metrics)
                max_more_than_onces += max_more_than_once
                print(f'Accuracy: {acc}')
                print(f'Max more than once: {max_more_than_onces}')
                with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
                    f.write(f'Accuracy: {acc}\n')
                    f.write(f'Max more than once: {max_more_than_onces}\n')
                    f.write(f"Sample size {len(metrics)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    # parser.add_argument('--similarity', type=str, default='clip')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--sampling_steps', type=int, default=50)
    parser.add_argument('--optim_steps', type=int, default=5)
    parser.add_argument('--optim_lr', type=float, default=0.01)
    parser.add_argument('--img_retrieval', action='store_true')
    parser.add_argument('--distance', type=str, default='loss')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)

    args.run_id = f'{args.task}_img_opt_seed{args.seed}_steps{args.sampling_steps}_subset{args.subset}_img_retrieval{args.img_retrieval}_distance{args.distance}_optim_steps{args.optim_steps}_optim_lr{args.optim_lr}'

    if args.cache:
        args.cache_dir = f'./cache/{args.run_id}'
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
            
    main(args)