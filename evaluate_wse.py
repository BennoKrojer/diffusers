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
                print(f'Batch {i}, Text {txt_idx}, Image {img_idx}')
                dists = model(prompt=list(text), image=resized_img, scoring=True, guidance_scale=0.0)
                dists = dists.mean(dim=1)
                dists = -dists
                scores.append(dists)

                # for n, image_pred in enumerate(images_pred):
                #     image_pred[0].save(f'{args.cache_dir}/_{i}_{img_idx}_{n}.png')
                # # save resized image too
                # resized_img = resized_img.cpu()
                # resized_img = to_pil_image(resized_img.squeeze())
                # resized_img.save(f'{args.cache_dir}/_{i}_{img_idx}_resized.png')
        model.reset_sampling()

        scores = torch.stack(scores).permute(1, 0) if args.batchsize > 1 else torch.stack(scores).unsqueeze(0)
        return scores
        

def main(args):

    accelerator = Accelerator()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    model = model.to(accelerator.device)

    scorer = Scorer(args)
    dataset = get_dataset(args.task, f'datasets/{args.task}', transform=None)

    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    model, dataloader = accelerator.prepare(model, dataloader)

    r1s = []
    r5s = []
    metrics = []
    all_scores = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if args.subset and i % 10 != 0:
            continue
        scores = scorer.score_batch(i, args, batch, model)
        scores = scores.contiguous()
        accelerator.wait_for_everyone()
        scores = accelerator.gather(scores)
        batch[-1] = accelerator.gather(batch[-1])
        all_scores.append(list(scores.cpu().numpy()[0]))

        if args.task == 'winoground':
            score = evaluate_scores(args, scores, batch)
            metrics.append(score)
            text_score = sum([m[0] for m in metrics]) / len(metrics)
            img_score = sum([m[1] for m in metrics]) / len(metrics)
            group_score = sum([m[2] for m in metrics]) / len(metrics)
            print(f'Text score: {text_score}')
            print(f'Image score: {img_score}')
            print(f'Group score: {group_score}')
        elif args.task in ['flickr30k', 'imagecode', 'imagenet']:
            r1,r5,_ = evaluate_scores(args, scores, batch)
            r1s.append(r1)
            r5s.append(r5)
            r1 = sum(r1s) / len(r1s)
            r5 = sum(r5s) / len(r5s)
            print(f'R@1: {r1}')
            print(f'R@5: {r5}')
        else:
            acc = evaluate_scores(args, scores, batch)
            metrics.append(acc)
            acc = sum(metrics) / len(metrics)
            print(f'Accuracy: {acc}')
    # write all scores to csv
    with open(f'./cache/{args.run_id}_scores.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(all_scores)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--similarity', type=str, default='clip')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--subset', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)

    args.run_id = f'{args.task}/wse_sanity_14mar_{args.similarity}_seed{args.seed}'

    if args.cache:
        args.cache_dir = f'./cache/{args.run_id}'
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
            
    main(args)