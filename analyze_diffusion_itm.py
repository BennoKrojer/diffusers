import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np

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
from utils import evaluate_scores, save_bias_scores, save_bias_results
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
        imgs_resize =[img.cuda() for img in imgs_resize]

        scores = []
        for txt_idx, text in enumerate(texts):
            for img_idx, resized_img in enumerate(imgs_resize):
                if len(resized_img.shape) == 3:
                    resized_img = resized_img.unsqueeze(0)
                
                print(f'Batch {i}, Text {txt_idx}, Image {img_idx}')
                dists = model(prompt=list(text), image=resized_img, scoring=True, guidance_scale=args.guidance_scale, sampling_steps=args.sampling_steps, unconditional=args.img_retrieval, gray_baseline=args.gray_baseline)
                dists = dists.to(torch.float32)
                dists = dists.mean(dim=1)
                dists = -dists
                scores.append(dists)

        scores = torch.stack(scores).permute(1, 0) if args.batchsize > 1 else torch.stack(scores).unsqueeze(0)
        return scores
        

def main(args):

    accelerator = Accelerator()
    if args.version == '2.1':
        model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    else:
        model_id = "./stable-diffusion-v1-5"
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(accelerator.device)
    if args.lora_dir != '':
        model.unet.load_attn_procs(args.lora_dir)

    scorer = Scorer(args)
    dataset = get_dataset(args.task, f'datasets/{args.task}', transform=None, targets=args.targets)

    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    model, dataloader = accelerator.prepare(model, dataloader)

    r1s = []
    max_more_than_onces = 0
    metrics = []
    prediction_results = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i < args.skip:
            continue
        if args.subset and i % 10 != 0:
            continue
        scores = scorer.score_batch(i, args, batch, model)
        scores = scores.contiguous()
        accelerator.wait_for_everyone()
        # print(scores)
        scores = accelerator.gather(scores)
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
            else:
                r1, max_more_than_once = evaluate_scores(args, scores, batch)
                r1s += r1
                max_more_than_onces += max_more_than_once
                r1_final = sum(r1s) / len(r1s)
                print(f'R@1: {r1_final}')
                print(f'Max more than once: {max_more_than_onces}')
                
                for j, (text, img_path) in enumerate(zip(batch[1][0], batch[0][0])):
                    prediction_result = {
                        'id': i * args.batchsize + j,
                        'correct': r1[j],
                        'text': text,
                        'img_path': img_path,
                    }
                    prediction_results.append(prediction_result)

                with open(f'./complementary_analysis/{args.run_id}_predictions.json', 'w') as f:
                    json.dump(prediction_results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    # parser.add_argument('--similarity', type=str, default='clip')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0, help='number of batches to skip\nuse: skip if i < args.skip\ni.e. put 49 if you mean 50')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--sampling_steps', type=int, default=250)
    parser.add_argument('--img_retrieval', action='store_true')
    parser.add_argument('--gray_baseline', action='store_true')
    parser.add_argument('--version', type=str, default='2.1')
    parser.add_argument('--lora_dir', type=str, default='')
    parser.add_argument('--guidance_scale', type=float, default=0.0)
    parser.add_argument('--targets', type=str, nargs='*', help="which target groups for mmbias",default='')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)

    if args.lora_dir:
        if 'mixed' in args.lora_dir:
            lora_type = 'mixed'
        elif 'LONGER' in args.lora_dir:
            lora_type = 'vanilla_LONGER'
        elif 'randneg' in args.lora_dir:
            lora_type = 'randneg'
        elif 'hardimgneg' in args.lora_dir:
            lora_type = 'hardimgneg'
        elif 'hardneg1.0' in args.lora_dir:
            lora_type = "hard_neg1.0"
        elif 'vanilla_coco' in args.lora_dir:
            lora_type = "vanilla_coco"
        elif "unhinged" in args.lora_dir:
            lora_type = "unhinged_hard_neg"
        elif "vanilla" in args.lora_dir:
            lora_type = "vanilla"
        elif "relativistic" in args.lora_dir:
            lora_type = "relativistic"
        elif "inferencelike" in args.lora_dir:
            lora_type = "inferencelike"

    args.run_id = f'{args.task}_diffusion_classifier_{args.version}_seed{args.seed}_steps{args.sampling_steps}_subset{args.subset}{args.targets}_img_retrieval{args.img_retrieval}_{"lora_" + lora_type if args.lora_dir else ""}_gray{args.gray_baseline}'
    if args.cache:
        args.cache_dir = f'./cache/{args.run_id}'
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
            
    main(args)