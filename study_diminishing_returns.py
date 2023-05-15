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
import matplotlib.pyplot as plt

SAMPLING = 500

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

        scores = {k: [] for k in range(5,SAMPLING,5)}
        for txt_idx, text in enumerate(texts):
            for img_idx, resized_img in enumerate(imgs_resize):
                if len(resized_img.shape) == 3:
                    resized_img = resized_img.unsqueeze(0)
                
                print(f'Batch {i}, Text {txt_idx}, Image {img_idx}')
                dists = model(prompt=list(text), image=resized_img, scoring=True, guidance_scale=0.0, sampling_steps=SAMPLING, unconditional=args.img_retrieval, gray_baseline=args.gray_baseline)
                dists = dists.to(torch.float32)
                for sample_size in range(5,SAMPLING,5):
                    indices = torch.linspace(0, dists.shape[1]-1, steps=sample_size).long()
                    sub_dists = dists[:, indices].mean(dim=1)
                    sub_dists = -sub_dists
                    scores[sample_size].append(sub_dists)
                # dists = dists.mean(dim=1)
                # dists = -dists
                # scores.append(dists)
        for k in scores.keys():
            scores[k] = torch.stack(scores[k]).permute(1, 0) if args.batchsize > 1 else torch.stack(scores[k]).unsqueeze(0)
        # scores = torch.stack(scores).permute(1, 0) if args.batchsize > 1 else torch.stack(scores).unsqueeze(0)
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

    r1s = {k:[] for k in range(5,SAMPLING,5)}
    r5s = {k:[] for k in range(5,SAMPLING,5)}
    max_more_than_onces = 0
    metrics = []
    ids = []
    clevr_dict = {}
    bias_scores = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i % 8 != 0:
            continue
        scores = scorer.score_batch(i, args, batch, model)
        new_scores = {}
        for k in scores.keys():
            new_scores[k] = scores[k].contiguous()
            accelerator.wait_for_everyone()
            new_scores[k] = accelerator.gather(new_scores[k])
        scores = new_scores
        batch[-1] = accelerator.gather(batch[-1])

        # scores = scores_.contiguous()
        # accelerator.wait_for_everyone()
        # # print(scores)
        # scores_ = accelerator.gather(scores_)
        # batch[-1] = accelerator.gather(batch[-1])
        if True:
            for k in scores.keys():
                scores_ = scores[k]
                r1,r5, max_more_than_once = evaluate_scores(args, scores_, batch)
                r1s[k] += r1
                r5s[k] += r5
                # max_more_than_onces += max_more_than_once
                # r1 = sum(r1s) / len(r1s)
                # r5 = sum(r5s) / len(r5s)
                # print(f'R@1: {r1}')
                # print(f'R@5: {r5}')
                # print(f'Max more than once: {max_more_than_onces}')
                # with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
                #     f.write(f'R@1: {r1}\n')
                #     f.write(f'R@5: {r5}\n')
                #     f.write(f'Max more than once: {max_more_than_onces}\n')
                #     f.write(f"Sample size {len(r1s)}\n")

    # plot r1 and r5 for each sample size
        if True:
            plt.clf()  # Clear the figure
            r1_points = []
            r5_points = []
            for k in r1s.keys():
                r1 = sum(r1s[k]) / len(r1s[k])
                r5 = sum(r5s[k]) / len(r5s[k])
                r1_points.append((k, r1))
                r5_points.append((k, r5))

            # Sort points by keys for correct line plotting
            r1_points.sort(key=lambda x: x[0])
            r5_points.sort(key=lambda x: x[0])

            # Unzip the points for plotting
            r1_x, r1_y = zip(*r1_points)
            r5_x, r5_y = zip(*r5_points)

            # plot it
            plt.plot(r1_x, r1_y, 'r-')
            plt.plot(r5_x, r5_y, 'b-')
            plt.xlabel('Sample size')
            plt.ylabel('R@1 and R@5')
            plt.legend(['R@1', 'R@5'])
            plt.savefig(f'diminsh_returns_r1_r5.png')

            

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
    parser.add_argument('--targets', type=str, nargs='*', help="which target groups for mmbias",default='')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)

    if args.lora_dir:
        if 'hardimgneg' in args.lora_dir:
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

    args.run_id = f'{args.task}_diffusion_classifier_{args.version}_seed{args.seed}_steps{args.sampling_steps}_subset{args.subset}{args.targets}_img_retrieval{args.img_retrieval}_{"lora_" + lora_type if args.lora_dir else ""}'
    if args.cache:
        args.cache_dir = f'./cache/{args.run_id}'
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
            
    main(args)