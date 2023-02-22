import torch
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import glob
import os
from PIL import Image
from tqdm.auto import tqdm
import argparse
import json
import clip
import random

from datasets import get_dataset
from torch.utils.data import DataLoader
from utils import evaluate_scores
import csv

from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

import cProfile

class Scorer:
    def __init__(self, args, clip_model=None, preprocess=None, precomputed=None):
        self.similarity = args.similarity
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.cache_dir = args.cache_dir if args.cache else None
        self.precomputed = precomputed


    def score_batch(self, i, args, batch):
        """
        Takes a batch of images and captions and returns a score for each image-caption pair.
        """

        imgs, texts = batch[0], batch[1]
        imgs, imgs_resize = imgs[0], imgs[1]
        imgs, imgs_resize = [img.cuda() for img in imgs], [img.cuda() for img in imgs_resize]

        scores = []

        for txt_idx, text in enumerate(texts):
            if args.task == 'imagenet':
                gen = self.precomputed[txt_idx]
            elif args.task == 'winoground':
                gen = self.precomputed[i][txt_idx]
            if self.cache_dir:
                gen.save(f'{self.cache_dir}/{i}.png')
            for img_idx, img in enumerate(imgs):
                resized_img = imgs_resize[img_idx]
                score = self.score_pair(img, resized_img, gen)
                scores.append(score)

        scores = torch.stack(scores).permute(1, 0) if args.batchsize > 1 else torch.stack(scores).unsqueeze(0)
        return scores
        
    def score_pair(self, img, resized_img, gen):
        """
        Takes a batch of images and a batch of generated images and returns a score for each image pair.
        """
        if args.task in ['imagenet', 'winoground']:
            with torch.no_grad():
                img_latent = self.clip_model.encode_image(img).squeeze().float()
                gen_latent = gen
        else:
            gen = torch.stack([self.preprocess(g) for g in gen]).to('cuda')
            with torch.no_grad():
                img_latent = self.clip_model.encode_image(img).squeeze().float()
                gen_latent = self.clip_model.encode_image(gen).squeeze().float()
        
        diff = img_latent - gen_latent
        score = torch.norm(diff, p=2, dim=-1)
        score = - score # lower is better since it is a similarity similarity
        return score


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda')
    if args.task == 'imagenet':
        precomputed = []
        for i in range(1000):
            precomputed.append(torch.load(f'./cache/imagenet/txt2img_seed_{args.seed}/{i}.pt').to('cuda')) 
        scorer = Scorer(args, clip_model=clip_model, preprocess=preprocess, precomputed=precomputed)
    elif args.task == 'winoground':
        precomputed = []
        for i in range(400):
            if os.path.exists(f'./cache/winoground/dalle_txt2img/{i}_0_seed_{args.seed}.pt'):
                img0 = torch.load(f'./cache/winoground/dalle_txt2img/{i}_0_seed_{args.seed}.pt').to('cuda')
                img1 = torch.load(f'./cache/winoground/dalle_txt2img/{i}_1_seed_{args.seed}.pt').to('cuda')
                precomputed.append([img0, img1])
            else:
                img0 = torch.rand(precomputed[-1][0].shape).to('cuda')
                img1 = torch.rand(precomputed[-1][1].shape).to('cuda')
                precomputed.append([img0, img1])
        scorer = Scorer(args, clip_model=clip_model, preprocess=preprocess, precomputed=precomputed)
    else:
        scorer = Scorer(args, clip_model=clip_model, preprocess=preprocess)

    dataset = get_dataset(args.task, f'datasets/{args.task}', transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    
    metrics = []
    all_scores = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if args.subset and i % 10 != 0:
            continue
        scores = scorer.score_batch(i, args, batch)
        all_scores.append(list(scores.cpu().numpy()[0]))
        score = evaluate_scores(args, scores, batch)
        metrics.append(score)
        if args.task == 'winoground':
            text_score = sum([m[0] for m in metrics]) / len(metrics)
            img_score = sum([m[1] for m in metrics]) / len(metrics)
            group_score = sum([m[2] for m in metrics]) / len(metrics)
            print(f'Text score: {text_score}')
            print(f'Image score: {img_score}')
            print(f'Group score: {group_score}')
        else:
            accuracy = sum(metrics) / len(metrics)
            print(f'Retrieval Accuracy: {accuracy}')
    # write all scores to csv
    with open(f'./cache/{args.run_id}_scores.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(all_scores)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--img2img', action='store_true')
    parser.add_argument('--similarity', type=str, default='clip')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--strength', type=float, default=0.8)
    parser.add_argument('--subset', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)

    args.run_id = f'{args.task}/dalle_{"img2img" if args.img2img else "text2img"}_{args.similarity}_strength{args.strength}_seed{args.seed}'

    if args.cache:
        args.cache_dir = f'./cache/{args.run_id}'
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
            
    main(args)