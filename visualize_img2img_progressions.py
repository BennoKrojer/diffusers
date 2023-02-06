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

from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

import cProfile

STARTER_IDX = 6

class Scorer:
    def __init__(self, args, clip_model=None, preprocess=None):
        self.similarity = args.similarity
        if self.similarity == 'clip':
            self.clip_model = clip_model
            self.preprocess = preprocess
        else:
            vae = AutoencoderKL.from_pretrained('./stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
            vae.to(device='cuda', dtype=torch.bfloat16)
            self.vae_model = StableDiffusionImg2LatentPipeline(vae).to('cuda')
        self.cache_dir = args.cache_dir if args.cache else None


    def score_batch(self, i, args, batch, model):
        """
        Takes a batch of images and captions and returns a score for each image-caption pair.
        """

        imgs, texts = batch[0], batch[1]
        imgs, imgs_resize = imgs[0], imgs[1]
        imgs, imgs_resize = [img.cuda() for img in imgs], [img.cuda() for img in imgs_resize]
        correct_idx = batch[-1]

        scores = []

        for text in texts:
            if not args.img2img:
                gen, latent = model(prompt=list(text))
                gen = gen.images
                if self.cache_dir:
                    torch.save(latent, f'{self.cache_dir}/{i}.pt')
                    gen.save(f'{self.cache_dir}/{i}.png')
            for img_idx, img in enumerate(imgs):
                if img_idx != STARTER_IDX: #TODO: DELETE THIS
                    continue
                resized_img = imgs_resize[img_idx]
                if args.img2img:
                    visualize_dir = f'analysis/img2img_noise_levels/{args.datapoint_id}_{"in" if img_idx != correct_idx else ""}correct_starter_img'
                    print('WRITING TO:', visualize_dir)
                    if not os.path.exists(visualize_dir):
                        os.makedirs(visualize_dir)
                    gen, latent = model(prompt=list(text), init_image=resized_img, strength=args.strength, num_inference_steps=args.num_inference_steps, visualize_dir=visualize_dir)
                    gen = gen.images
                    if self.cache_dir:
                        # torch.save(latent, f'{self.cache_dir}/{i}_{img_idx}.pt')
                        gen[0].save(f'{self.cache_dir}/{i}_{img_idx}.png')
                score = self.score_pair(img, resized_img, gen, latent)
                scores.append(score)

        scores = torch.stack(scores).permute(1, 0) if args.batchsize > 1 else torch.stack(scores).unsqueeze(0)
        return scores
        
    def score_pair(self, img, resized_img, gen, latent):
        """
        Takes a batch of images and a batch of generated images and returns a score for each image pair.
        """
        if self.similarity == 'clip':
            gen = torch.stack([self.preprocess(g) for g in gen]).to('cuda')
            with torch.no_grad():
                img_latent = self.clip_model.encode_image(img).squeeze().float()
                gen_latent = self.clip_model.encode_image(gen).squeeze().float()
        else:
            #flatten except first dimension
            gen_latent = latent.reshape(latent.shape[0], -1)
            img_latent = self.vae_model(resized_img).reshape(latent.shape[0], -1)     
        
        diff = img_latent - gen_latent
        score = torch.norm(diff, p=2, dim=-1)
        score = - score # lower is better since it is a similarity similarity
        return score


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id_or_path = "./stable-diffusion-v1-5"
    if args.img2img:
        model = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id_or_path,
            safety_checker=None
        )
    else:
        model = StableDiffusionPipeline.from_pretrained(
            model_id_or_path,
            safety_checker=None
        )        

    model = model.to(device)
    
    
    if args.similarity == 'clip':
        clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda')
        scorer = Scorer(args, clip_model=clip_model, preprocess=preprocess)

        dataset = get_dataset(args.task, f'datasets/{args.task}', transform=preprocess)
    else:
        scorer = Scorer(args)
        dataset = get_dataset(args.task, f'datasets/{args.task}', transform=None)

    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)


    metrics = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i > args.datapoint_id:
            break
        if i != args.datapoint_id: #TODO: DELETE THIS
            continue
        scores = scorer.score_batch(i, args, batch, model)
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
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--datapoint_id', type=int, default=0)
    args = parser.parse_args()


    if args.cache:
        args.cache_dir = f'./cache/{args.task}/{"img2img" if args.img2img else "text2img"}_{args.similarity}_seed{args.seed}'
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
    for strength in [0.8, 0.75, 0.6, 0.5, 0.4, 0.3]:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(args.cuda_device)

        args.strength = strength
        main(args)

    main(args)