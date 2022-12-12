import torch
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from glob import glob
import os
from PIL import Image
from tqdm.auto import tqdm
import argparse
import json
import random


from src.diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir_captions',type=str, default='flickr/img2img_captions_latent')
parser.add_argument('--cuda_id',type=int, default=0)
parser.add_argument('--guidance_scale',type=float, default=7.5)
parser.add_argument('--strength', type=float, default=0.8)
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--txt2img', action='store_true')
parser.add_argument('--subset', action='store_true')
parser.add_argument('--parallel', action='store_true')
args = parser.parse_args()  

#set seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


args.save_dir_captions = f'{args.save_dir_captions}_guidance_scale_{args.guidance_scale}_strength_{args.strength}_steps_{args.num_inference_steps}_{"txt2img_" if args.txt2img else ""}seed_{args.seed}'

if not os.path.exists(args.save_dir_captions):
    os.makedirs(args.save_dir_captions)

device = torch.device(f'cuda:{args.cuda_id}')

# initialize pipeline
model_id_or_path = "./stable-diffusion-v1-5"
if args.txt2img:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id_or_path,
        safety_checker=None
    )
else:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        safety_checker=None
    )
pipe = pipe.to(device)

base_dir = 'flickr30'
img_dir = base_dir + '/images/'
top10 = json.load(open('flickr30/top10_RN50x64.json', 'r'))

for i, (caption, (_, all_imgs)) in tqdm(enumerate(top10.items()), total=len(top10)):
    if args.subset and i > 160:
        break
    if args.parallel and i % 8 != args.cuda_id:
        continue
    if args.txt2img:
        if not os.path.exists(f'{args.save_dir_captions}/{i}.pt'):
            generated, latent = pipe(caption, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
            torch.save(latent, f'{args.save_dir_captions}/{i}.pt')
            generated.images[0].save(f'{args.save_dir_captions}/{i}.png')
    if not args.txt2img:
        for j in range(10):
            img_id = all_imgs[j].split('/')[-1].split('.')[0]
            if not os.path.exists(f'{args.save_dir_captions}/{i}_{img_id}.pt'):
                img = Image.open(all_imgs[j]).convert('RGB').resize((512, 512))
                generated, latent = pipe(prompt=caption, init_image=img, strength=args.strength, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps)
                torch.save(latent, f'{args.save_dir_captions}/{i}_{img_id}.pt')
                generated.images[0].save(f'{args.save_dir_captions}/{i}_{img_id}.png')