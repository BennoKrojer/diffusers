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
import numpy as np
from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('initializing inner models')
vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
vae.to(device='cuda', dtype=torch.bfloat16)

pipe_img = StableDiffusionImg2LatentPipeline(vae)

pipe = pipe_img.to(device)

base_dir = '/home/krojerb/scratch/flickr30k_images/'

for path in glob(base_dir + 'flickr30k_images/*'):
    img_id = path.split('/')[-1].split('.')[0]
    img = Image.open(path).convert("RGB").resize((512,512))
    latent0_img = pipe_img(img0)
    save_path = os.path.join(base_dir, 'latent_images', img_id+'.pt')
    torch.save(latent0_img, save_path)

data = json.load(open('imagecode/valid_data.json', 'r'))

for i, (set_id, descriptions) in tqdm(enumerate(data.items()), total=len(data)):
    if i % 8 != args.cuda_id:
        continue
    print("RUNNING SET: ", i, set_id)
    for idx, description in descriptions.items():
        all_imgs = glob(f'./imagecode/image-sets/{set_id}/*.jpg')
        all_imgs = sorted(all_imgs, key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))
        for i in range(10):
            if not os.path.exists(f'{args.save_dir_captions}/{set_id}_{idx}_{i}.pt'):
                img = Image.open(all_imgs[i]).convert('RGB').resize((512, 512))
                generated, latent = pipe(prompt=description, init_image=img, strength=args.strength, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps)
                torch.save(latent, f'{args.save_dir_captions}/{set_id}_{idx}_{i}.pt')
                generated.images[0].save(f'{args.save_dir_captions}/{set_id}_{idx}_{i}.png')