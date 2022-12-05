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


from src.diffusers import StableDiffusionImg2ImgPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir_captions',type=str, default='imagecode/img2img_captions_latent')
parser.add_argument('--cuda_id',type=int, default=0)
parser.add_argument('--guidance_scale',type=float, default=7.5)
parser.add_argument('--strength', type=float, default=0.6)
parser.add_argument('--num_inference_steps', type=int, default=50)

args = parser.parse_args()

args.save_dir_captions = f'{args.save_dir_captions}_guidance_scale_{args.guidance_scale}_strength_{args.strength}_steps_{args.num_inference_steps}'

if not os.path.exists(args.save_dir_captions):
    os.makedirs(args.save_dir_captions)

device = torch.device(f'cuda:{args.cuda_id}')

# initialize pipeline
model_id_or_path = "./stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    safety_checker=None
)
pipe = pipe.to(device)

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