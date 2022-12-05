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
import random

from src.diffusers import StableDiffusionImg2ImgPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir_captions',type=str, default='winoground/img2img_captions_latent')
parser.add_argument('--guidance_scale',type=float, default=7.5)
parser.add_argument('--strength', type=float, default=0.8)
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = torch.device(f'cuda:{args.seed}')

args.save_dir_captions = f'{args.save_dir_captions}_guidance_scale_{args.guidance_scale}_strength_{args.strength}_steps_{args.num_inference_steps}_seed_{args.seed}'
if not os.path.exists(args.save_dir_captions):
    os.makedirs(args.save_dir_captions)
#set seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# initialize pipeline
model_id_or_path = "./stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    safety_checker=None
    # revision="fp16", 
    # torch_dtype=torch.float16,
) # TODO try just modifying their code and saving the latents

# or download via git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
# and pass `model_id_or_path="./stable-diffusion-v1-5"`.
pipe = pipe.to(device)


data = json.load(open('winoground/data.json', 'r'))

for i, ex in tqdm(enumerate(data), total=len(data)):

    cap0 = ex['caption_0']
    cap1 = ex['caption_1']
    img_id = ex['id']
    img_path0 = f'winoground/images/ex_{img_id}_img_0.png'
    img_path1 = f'winoground/images/ex_{img_id}_img_1.png'

    img0 = Image.open(img_path0).convert("RGB")
    img1 = Image.open(img_path1).convert("RGB")

    img0 = img0.resize((512, 512))
    img1 = img1.resize((512, 512))
    
    if not os.path.exists(f'{args.save_dir_captions}/ex_{img_id}_img_0_cap_0.png'):
        img00, latent00 = pipe(prompt=cap0, init_image=img0, strength=args.strength, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
        torch.save(latent00, f"{args.save_dir_captions}/ex_{img_id}_latent_0_cap_0.pt")
        img00.images[0].save(f"{args.save_dir_captions}/ex_{img_id}_img_0_cap_0.png")

    if not os.path.exists(f'{args.save_dir_captions}/ex_{img_id}_img_1_cap_0.png'):
        img01, latent01 = pipe(prompt=cap0, init_image=img1, strength=args.strength, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
        torch.save(latent01, f"{args.save_dir_captions}/ex_{img_id}_latent_1_cap_0.pt")
        img01.images[0].save(f"{args.save_dir_captions}/ex_{img_id}_img_1_cap_0.png")
    
    if not os.path.exists(f'{args.save_dir_captions}/ex_{img_id}_img_0_cap_1.png'):
        img10, latent10 = pipe(prompt=cap1, init_image=img0, strength=args.strength, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
        torch.save(latent10, f"{args.save_dir_captions}/ex_{img_id}_latent_0_cap_1.pt")
        img10.images[0].save(f"{args.save_dir_captions}/ex_{img_id}_img_0_cap_1.png")

    if not os.path.exists(f'{args.save_dir_captions}/ex_{img_id}_img_1_cap_1.png'):
        img11, latent11 = pipe(prompt=cap1, init_image=img1, strength=args.strength, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
        torch.save(latent11, f"{args.save_dir_captions}/ex_{img_id}_latent_1_cap_1.pt")
        img11.images[0].save(f"{args.save_dir_captions}/ex_{img_id}_img_1_cap_1.png")