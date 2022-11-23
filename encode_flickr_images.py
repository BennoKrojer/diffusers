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

# initialize inner models
print('initializing inner models')
vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet', use_auth_token='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')
scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)

# to gpu
vae.to(device='cuda', dtype=torch.bfloat16)
text_encoder.to(device='cuda', dtype=torch.bfloat16)
unet.to(device='cuda', dtype=torch.bfloat16)


pipe_img = StableDiffusionImg2LatentPipeline(vae)

pipe = pipe_img.to(device)

base_dir = '/home/krojerb/scratch/flickr30k_images/'

for path in glob(base_dir + 'flickr30k_images/*'):
    img_id = path.split('/')[-1].split('.')[0]
    img = Image.open(path).convert("RGB").resize((512,512))
    latent0_img = pipe_img(img0)
    save_path = os.path.join(base_dir, 'latent_images', img_id+'.pt')
    torch.save(latent0_img, save_path)