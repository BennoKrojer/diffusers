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
import numpy as np
from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reverse', action='store_true')
args = parser.parse_args()

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


# initialize pipeline
print('initializing StableDiffusionText2LatentPipeline')
pipe_text = StableDiffusionText2LatentPipeline(
    vae,
    text_encoder,
    tokenizer,
    unet,
    scheduler
    )
pipe_text = pipe_text.to(device)

base_dir = '/home/krojerb/scratch/flickr30k_images'
data = open(f'{base_dir}/annotations/valid_ann.jsonl', 'r')
batchsize  = 1

captions = []
for line in data:
    line = json.loads(line)
    sents = line['sentences']
    for i, sent in enumerate(sents):
        captions.append((sent, line['id']+'_'+str(i)))
print('Number of captions: ', len(captions))
if args.reverse:
    captions = captions[::-1]

# batchify
for i in tqdm(range(0, len(captions), batchsize)):
    batch = captions[i:i+batchsize]
    batch_c = [x[0] for x in batch]
    ids = [x[1] for x in batch]
    if os.path.exists(f'{base_dir}/captions/{ids[-1]}.npy'):
        continue
    print(f'processing example {i} to {i+batchsize}')
    latent = pipe_text(batch)
    for j, id in enumerate(ids):
        np.save(f'{base_dir}/captions/{id}.npy', latent[j].cpu().numpy())
        print("SAVING", f'{base_dir}/captions/{id}.npy')