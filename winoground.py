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


from src.diffusers import StableDiffusionText2LatentPipeline


parser = argparse.ArgumentParser()
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
pipe = StableDiffusionText2LatentPipeline(
    vae,
    text_encoder,
    tokenizer,
    unet,
    scheduler
    )

pipe = pipe.to(device)

data = json.load(open('winoground/data.json', 'r'))

for i, ex in enumerate(data):
    cap0 = ex['caption_0']
    cap1 = ex['caption_1']

    latent0 = pipe(cap0)
    latent1 = pipe(cap1)
    
    # 4. save the latent representation
    save_path0 = os.path.join(args.save_dir,f'ex{i}_caption0_latent.pt')
    torch.save(latent0, save_path0)
    save_path1 = os.path.join(args.save_dir,f'ex{i}_caption1_latent.pt')
    torch.save(latent0, save_path1)