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


from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir_captions',type=str, default='winoground/captions_latent')
parser.add_argument('--save_dir_images',type=str, default='winoground/images_latent')
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
 
pipe_img = StableDiffusionImg2LatentPipeline(vae)

pipe = pipe_img.to(device)

pipe_text = pipe_text.to(device)

data = json.load(open('winoground/data.json', 'r'))

for i, ex in tqdm(enumerate(data)):

    cap0 = ex['caption_0']
    cap1 = ex['caption_1']
    img_id = ex['id']
    # img_path0 = f'winoground/images/ex_{img_id}_img_0.png'
    # img_path1 = f'winoground/images/ex_{img_id}_img_1.png'

    # fname0 = img_path0.split('/')[-1].split('.')[0]
    # fname1 = img_path1.split('/')[-1].split('.')[0]
    # if os.path.exists(os.path.join(args.save_dir_images,f'{fname1}_latent.pt')):
    #     continue
    save_path1 = os.path.join(args.save_dir_captions,f'ex{i}_caption1_latent.pt')
    if os.path.exists(save_path1):
        continue

    # img0 = Image.open(img_path0).convert("RGB").resize((512,512))
    # img1 = Image.open(img_path1).convert("RGB").resize((512,512))

    # latent0_text = pipe_text(cap0)
    latent1_text = pipe_text(cap1)
    
    # save_path0 = os.path.join(args.save_dir_captions,f'ex{i}_caption0_latent.pt')
    # torch.save(latent0_text, save_path0)
    torch.save(latent1_text, save_path1)

    # latent0_img = pipe_img(img0)
    # latent1_img = pipe_img(img1)

    # save_path = os.path.join(args.save_dir_images,f'{fname0}_latent.pt')
    # torch.save(latent0_img, save_path)
    # save_path = os.path.join(args.save_dir_images,f'{fname1}_latent.pt')
    # torch.save(latent1_img, save_path)
    