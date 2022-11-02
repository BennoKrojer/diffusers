import torch
# from torchvision.transforms.functional import to_pil_image
# from transformers import CLIPTextModel, CLIPTokenizer
# from transformers import logging
from diffusers import AutoencoderKL#, UNet2DConditionModel, PNDMScheduler
import glob
import os
from PIL import Image
from tqdm.auto import tqdm
import argparse


from src.diffusers import StableDiffusionPipeline, StableDiffusionImg2LatentPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',type=str)
parser.add_argument('--image_dir',type=str,help="Full path to directory containing jpeg images we want to encode")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize inner models
vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
# tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
# text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
# unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet', use_auth_token='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')
# scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)

# to gpu
vae.to(device='cuda', dtype=torch.bfloat16)
# text_encoder.to(device='cuda', dtype=torch.bfloat16)
# unet.to(device='cuda', dtype=torch.bfloat16)


# initialize pipeline
pipe = StableDiffusionImg2LatentPipeline(
    vae,
    # text_encoder,
    # tokenizer,
    # unet,
    # scheduler
    )

pipe = pipe.to(device)


# process images

# 1. get all images from a folder passed as arg
paths = glob.glob(f"{args.image_dir}/*")

# 2. process each image into torch.FloatTensor or PIL.Image.Image
for img_path in tqdm(paths):
    print(img_path)
    fname = img_path.split('/')[-1].split('.')[0]
    print(fname)
    img = Image.open(img_path).convert("RGB").resize((512,512))
    
    # 3. run pipeline on each image
    latent = pipe(img)
    
    # 4. save the latent representations 
    print(type(latent))
    print(latent.shape)
    print(latent)
    if args.save_dir:
        save_path = os.path.join(args.save_dir,f'{fname}_latent.pt')
        torch.save(latent, save_path)