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


from src.diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionTextImg2LatentPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize inner models
vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet', use_auth_token='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')
scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)

# to gpu
vae.to(device='cuda')#, dtype=torch.bfloat16)
text_encoder.to(device='cuda')#, dtype=torch.bfloat16)
unet.to(device='cuda')#, dtype=torch.bfloat16)


# initialize pipeline
pipe = StableDiffusionTextImg2LatentPipeline(
    vae,
    text_encoder,
    tokenizer,
    unet,
    scheduler,
    )

pipe = pipe.to(device)

# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(...)
# process images

img_path = '/home/krojerb/img-gen-project/diffusers/animals/sketch-mountains-input.jpeg'
print(img_path)
prompt = "A scary landscape"
fname = "img-"+img_path.split('/')[-1].split('.')[0]+"_strength=0.8"+"_prompt-"+prompt.replace(' ','_')
print(fname)
img = Image.open(img_path).convert("RGB").resize((768, 512))#.resize((512,512))

# 3. run pipeline on each prompt image pair
# original hpyerparams 
# strength: float = 0.8,
# num_inference_steps: Optional[int] = 50,
# guidance_scale: Optional[float] = 7.5,
# num_images_per_prompt: Optional[int] = 1,
# eta: Optional[float] = 0.0,
# output_type: Optional[str] = "pil",
# return_dict: bool = True,
# callback_steps: Optional[int] = 1,

latent = pipe(prompt, img, strength=0.8, guidance_scale=7.5)
img_save_path = os.path.join('/home/krojerb/img-gen-project/diffusers/generated_images',f'{fname}.png')
# images[0].save(img_save_path)

image = vae.decode(latent).sample
image = (image / 2 + 0.5).clamp(0, 1)
im = to_pil_image(image[0]) 
im.save(img_save_path)