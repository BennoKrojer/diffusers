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


from src.diffusers import StableDiffusionText2LatentPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',type=str)
parser.add_argument('--prompt',type=str)
parser.add_argument('--batch_size',type=int)
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


# 1. get prompts passed as arg
prompts = [args.prompt]

# TODO once max batch size is found, process text in batches of that sieze
    
batch_size = args.batch_size if args.batch_size else 1

def batchify(prompts, batch_size=1):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(prompts), batch_size):
        yield prompts[i:i + batch_size]

print('processing prompts')
# 2. process each image into torch.FloatTensor or PIL.Image.Image
for batch in tqdm(batchify(prompts)):
    dictionary = {}
        
    # 3. run pipeline on each query
    latents = pipe(batch)
    # output shape is (batch_size,4,*,*)
    
    # 4. save the latent representation
    for i, prompt in enumerate(batch):
        fname = prompt.replace(' ','_')
        save_path = os.path.join(args.save_dir,f'{fname}_latent.pt')
        latent_img = latents[i]
        torch.save(latent_img, save_path)
