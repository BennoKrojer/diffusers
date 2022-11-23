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
import time


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
if args.prompt:
    prompts = [args.prompt]
else:
    prompts = ['archery in space','silly dog','flying unicorn']*10000

# TODO once max batch size is found, process text in batches of that sieze
    
batch_size = args.batch_size if args.batch_size else 1

def batchify(prompts, batch_size=1):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(prompts), batch_size):
        yield prompts[i:i + batch_size]

print('processing prompts')
# 2. process each image into torch.FloatTensor or PIL.Image.Image
times = 0
i=0
batches = batchify(prompts, batch_size)
for batch in tqdm(batches):
    dictionary = {}
        
    # 3. run pipeline on each query
    start_time = time.time()
    latents = pipe(batch)
    tot_time = time.time() - start_time
    if len(batch) == batch_size:
        times += tot_time
    print(f'batch {i} of size {len(batch)} took {tot_time} seconds.')
    # output shape is (batch_size,4,*,*)
    
    # 4. save the latent representation
    if args.save_dir:
        for i, prompt in enumerate(batch):
            fname = prompt.replace(' ','_')
            save_path = os.path.join(args.save_dir,f'{fname}_latent.pt')
            latent_img = latents[i]
            torch.save(latent_img, save_path)
    i += 1
    if i > 5:
        break
print(f"average batch time is {times/i}.")
