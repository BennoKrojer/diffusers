from PIL import Image
import torch
import clip
import os
from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import json
from glob import glob
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_id_or_path = "./stable-diffusion-v1-5"
vae = AutoencoderKL.from_pretrained(model_id_or_path, subfolder='vae', use_auth_token='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')
vae.to(device='cuda')
pipe_img = StableDiffusionImg2LatentPipeline(vae)
pipe = pipe_img.to(device)
USE_CLIP = True

if USE_CLIP:
    print("Loading CLIP")
    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda:0')

if __name__ == "__main__":


    base_dir = 'datasets/flickr30'
    img_dir = base_dir + '/images/'
    top10 = json.load(open('datasets/flickr30k/top10_RN50x64.json', 'r'))
    experiment_name = 'img2img_captions_latent_guidance_scale_7.5_strength_0.8_steps_50_txt2img_seed_0'
    print('EVALUATING', experiment_name)
    acc = 0
    total = 0
    ACROSS_SEEDS = True
    AVG = False
    LATENT_AVG = False
    
    for i, (caption, (correct_path, all_images)) in tqdm(enumerate(top10.items()), total=len(top10.items())):
        correct_path = 'datasets/' + correct_path
        all_images = ['datasets/' + path for path in all_images]
        if correct_path not in all_images:
            total += 1
            continue
        correct_idx = all_images.index(correct_path)
        smallest_dist = 100000000000
        best_i = 0

        if USE_CLIP:
            image_after = Image.open(f'flickr/{experiment_name}/{i}.png').convert('RGB')
            image_after = preprocess(image_after).unsqueeze(0).to(device)
            with torch.no_grad():
                latent_after = clip_model.encode_image(image_after).squeeze().float()
        else:
            latent_after = torch.load(f'flickr/{experiment_name}/{i}.pt', map_location=torch.device(device)).flatten().float()
        for j in range(10):
            img_id = all_images[j].split('/')[-1].split('.')[0]
            img = Image.open(all_images[j]).convert('RGB')
            if USE_CLIP:
                img = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    latent_before = clip_model.encode_image(img).squeeze().float()
            else:
                img = img.resize((512, 512))
                latent_before = pipe_img(img).flatten().float()

            dist = torch.dot(latent_after - latent_before, latent_after - latent_before)
            if j == 0 or smallest_dist > dist:
                smallest_dist = dist
                best_i = j
        acc += best_i == correct_idx
        total += 1

        print(acc/total)
