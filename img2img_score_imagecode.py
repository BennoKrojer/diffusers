from PIL import Image
import torch
# import clip
import os
from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import json
from glob import glob
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_id_or_path = "./stable-diffusion-v1-5"
vae = AutoencoderKL.from_pretrained(model_id_or_path, subfolder='vae', use_auth_token='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')
vae.to(device='cuda', dtype=torch.bfloat16)
pipe_img = StableDiffusionImg2LatentPipeline(vae)
pipe = pipe_img.to(device)
USE_CLIP = True

# clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda:0')
if USE_CLIP:
    print("Loading CLIP")
    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda:0')

if __name__ == "__main__":
    from tqdm import tqdm

    METRIC = 'dot'

    data = json.load(open('datasets/imagecode/valid_data.json', 'r'))

    acc = 0
    total = 0
    
    for i, (set_id, descriptions) in tqdm(enumerate(data.items()), total=len(data)):
        print("RUNNING SET: ", i, set_id)
        if 'open-images' not in set_id:
            continue
        for idx, description in descriptions.items():
            idx = int(idx)
            all_imgs = glob(f'datasets/imagecode/image-sets/{set_id}/*.jpg')
            all_imgs = sorted(all_imgs, key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))
            # correct_img = all_imgs[idx]
            # correct_img = Image.open(correct_img).convert('RGB').resize((512, 512))
            # latent_correct = pipe_img(correct_img).flatten().float()
            smallest_dist = 100000000000
            best_i = 0
            dists = []
            for i in range(10):

                if USE_CLIP:
                    img = preprocess(Image.open(all_imgs[i]).convert('RGB')).unsqueeze(0).to(device)
                    with torch.no_grad():
                        latent_before = clip_model.encode_image(img).squeeze().float()
                else:
                    img = Image.open(all_imgs[i]).convert('RGB').resize((512, 512))
                    latent_before = pipe_img(img).flatten().float()

                if USE_CLIP:
                    img = Image.open(f'datasets/imagecode/img2img_captions_latent_guidance_scale_7.5/{set_id}_{idx}_{i}.png').convert('RGB')
                    img = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        latent_after = clip_model.encode_image(img).squeeze().float()
                else:        
                    latent_after = torch.load(f'datasets/imagecode/img2img_captions_latent_guidance_scale_7.5/{set_id}_{idx}_{i}.pt', map_location=torch.device(device)).flatten().float()
                
                dist = torch.dot(latent_after - latent_before, latent_after - latent_before)
                dists.append(dist)
                if i == 0 or smallest_dist > dist:
                    smallest_dist = dist
                    best_i = i
            acc += best_i == idx
            total += 1

            print(acc/total)
