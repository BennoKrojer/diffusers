from glob import glob
import os
import random
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from score_winoground import score

random.seed(2022)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
vae.to(device=device,)# dtype=torch.bfloat16)

winoground_dir = '/home/krojerb/img-gen-project/diffusers/winoground/'
latent_dir = os.path.join(winoground_dir,'captions_latent')
images_dir = os.path.join(winoground_dir,'images_latent')
samples = random.sample(range(400), 20)
print(samples)


metric = torch.nn.CosineSimilarity(dim=0)
scores = {}
for i in tqdm(range(400)):
    for j in [0,1]:
        file = os.path.join(latent_dir,f"ex{i}_caption{j}_latent.pt")
        fname = f"ex{i}_caption{j}_generated.png"
        latent = torch.load(file).to(device)
        image = vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        im = to_pil_image(image[0]) 
        img_save_path = os.path.join(os.path.join(winoground_dir,'all_generated_images'),fname)
        im.save(img_save_path)
        
    # score the images and print result
    c0_i0, c0_i1, c1_i0, c1_i1 = score(i,
                                        metric='cos',
                                        caption_path=latent_dir,
                                        image_path=images_dir) 
    scores[i] = {'c0_i0':c0_i0,'c0_i1':c0_i1,'c1_i0':c1_i0,'c1_i1':c1_i1}
print(scores)