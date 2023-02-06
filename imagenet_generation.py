import torch
from torchvision.transforms.functional import to_pil_image
# from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import glob
import os
from PIL import Image
from tqdm.auto import tqdm
import argparse
import json
# import clip
import random

from datasets import get_dataset
from torch.utils.data import DataLoader
from utils import evaluate_scores

from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

# import cProfile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

classes = json.load(open('imagenet_classes.json')) # dict

model_id_or_path = "./stable-diffusion-v1-5"
model = StableDiffusionPipeline.from_pretrained(
    model_id_or_path,
    safety_checker=None
    )        

model = model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./.cache/imagenet/generated_imgs/')
    
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for i, text in classes.items():
        print(i)
        gen, latent = model(prompt=list(text))
        gen = gen.images
        torch.save(latent, f'{args.save_dir}/{i}.pt')
        gen.save(f'{args.save_dir}/{i}.png')
    
    