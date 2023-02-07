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
import random

from datasets import get_dataset
from torch.utils.data import DataLoader
from utils import evaluate_scores

from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./cache/imagenet/txt2img')
    
    args = parser.parse_args()

    args.save_dir = f'{args.save_dir}_seed_{args.seed}'

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    classes = json.load(open('imagenet_classes.json')) # dict
    json.dump(classes, open('imagenet_classes.json', 'w'), indent=4)

    model_id_or_path = "./stable-diffusion-v1-5"
    model = StableDiffusionPipeline.from_pretrained(
        model_id_or_path,
        safety_checker=None
        )        

    model = model.to(device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for i, text in classes.items():
        # text = 'a photo of a ' + text
        print(text)
        if f'{args.save_dir}/{i}.png' in glob.glob(f'{args.save_dir}/*.png'):
            continue
        gen, latent = model(prompt=text)
        gen = gen.images[0]
        # torch.save(latent, f'{args.save_dir}/{i}.pt')
        gen.save(f'{args.save_dir}/{i}.png')