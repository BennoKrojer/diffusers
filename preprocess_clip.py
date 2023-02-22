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
import clip
import random

from datasets import get_dataset
from torch.utils.data import DataLoader
from utils import evaluate_scores
import csv

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda')

    for img_path in tqdm(glob.glob(f'{args.dir}/*.png')):
        img = Image.open(img_path).convert('RGB')
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_latent = clip_model.encode_image(img).squeeze().float().cpu()
        torch.save(img_latent, img_path.replace('.png', '.pt'))
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--dir', type=str, default='cache/winoground/dalle_txt2img')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)

    main(args)