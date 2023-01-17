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

from src.diffusers import StableDiffusionText2LatentPipeline, StableDiffusionImg2LatentPipeline


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id_or_path = "./stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        safety_checker=None
        # revision="fp16", 
        # torch_dtype=torch.float16,
    ) # TODO try just modifying their code and saving the latents

    # or download via git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
    # and pass `model_id_or_path="./stable-diffusion-v1-5"`.
    pipe = pipe.to(device)

    

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('img2img', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache_dir',type=str)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args)