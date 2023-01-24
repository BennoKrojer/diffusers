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

from dataset.datasets import get_dataset
from torch.utils.data import DataLoader

from src.diffusers import StableDiffusionText2Latentmodelline, StableDiffusionImg2Latentmodelline

class Scorer:
    def __init__(self, args):
        self.metric = args.metric
        if self.metric == 'clip':
            self.clip_model, self.preprocess = clip.load('ViT-L/14@336px', device='cuda:0')


    def score_batch(self, args, batch, model):
        imgs, texts = batch[0], batch[1]

        scores = []

        for text in texts:
            if args.txt2img:
                img, latent = model(prompt=text)
                
            else:
                result_imgs = []
                for img in imgs:
                    img, latent = model(prompt=text, init_image=img)
                    result_imgs.append((img, latent))
                result[text] = result_imgs
        
        return result
        
        
    # img00, latent00 = model(prompt=cap0, init_image=img0, strength=args.strength, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
    # img01, latent01 = model(prompt=cap0, init_image=img1, strength=args.strength, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
    # img10, latent10 = model(prompt=cap1, init_image=img0, strength=args.strength, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
    # img11, latent11 = model(prompt=cap1, init_image=img1, strength=args.strength, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)

    # image0 = preprocess(image0).unsqueeze(0).cuda()
    # image1 = preprocess(image1).unsqueeze(0).cuda()
    # with torch.no_grad():
    #     caption_latent0 = clip_model.encode_image(image0).squeeze().float()
    #     caption_latent1 = clip_model.encode_image(image1).squeeze().float()

    # original_image0 = Image.open(os.path.join('winoground/images', f'ex_{i}_img_0.png'))
    # original_image1 = Image.open(os.path.join('winoground/images', f'ex_{i}_img_1.png')) 
    # original_image0 = preprocess(original_image0).unsqueeze(0).cuda()
    # original_image1 = preprocess(original_image1).unsqueeze(0).cuda()
    # with torch.no_grad():
    #     img_latent0 = clip_model.encode_image(original_image0).squeeze().float()
    #     img_latent1 = clip_model.encode_image(original_image1).squeeze().float()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda:0')

    model_id_or_path = "./stable-diffusion-v1-5"
    if args.txt2img:
        model = StableDiffusionPipeline.from_pretrained(
            model_id_or_path,
            safety_checker=None
        )
    else:
        model = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id_or_path,
            safety_checker=None
        )
    model = model.to(device)

    dataset = get_dataset(args.task, f'datasets/{args.task}')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    scorer = Scorer(args)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        scorer.process_batch(args, batch, clip_model, preprocess)

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--img2img', action='store_true')
    parser.add_argument('--similarity', type=str, default='clip')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache_dir',type=str)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args)