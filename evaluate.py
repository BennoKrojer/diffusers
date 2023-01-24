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
from utils import evaluate_scores

from src.diffusers import StableDiffusionText2Latentmodelline, StableDiffusionImg2Latentmodelline

class Scorer:
    def __init__(self, args):
        self.metric = args.metric
        if self.metric == 'clip':
            self.clip_model, self.preprocess = clip.load('ViT-L/14@336px', device='cuda:0')
        else:
            vae = AutoencoderKL.from_pretrained('/stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
            vae.to(device='cuda', dtype=torch.bfloat16)
            self.vae_model = StableDiffusionImg2LatentPipeline(vae).to('cuda')

    def score_batch(self, args, batch, model, vae_model):
        """
        Takes a batch of images and captions and returns a score for each image-caption pair.
        """

        imgs, texts = batch[0], batch[1]

        scores = []

        for text in texts:
            if args.txt2img:
                gen, latent = model(prompt=text)
            for img in imgs:
                if not args.txt2img:
                    gen, latent = model(prompt=text, init_image=img)
                score = self.score_pair(img, gen, latent)
                scores.append(score)

        return scores
        
    def score_pair(self, img, gen, latent):
        """
        Takes a batch of images and a batch of generated images and returns a score for each image pair.
        """
        if self.metric == 'clip':
            img = self.preprocess(img).unsqueeze(0).cuda()
            gen = self.preprocess(gen).unsqueeze(0).cuda()
            with torch.no_grad():
                img_latent = self.clip_model.encode_image(img).squeeze().float()
                gen_latent = self.clip_model.encode_image(gen).squeeze().float()
        else:
            gen_latent = latent
            img_latent = self.vae_model(img)

        score = torch.dot(img_latent - gen_latent, img_latent - gen_latent)
        score = - score # lower is better since it is a similarity metric
        return score



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        scores = scorer.process_batch(args, batch, model, vae_model)
        evaluate_scores(args, scores, batch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--img2img', action='store_true')
    parser.add_argument('--similarity', type=str, default='clip')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache_out_dir',type=str, default='')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args)