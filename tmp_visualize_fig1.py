import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline
import glob
import os
from PIL import Image
from tqdm import tqdm
import argparse
import json
import random

from datasets_loading import get_dataset
from torch.utils.data import DataLoader
from utils import evaluate_scores
import csv
from accelerate import Accelerator

import cProfile
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

import matplotlib.font_manager as fm

# List of preferred alternative fonts to "Times New Roman"
alternative_fonts = ["Palatino", "Georgia", "Garamond", "TeX Gyre Heros"]

available_fonts = []
for font in fm.findSystemFonts():
    try:
        available_fonts.append(fm.get_font(font).family_name)
    except RuntimeError:
        pass
print(available_fonts)
# Check if each alternative font is available
for alt_font in alternative_fonts:
    if alt_font in available_fonts:
        print(f"Alternative font found: {alt_font}")
        mpl.rcParams['font.family'] = alt_font
        break
else:
    print("No alternative fonts found, using the default font.")


# Set the general style of seaborn (white background with gridlines)
sns.set(style="whitegrid")

class Scorer:
    def __init__(self, args, clip_model=None, preprocess=None):

        # vae = AutoencoderKL.from_pretrained('./stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
        # vae.to(device='cuda', dtype=torch.bfloat16)
        # self.vae_model = StableDiffusionImg2LatentPipeline(vae).to('cuda')
        self.cache_dir = args.cache_dir if args.cache else None

        flickr_stuff = json.load(open('datasets/flickr30k/val_top10_RN50x64.json'))
        self.texts = flickr_stuff.keys()
        # get 50 random texts
        self.texts = random.sample(self.texts, 10)
        print(self.texts)


    def score_batch(self, idx, args, batch, model):
        """
        Takes a batch of images and captions and returns a score for each image-caption pair.
        """

        imgs, texts = batch[0], batch[1]
        imgs, imgs_resize = imgs[0], imgs[1]
        imgs_resize =[img.to('cuda:7') for img in imgs_resize]

        scores = []
        for img_idx, resized_img in enumerate(imgs_resize[:2]):
            if len(resized_img.shape) == 3:
                resized_img = resized_img.unsqueeze(0)
            intermediate_scores = []
            for txt_idx, text in enumerate(texts + self.texts): 
                if txt_idx > 0:
                    text = [text]   
                print(f'Batch {idx}, Text {txt_idx}, Image {img_idx}')
                dists = model(prompt=list(text), image=resized_img, scoring=True, guidance_scale=0.0, sampling_steps=args.sampling_steps, unconditional=False, gray_baseline=args.gray_baseline)
                dists = dists.to(torch.float32)
                intermediate_scores.append(dists.squeeze()[2].detach().cpu())
            uncond_dist = model(prompt=list(text), image=resized_img, scoring=True, guidance_scale=0.0, sampling_steps=args.sampling_steps, weird_thing=True, gray_baseline=args.gray_baseline)
            uncond_dist = uncond_dist.to(torch.float32)
            intermediate_scores.append(uncond_dist.squeeze()[2].detach().cpu())
            scores.append(intermediate_scores)
        data = np.array(scores)

        x_labels = ['Image 1' if i == 0 else 'Image 2' for i in range(data.shape[0])]

        fig, ax = plt.subplots(figsize=(15, 3))

        palette = sns.color_palette("colorblind")

        separator = 0.5

        legend_elements = []

        for i in range(data.shape[0]):
            y = data[i, :]

            colors = [palette[0]] * len(y)
            colors[0] = palette[1]
            colors[-1] = palette[2]

            markers = ['o'] * len(y)
            markers[0] = '*'
            markers[-1] = '^'

            y_coordinates = [i * separator] * len(y)
            if i == 0:
                y_coordinates = [y - 0.6 for y in y_coordinates]
            else:
                y_coordinates = [y for y in y_coordinates]

            for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11]:
                if j == 0 or j == len(y) - 1:
                    size = 600
                else:
                    size = 100
                ax.scatter(y[j], y_coordinates[j], color=colors[j], marker=markers[j], s=size)

                if j == 0 and i == 0:
                    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', label='caption 1', markerfacecolor=colors[j], markersize=18))
                elif j == len(y) - 1 and i == 0:
                    legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', label='unconditional', markerfacecolor=colors[j], markersize=18))
                elif j == 1 and i == 0:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='random captions', markerfacecolor=colors[j], markersize=18))

        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=-1)
        ax.set_xlabel('Denoising Error', fontsize=14)
        ax.text(min(ax.get_xlim()) + 2, -separator / 2, 'Image 1', fontsize=25, verticalalignment='center')
        ax.text(min(ax.get_xlim()) + 0.6, separator / 2, 'Image 2', fontsize=25, verticalalignment='center')

        # Add legend
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=16)

        fig.tight_layout()

        plt.savefig(f'1D_scatter_plots_samefig.pdf', dpi=300, bbox_inches='tight')


def main(args):

    # accelerator = Accelerator()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler)
    model = model.to('cuda:7')
    if args.lora_dir != '':
        model.unet.load_attn_procs(args.lora_dir)

    scorer = Scorer(args)
    dataset = get_dataset(args.task, f'datasets/{args.task}', transform=None)

    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=0)

    # model, dataloader = accelerator.prepare(model, dataloader)

    r1s = []
    r5s = []
    max_more_than_onces = 0
    metrics = []
    ids = []
    clevr_dict = {}
    bias_scores = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        scores = scorer.score_batch(i, args, batch, model)
        break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    # parser.add_argument('--similarity', type=str, default='clip')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--sampling_steps', type=int, default=4)
    parser.add_argument('--img_retrieval', action='store_true')
    parser.add_argument('--gray_baseline', action='store_true')
    parser.add_argument('--lora_dir', type=str, default='')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)

    if args.lora_dir:
        if "vanilla" in args.lora_dir:
            lora_type = "vanilla"
        elif "relativistic" in args.lora_dir:
            lora_type = "relativistic"
        elif "inferencelike" in args.lora_dir:
            lora_type = "inferencelike"

    args.run_id = f'{args.task}_diffusion_classifier_seed{args.seed}_steps{args.sampling_steps}_subset{args.subset}_img_retrieval{args.img_retrieval}_{"lora_" + lora_type if args.lora_dir else ""}'

    if args.cache:
        args.cache_dir = f'./cache/{args.run_id}'
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
            
    main(args)