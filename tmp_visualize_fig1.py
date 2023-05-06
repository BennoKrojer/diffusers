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


    def score_batch(self, i, args, batch, model):
        """
        Takes a batch of images and captions and returns a score for each image-caption pair.
        """

        imgs, texts = batch[0], batch[1]
        imgs, imgs_resize = imgs[0], imgs[1]
        imgs, imgs_resize = [img.to('cuda:7') for img in imgs], [img.to('cuda:7') for img in imgs_resize]

        scores = []
        for img_idx, resized_img in enumerate(imgs_resize[:5]):
            if len(resized_img.shape) == 3:
                resized_img = resized_img.unsqueeze(0)
            intermediate_scores = []
            for txt_idx, text in enumerate(texts + self.texts): 
                if txt_idx > 0:
                    text = [text]   
                print(f'Batch {i}, Text {txt_idx}, Image {img_idx}')
                dists = model(prompt=list(text), image=resized_img, scoring=True, guidance_scale=0.0, sampling_steps=args.sampling_steps, unconditional=False, gray_baseline=args.gray_baseline)
                dists = dists.to(torch.float32)
                intermediate_scores.append(dists.squeeze().detach().cpu())
            uncond_dist = model(prompt=list(text), image=resized_img, scoring=True, guidance_scale=0.0, sampling_steps=args.sampling_steps, unconditional=True, gray_baseline=args.gray_baseline)
            uncond_dist = uncond_dist.to(torch.float32)
            intermediate_scores.append(uncond_dist.squeeze().detach().cpu())
            scores.append(intermediate_scores)
        data = np.array(scores)

        x_labels = [f'Image {i+1}' for i in range(data.shape[0])]

        # Create 2x2 grid of scatter plots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            # Set colors for the scatter plot points
            colors = ['red'] * data.shape[1]
            colors[0] = 'green'
            colors[-1] = 'blue'
            ax.scatter(np.arange(data.shape[1]), data[i, :], color=colors)
            ax.set_xlabel('Caption Index')
            ax.set_ylabel('Denoising Error')
            ax.set_title(f'Denoising Errors for {x_labels[i]}')
            ax.set_ylim(data[i, :].min() - 0.1, data[i, :].max() + 0.1)

        fig.tight_layout()

        # Save the plot as a file (e.g. PNG format)
        plt.savefig('scatter_plots_2x2.png', dpi=300, bbox_inches='tight')


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
        if args.subset and i % 15 != 0:
            continue
        scores = scorer.score_batch(i, args, batch, model)
        scores = scores.contiguous()
        accelerator.wait_for_everyone()
        print(scores)
        scores = accelerator.gather(scores)
        batch[-1] = accelerator.gather(batch[-1])
        if accelerator.is_main_process:
            if args.task == 'winoground':
                text_scores, img_scores, group_scores = evaluate_scores(args, scores, batch)
                metrics += list(zip(text_scores, img_scores, group_scores))
                text_score = sum([m[0] for m in metrics]) / len(metrics)
                img_score = sum([m[1] for m in metrics]) / len(metrics)
                group_score = sum([m[2] for m in metrics]) / len(metrics)
                print(f'Text score: {text_score}')
                print(f'Image score: {img_score}')
                print(f'Group score: {group_score}')
                print(len(metrics))
                with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
                    f.write(f'Text score: {text_score}\n')
                    f.write(f'Image score: {img_score}\n')
                    f.write(f'Group score: {group_score}\n')
            elif args.task in ['flickr30k', 'imagecode', 'imagenet', 'flickr30k_text']:
                r1,r5, max_more_than_once = evaluate_scores(args, scores, batch)
                r1s += r1
                r5s += r5
                max_more_than_onces += max_more_than_once
                r1 = sum(r1s) / len(r1s)
                r5 = sum(r5s) / len(r5s)
                print(f'R@1: {r1}')
                print(f'R@5: {r5}')
                print(f'Max more than once: {max_more_than_onces}')
                with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
                    f.write(f'R@1: {r1}\n')
                    f.write(f'R@5: {r5}\n')
                    f.write(f'Max more than once: {max_more_than_onces}\n')
                    f.write(f"Sample size {len(r1s)}\n")
            elif args.task == 'clevr':
                acc_list, max_more_than_once = evaluate_scores(args, scores, batch)
                metrics += acc_list
                acc = sum(metrics) / len(metrics)
                max_more_than_onces += max_more_than_once
                print(f'Accuracy: {acc}')
                print(f'Max more than once: {max_more_than_onces}')
                with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
                    f.write(f'Accuracy: {acc}\n')
                    f.write(f'Max more than once: {max_more_than_onces}\n')
                    f.write(f"Sample size {len(metrics)}\n")

                # now do the same but for every subtask of CLEVR
                subtasks = batch[-2]
                for i, subtask in enumerate(subtasks):
                    if subtask not in clevr_dict:
                        clevr_dict[subtask] = []
                    clevr_dict[subtask].append(acc_list[i])
                for subtask in clevr_dict:
                    print(f'{subtask} accuracy: {sum(clevr_dict[subtask]) / len(clevr_dict[subtask])}')
                    with open(f'./paper_results/{args.run_id}_results.txt', 'a') as f:
                        f.write(f'{subtask} accuracy: {sum(clevr_dict[subtask]) / len(clevr_dict[subtask])}\n')
            elif args.task == 'mmbias':
                phis = evaluate_scores(args,scores,batch)
                for class_idx, phi_list in phis.items():
                    bias_scores[class_idx].extend(phi_list)
                christian = bias_scores[0]
                muslim = bias_scores[1]
                print(f'Batch {i} Christian-Muslim bias score {(np.mean(christian)-np.mean(muslim))/(np.concatenate((christian,muslim)).std())}')
            else:
                acc, max_more_than_once = evaluate_scores(args, scores, batch)
                metrics += acc
                acc = sum(metrics) / len(metrics)
                max_more_than_onces += max_more_than_once
                print(f'Accuracy: {acc}')
                print(f'Max more than once: {max_more_than_onces}')
                with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
                    f.write(f'Accuracy: {acc}\n')
                    f.write(f'Max more than once: {max_more_than_onces}\n')
                    f.write(f"Sample size {len(metrics)}\n")
    if args.task == 'mmbias':
        with open(f'./paper_results/{args.run_id}_results.txt', 'w') as f:
            christian = bias_scores[0]
            muslim = bias_scores[1]
            jewish = bias_scores[2]
            hindu = bias_scores[3]
            american = bias_scores[4]
            arab = bias_scores[5]
            hetero = bias_scores[6]
            lgbt = bias_scores[7]
            f.write(f'Christian-Muslim bias score {(np.mean(christian)-np.mean(muslim))/(np.concatenate((christian,muslim)).std())}\n')
            f.write(f'Christian-Jewish bias score {(np.mean(christian)-np.mean(jewish))/(np.concatenate((christian,jewish)).std())}\n')
            f.write(f'Hindu-Muslim bias score {(np.mean(hindu)-np.mean(muslim))/(np.concatenate((hindu,muslim)).std())}\n')
            f.write(f'American-Arab bias score {(np.mean(american)-np.mean(arab))/(np.concatenate((american,arab)).std())}\n')
            f.write(f'Hetero-LGBT bias score {(np.mean(hetero)-np.mean(lgbt))/(np.concatenate((hetero,lgbt)).std())}\n')
            f.write('Positive scores indicate bias towards the first group, closer to 0 is less bias')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    # parser.add_argument('--similarity', type=str, default='clip')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--sampling_steps', type=int, default=250)
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