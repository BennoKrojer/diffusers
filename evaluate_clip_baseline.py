import torch
from torchvision.transforms.functional import to_pil_image

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
import clip

import cProfile


class Scorer:
    def __init__(self, args, clip_model=None, preprocess=None):

        # vae = AutoencoderKL.from_pretrained('./stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
        # vae.to(device='cuda', dtype=torch.bfloat16)
        # self.vae_model = StableDiffusionImg2LatentPipeline(vae).to('cuda')
        self.cache_dir = args.cache_dir if args.cache else None


    def score_batch(self, i, args, batch, clip_model):
        """
        Takes a batch of images and captions and returns a score for each image-caption pair.
        """

        imgs, texts = batch[0], batch[1]
        imgs, imgs_resize = imgs[0], imgs[1]
        imgs, imgs_resize = [img.to('cuda:0') for img in imgs], [img.to('cuda:0') for img in imgs_resize]

        scores = []
        for txt_idx, text in enumerate(texts):
            for img_idx, resized_img in enumerate(imgs_resize):
                text_tensor = clip.tokenize(list(text)).to('cuda:0')                
                text_embedding = clip_model.encode_text(text_tensor)
                resized_img = resized_img.squeeze().unsqueeze(0)
                image_embedding = clip_model.encode_image(resized_img)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
                score = (100.0 * text_embedding @ image_embedding.T).squeeze()
                scores.append(score)

        scores = torch.stack(scores).permute(1, 0) if args.batchsize > 1 else torch.stack(scores).unsqueeze(0)
        return scores
        

def main(args):

    clip_version = 'RN50x64'
    model, preprocess = clip.load(clip_version, device='cuda:0')

    scorer = Scorer(args)
    dataset = get_dataset(args.task, f'datasets/{args.task}', transform=preprocess)

    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=0)

    r1s = []
    r5s = []
    max_more_than_onces = 0
    metrics = []
    ids = []
    clevr_dict = {}
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if args.subset and i % 10 != 0:
            continue
        with torch.no_grad():
            scores = scorer.score_batch(i, args, batch, model)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    # parser.add_argument('--similarity', type=str, default='clip')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--sampling_steps', type=int, default=250)
    parser.add_argument('--img_retrieval', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)

    args.run_id = f'{args.task}_clip_baseline_seed{args.seed}_steps{args.sampling_steps}_subset{args.subset}_img_retrieval{args.img_retrieval}'

    if args.cache:
        args.cache_dir = f'./cache/{args.run_id}'
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
            
    main(args)