import torch
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from glob import glob
import os
from PIL import Image
from tqdm.auto import tqdm
import argparse
import json
import numpy as np
import clip
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
clip_version = 'RN50x64'
clip_model, preprocess = clip.load(clip_version, device='cuda:0')

base_dir = 'datasets/flickr30k'
img_dir = base_dir + '/images/'
data = open(f'{base_dir}/train_ann.jsonl', 'r')

captions = []
image_files = []
for j, line in enumerate(data):
    line = json.loads(line)
    sents = line['sentences']
    image_files.append(img_dir + line['img_path'])
    for i, sent in enumerate(sents):
        captions.append((sent, img_dir + line['img_path']))
print('Number of captions: ', len(captions))

def get_clip_features(captions):
    img_features = {}
    text_features = {}
    for i, caption in tqdm(enumerate(captions), total=len(captions)):
        text = clip.tokenize(caption[0], truncate=True).to(device)
        with torch.no_grad():
            text_feature = clip_model.encode_text(text)
            text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)
        text_features[i] = text_feature
    for i, img_path in tqdm(enumerate(image_files), total=len(image_files)):
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = clip_model.encode_image(image)
            image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
        img_features[img_path] = image_feature
    return img_features, text_features

img_features, text_features = get_clip_features(captions)

# for each text, find the closest image
def get_ranked_images(text_features, img_features):
    all_img_features = torch.cat(list(img_features.values()))
    keys = list(img_features.keys())
    all_text_features = torch.cat(list(text_features.values()))
    scores = torch.matmul(all_text_features, all_img_features.T)
    scores = scores.cpu().numpy()
    #get max indices for each text
    maxs = scores.argsort(axis=1)[:, -20:]
    # fill in ranked_images
    ranked_images = {}
    for i, max in enumerate(maxs):
        ranked_images[i] = [keys[m] for m in max]
    return ranked_images

ranked_images = get_ranked_images(text_features, img_features)
# get R1, R5, R10
def get_ranks(ranked_images, captions):
    r1 = 0
    r5 = 0
    r10 = 0
    for i, ranked_image in tqdm(ranked_images.items(), total=len(ranked_images)):
        if captions[i][1] in ranked_image[:1]:
            r1 += 1
        if captions[i][1] in ranked_image[:5]:
            r5 += 1
        if captions[i][1] in ranked_image[:10]:
            r10 += 1
    return r1, r5, r10

r1, r5, r10 = get_ranks(ranked_images, captions)
print('R1: ', r1 / len(captions))
print('R5: ', r5 / len(captions))
print('R10: ', r10 / len(captions))

# create dictionary of top 10 images for each caption
top10 = {}
for i, ranked_image in tqdm(ranked_images.items(), total=len(ranked_images)):
    #remove captions[i][1] from ranked_image[:10]
    nearest_imgs = ranked_image[:10]
    if captions[i][1] in nearest_imgs:
        nearest_imgs.remove(captions[i][1])
    else:
        nearest_imgs = nearest_imgs[:9]

    top10[captions[i][0]] = [captions[i][1]] + nearest_imgs

# save dictionary
with open(f'{base_dir}/train_top10_{clip_version}.json', 'w') as f:
    json.dump(top10, f, indent=4)
