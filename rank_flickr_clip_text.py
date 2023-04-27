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
    for i, sent in enumerate(sents[:6]):
        captions.append((sent, img_dir + line['img_path']))
print('Number of captions: ', len(captions))

# reformat captions as a mapping from image path to list of captions
def get_captions(captions):
    captions_dict = {}
    for caption in captions:
        if caption[1] in captions_dict:
            captions_dict[caption[1]].append(caption[0])
        else:
            captions_dict[caption[1]] = [caption[0]]
    return captions_dict

def get_clip_features(captions):
    img_features = {}
    text_features = []
    tokenized = [clip.tokenize(caption[0], truncate=True).to(device) for caption in captions]
    tokenized = torch.cat(tokenized)

    #batchsize 64
    for i in tqdm(range(0, len(tokenized), 64)):
        with torch.no_grad():
            text_feature = clip_model.encode_text(tokenized[i:i+64])
            text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)
        text_features.append(text_feature)
    text_features = torch.cat(text_features)
    # batchsize 64
    for i in tqdm(range(0, len(image_files), 64)):
        images = torch.cat([preprocess(Image.open(img_path)).unsqueeze(0).to(device) for img_path in image_files[i:i+64]])
        with torch.no_grad():
            image_feature = clip_model.encode_image(images)
            image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
        for j, img_path in enumerate(image_files[i:i+64]):
            img_features[img_path] = image_feature[j]
    # for i, caption in tqdm(enumerate(captions), total=len(captions)):
    #     with torch.no_grad():
    #         text_feature = clip_model.encode_text(text)
    #         text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)
    #     text_features[i] = text_feature
    # for i, img_path in tqdm(enumerate(image_files), total=len(image_files)):
    #     image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         image_feature = clip_model.encode_image(image)
    #         image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
    #     img_features[img_path] = image_feature

    return img_features, text_features

img_features, text_features = get_clip_features(captions)

# for each text, find the closest image
def get_ranked_images(text_features, img_features):
    all_img_features = torch.stack(list(img_features.values()))
    keys = list(img_features.keys())
    scores = torch.matmul(text_features, all_img_features.T)
    scores = scores.cpu().numpy()
    scores = -scores
    #get max indices for each text
    maxs = scores.argsort(axis=1)[:, :20]
    # fill in ranked_images
    ranked_images = {}
    for i, max in enumerate(maxs):
        ranked_images[i] = [keys[m] for m in max]
    return ranked_images

def get_ranked_texts(text_features, img_features):
    all_text_features = text_features
    all_img_features = torch.stack(list(img_features.values()))
    keys = list(img_features.keys())
    scores = torch.matmul(all_img_features, all_text_features.T)
    scores = scores.cpu().numpy()
    scores = -scores
    #get max indices for each image
    maxs = scores.argsort(axis=1)[:, :20]
    # fill in ranked_texts
    ranked_texts = {}
    for i, max in enumerate(maxs):
        ranked_texts[keys[i]] = [captions[m][0] for m in max]
    return ranked_texts



ranked_texts = get_ranked_texts(text_features, img_features)

def get_ranks_text(ranked_texts, captions_dict):
    # ranked_texs: dict of image path to top 20 captions
    # captions_dict: dict of image path to list of captions
    # r1, r5, r10: number of images with top 1, 5, 10 captions in top 20 captions
    # so an example gets +1 for r1 if one of the five groundtruth captions is the first ranked one
    r1 = 0
    r5 = 0
    r10 = 0
    for img_path, ranked_text in tqdm(ranked_texts.items(), total=len(ranked_texts)):
        captions_list = captions_dict[img_path]
        r5_counted = False
        r10_counted = False
        for caption in captions_list:
            if caption in ranked_text[:1]:
                r1 += 1
            if caption in ranked_text[:5] and not r5_counted:
                r5 += 1
                r5_counted = True
            if caption in ranked_text[:10] and not r10_counted:
                r10 += 1
                r10_counted = True
    return r1, r5, r10

    
captions_dict = get_captions(captions)
r1, r5, r10 = get_ranks_text(ranked_texts, captions_dict)
print(f'R1: {r1/len(captions_dict)}, R5: {r5/len(captions_dict)}, R10: {r10/len(captions_dict)}')

# create dictionary of top 10 captions for each image
top10 = {}
for img_path, ranked_text in tqdm(ranked_texts.items(), total=len(ranked_texts)):
    best_non_groundtruth_captions = []
    captions_list = captions_dict[img_path]
    for caption in ranked_text:
        if caption not in captions_list:
            best_non_groundtruth_captions.append(caption)
        if len(best_non_groundtruth_captions) == 9:
            break
    
    for caption in captions_list:
        top10[img_path] = [caption] + best_non_groundtruth_captions

# save dictionary
with open(f'{base_dir}/train_top10_{clip_version}_text.json', 'w') as f:
    json.dump(top10, f, indent=4)
