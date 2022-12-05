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
# clip_version = 'ViT-L/14@336px'
clip_version = 'RN50x64'
clip_model, preprocess = clip.load(clip_version, device='cuda:0')

base_dir = 'flickr30'
img_dir = base_dir + '/images/'
data = open(f'{base_dir}/valid_ann.jsonl', 'r')

datapoints = []
captions = []
images = list(glob(img_dir + '*.jpg'))
image_features = []
valid_imgs = []

for line in data:
    line = json.loads(line)
    sents = line['sentences']
    valid_imgs.append(img_dir + line['img_path'])
    for i, sent in enumerate(sents):
        datapoints.append((sent, img_dir + line['img_path']))
        captions.append(sent)

images = valid_imgs

# process in batches
for image in tqdm(range(0, len(valid_imgs), 32)):
    image_batch = images[image:image+32]
    image_batch = [preprocess(Image.open(img_path)).unsqueeze(0).to(device) for img_path in image_batch]
    image_batch = torch.cat(image_batch)
    with torch.no_grad():
        image_feature = clip_model.encode_image(image_batch)
        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
    image_features.append(image_feature)
image_features = torch.cat(image_features)

text = clip.tokenize(captions).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = clip_model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy().T

# for each text, rank the image paths
ranking = {}
for i, caption in enumerate(captions):
    ranking[caption] = []
    for j in text_probs[i].argsort()[::-1]:
        ranking[caption].append(images[j])

# get R1, R5, R10 for each caption
r1 = 0
r5 = 0
r10 = 0
for i, caption in enumerate(captions):
    for j, image in enumerate(ranking[caption]):
        if image == datapoints[i][1]:
            if j == 0:
                r1 += 1
            if j < 5:
                r5 += 1
            if j < 10:
                r10 += 1
            break

print('R1: ', r1 / len(captions))
print('R5: ', r5 / len(captions))
print('R10: ', r10 / len(captions))

# create dictionary of caption to top10 image paths
top10 = {}
for i, caption in enumerate(captions):
    correct_path = datapoints[i][1]
    top10[caption] = [correct_path, ranking[caption][:10]]

# save to json
with open(f'{base_dir}/top10_{clip_version}.json', 'w') as f:
    json.dump(top10, f)
