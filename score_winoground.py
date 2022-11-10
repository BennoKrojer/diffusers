import torch
import glob
import argparse
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRIC = 'cos'

caption_path = 'winoground/captions_latent/'
image_path = 'winoground/images_latent/'

metrics = {'cos': torch.nn.CosineSimilarity(dim=0), 'dot': torch.dot, 'ce': torch.nn.CrossEntropyLoss()}

text_scores = 0
img_scores = 0
group_scores = 0

for i in tqdm(range(400)):
    caption_latent0 = torch.load(caption_path + f'ex{i}_caption0_latent.pt', map_location=torch.device(device)).flatten().float()
    caption_latent1 = torch.load(caption_path + f'ex{i}_caption1_latent.pt', map_location=torch.device(device)).flatten().float()

    img_latent0 = torch.load(image_path + f'ex_{i}_img_0_latent.pt', map_location=torch.device(device)).flatten().float()
    img_latent1 = torch.load(image_path + f'ex_{i}_img_1_latent.pt', map_location=torch.device(device)).flatten().float()


    c0_i0 = metrics[METRIC](caption_latent0, img_latent0)
    c0_i1 = metrics[METRIC](caption_latent0, img_latent1)
    c1_i0 = metrics[METRIC](caption_latent1, img_latent0)
    c1_i1 = metrics[METRIC](caption_latent1, img_latent1)
    if METRIC == 'ce':
        c0_i0, c0_i1, c1_i0, c1_i1 = -c0_i0, -c0_i1, -c1_i0, -c1_i1

    text_score = 1 if c0_i0 > c1_i0 and c1_i1 > c0_i1 else 0
    text_scores += text_score
    img_score = 1 if c0_i0 > c0_i1 and c1_i1 > c1_i0 else 0
    img_scores += img_score
    group_score = 1 if text_score and img_score else 0
    group_scores += group_score

    print(f'Group score: {(group_scores)/(i+1)}')
    print(f'Text score: {(text_scores)/(i+1)}')
    print(f'Image score: {(text_scores)/(i+1)}')