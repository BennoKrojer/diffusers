from PIL import Image
import torch
import os
import json
import numpy as np
import glob
from tqdm.auto import tqdm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = '/home/krojerb/scratch/flickr30k_images'
data = open(f'{base_dir}/annotations/valid_ann.jsonl', 'r')

metrics = {'cos': torch.nn.CosineSimilarity(dim=1), 'dot': torch.matmul}

# sort out only the validation images 
valid_ids = []
latent_imgs = {}

for line in tqdm(data):
    line = json.loads(line)
    id = line['id']
    valid_ids.append(id)
    img_path = os.path.join(base_dir, 'latent_images', id+'.pt')
    # load the encoded flickr image from saved numpy
    latent_img = torch.load(img_path, map_location=device).flatten().float()
    latent_imgs[id] = latent_img

# sort by id so we can access it later
valid_ids = sorted(valid_ids)
latent_imgs = dict(sorted(latent_imgs.items())) 

# make a big tensor stacking all the latent encoded images
flickr_imgs = torch.stack(list(latent_imgs.values()))
# shape is (30k, latent_size)


# loop through captions & calculate similarities
latent_cap_paths = glob.glob(os.path.join(base_dir,'captions/*'))

r1 = 0
r5 = 0
r10 = 0
results = {'cos':{'r1':0,'r5':0,'r10':0},'dot':{'r1':0,'r5':0,'r10':0}}

for path in tqdm(latent_cap_paths):
    caption_latent = torch.from_numpy(np.load(path)).to(device).flatten().float()
    caption_id = path.split('/')[-1].split('_')[0]
    correct_id = valid_ids.index(caption_id)
    for metric_name, metric in metrics.items():
        score = metric(flickr_imgs, caption_latent).argsort() # resulting shape is (30k)
        # want to find rank of correct image for the given captions
        rank = (score==correct_id).nonzero().item()
        
        if rank == 0:
            results[metric_name]['r1'] +=1
            results[metric_name]['r5'] +=1
            results[metric_name]['r10'] +=1
        elif rank <5:
            results[metric_name]['r5'] +=1
            results[metric_name]['r10'] +=1
        elif rank < 10:
            results[metric_name]['r10'] +=1

final_scores = {'cos':{'r1':0,'r5':0,'r10':0},'dot':{'r1':0,'r5':0,'r10':0}}
num_captions = len(latent_cap_paths) # 5070
for metric_name in metrics.keys():
    final_scores[metric_name]['r1'] = results[metric_name]['r1']/num_captions
    final_scores[metric_name]['r5'] = results[metric_name]['r5']/num_captions
    final_scores[metric_name]['r10'] = results[metric_name]['r10']/num_captions
    print("="*20)
    print(f"R@1 score (metric {metric_name}): {results[metric_name]['r1']}/{num_captions} = {final_scores[metric_name]['r1']}")
    print(f"R@5 score: {results[metric_name]['r5']}/{num_captions} = {final_scores[metric_name]['r5']}")
    print(f"R@10 score: {results[metric_name]['r10']}/{num_captions} = {final_scores[metric_name]['r10']}")
    print("="*20)
        
    
    
    
    