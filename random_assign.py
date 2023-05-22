import os
import shutil
import random
import json


def random_copy_and_rename(src_dir, dst_dir, log_file):
    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Get all .png files from the source directory
    files = [f for f in os.listdir(src_dir) if f.endswith('.png')]

    # Create pairs of images
    pairs = [(f, f.replace('zeroshot', 'finetuned')) for f in files if 'zeroshot' in f]
    
    mapping = {}

    for i, (img1, img2) in enumerate(pairs):
        new_names = ['A', 'B'] if random.random() < 0.5 else ['B', 'A']

        # Create a mapping
        mapping[f'{i}_{new_names[0]}.png'] = img1
        mapping[f'{i}_{new_names[1]}.png'] = img2

        # Copy and rename images
        shutil.copy2(os.path.join(src_dir, img1), os.path.join(dst_dir, f'{i}_{new_names[0]}.png'))
        shutil.copy2(os.path.join(src_dir, img2), os.path.join(dst_dir, f'{i}_{new_names[1]}.png'))

    # Save the mapping to a json file
    with open(log_file, 'w') as f:
        json.dump(mapping, f, indent=4)

# Usage
src_dir = 'generated_drawbench_seed42'
dst_dir = 'generated_drawbench_seed42_random'
log_file = 'mapping_seed42.json'

random_copy_and_rename(src_dir, dst_dir, log_file)
