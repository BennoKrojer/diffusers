import os
import json
from PIL import Image

def rate_images(dst_dir, log_file):
    # Load the mapping
    with open(log_file, 'r') as f:
        mapping = json.load(f)

    ratings = {}

    for i in range(len(mapping) // 2):
        img_a = Image.open(os.path.join(dst_dir, f'{i}_A.png'))
        img_b = Image.open(os.path.join(dst_dir, f'{i}_B.png'))

        # Concatenate images horizontally
        concatenated = Image.new('RGB', (img_a.width + img_b.width, img_a.height))
        concatenated.paste(img_a, (0, 0))
        concatenated.paste(img_b, (img_a.width, 0))

        concatenated.show()

        choice = None
        while choice not in ['left', 'right', 'unclear']:
            choice = input(f"Image {i}: Select 'left', 'right', or 'unclear': ").lower()

        ratings[f'{i}'] = choice

        # Save ratings to a JSON file
        with open('ratings.json', 'w') as f:
            json.dump(ratings, f, indent=4)

# Usage
dst_dir = 'generated_vg_attribution_random'
log_file = 'mapping.json'

rate_images(dst_dir, log_file)
