import json
import os
from glob import glob

# top10 = json.load(open('flickr30/top10_RN50x64.json', 'r'))

# if not os.path.exists('flickr30/images_grouped'):
#     os.mkdir('flickr30/images_grouped')


# for i, (caption, (correct_path, all_images)) in enumerate(top10.items()):
#     if i > 100:
#         break
#     #makedir
#     if not os.path.exists(f'flickr30/images_grouped/{i}'):
#         os.mkdir(f'flickr30/images_grouped/{i}')
#     #copy all images to a folder
#     for j, img_path in enumerate(all_images):
#         img_id = img_path.split('/')[-1].split('.')[0]
#         os.system(f'cp {img_path} flickr30/images_grouped/{i}/{j}_{img_id}.png')
#     # copy correct image to same folder
#     img_id = correct_path.split('/')[-1].split('.')[0]
#     os.system(f'cp {correct_path} flickr30/images_grouped/{i}/{img_id}_correct.png')


# go through all files in flickr/generated_examples/flickr/img2img_captions_latent_guidance_scale_7.5_strength_0.8_steps_50_txt2img_seed_0 and delete all files that have an underscore in the name

for f in glob('flickr/generated_examples/flickr/img2img_captions_latent_guidance_scale_7.5_strength_0.8_steps_50_txt2img_seed_0/*'):
    if '_' in f:
        os.remove(f)