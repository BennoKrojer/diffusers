from PIL import Image
import torch
import clip
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda:0')

def score(i,metric,caption_path='winoground/captions_latent',image_path='winoground/images_latent', use_clip=False):
    metrics = {'cos': torch.nn.CosineSimilarity(dim=0), 'dot': torch.dot, 'ce': torch.nn.CrossEntropyLoss()}
    metric = metrics[metric]
    
    if use_clip:
        image0 = Image.open(os.path.join('winoground/all_generated_images', f'ex{i}_caption0_generated.png'))
        image1 = Image.open(os.path.join('winoground/all_generated_images', f'ex{i}_caption1_generated.png'))
        image0 = preprocess(image0).unsqueeze(0).cuda()
        image1 = preprocess(image1).unsqueeze(0).cuda()
        with torch.no_grad():
            caption_latent0 = clip_model.encode_image(image0).squeeze().float()
            caption_latent1 = clip_model.encode_image(image1).squeeze().float()

        original_image0 = Image.open(os.path.join('winoground/images', f'ex_{i}_img_0.png'))
        original_image1 = Image.open(os.path.join('winoground/images', f'ex_{i}_img_1.png')) 
        original_image0 = preprocess(original_image0).unsqueeze(0).cuda()
        original_image1 = preprocess(original_image1).unsqueeze(0).cuda()
        with torch.no_grad():
            img_latent0 = clip_model.encode_image(original_image0).squeeze().float()
            img_latent1 = clip_model.encode_image(original_image1).squeeze().float()      

    else:
        caption_latent0 = torch.load(caption_path + f'/ex{i}_caption0_latent.pt', map_location=torch.device(device)).flatten().float()
        caption_latent1 = torch.load(caption_path + f'/ex{i}_caption1_latent.pt', map_location=torch.device(device)).flatten().float()

        img_latent0 = torch.load(image_path + f'/ex_{i}_img_0_latent.pt', map_location=torch.device(device)).flatten().float()
        img_latent1 = torch.load(image_path + f'/ex_{i}_img_1_latent.pt', map_location=torch.device(device)).flatten().float()

    c0_i0 = metric(caption_latent0, img_latent0)
    c0_i1 = metric(caption_latent0, img_latent1)
    c1_i0 = metric(caption_latent1, img_latent0)
    c1_i1 = metric(caption_latent1, img_latent1)

    if metric == 'ce':
        c0_i0, c0_i1, c1_i0, c1_i1 = -c0_i0, -c0_i1, -c1_i0, -c1_i1

    
    
    return c0_i0, c0_i1, c1_i0, c1_i1


if __name__ == "__main__":
    from tqdm import tqdm

    METRIC = 'cos'

    caption_path = 'winoground/captions_latent/'
    image_path = 'winoground/images_latent/'

    text_scores = 0
    image_scores = 0
    group_scores = 0

    
    for i in tqdm(range(400)):
        c0_i0, c0_i1, c1_i0, c1_i1 = score(i,metric=METRIC,caption_path=caption_path,image_path=image_path, use_clip=True)
        text_score = 1 if c0_i0 > c1_i0 and c1_i1 > c0_i1 else 0
        img_score = 1 if c0_i0 > c0_i1 and c1_i1 > c1_i0 else 0
        group_score = 1 if text_score and img_score else 0
        text_scores += text_score
        image_scores += img_score
        group_scores += group_score

        print(f'Group score: {(group_scores)/(i+1)}')
        print(f'Text score: {(text_scores)/(i+1)}')
        print(f'Image score: {(image_scores)/(i+1)}')