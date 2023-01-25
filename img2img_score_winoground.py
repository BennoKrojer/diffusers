from PIL import Image
import torch
import clip
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda:0')

def score(i,metric,caption_path='winoground/img2img_captions_latent',image_path='winoground/images_latent', use_clip=False):
    metrics = {'cos': torch.nn.CosineSimilarity(dim=0), 'dot': torch.dot}
    metric = metrics[metric]
    
    if use_clip:
        
        original_image0 = Image.open(os.path.join('winoground/images', f'ex_{i}_img_0.png'))
        original_image1 = Image.open(os.path.join('winoground/images', f'ex_{i}_img_1.png')) 
        original_image0 = preprocess(original_image0).unsqueeze(0).cuda()
        original_image1 = preprocess(original_image1).unsqueeze(0).cuda()
        with torch.no_grad():
            img_latent0 = clip_model.encode_image(original_image0).squeeze().float()
            img_latent1 = clip_model.encode_image(original_image1).squeeze().float()      


        cap0_img0 = Image.open(caption_path + f'/ex_{i}_img_0_cap_0.png')
        cap0_img1 = Image.open(caption_path + f'/ex_{i}_img_1_cap_0.png')
        cap1_img0 = Image.open(caption_path + f'/ex_{i}_img_0_cap_1.png')
        cap1_img1 = Image.open(caption_path + f'/ex_{i}_img_1_cap_1.png')
        cap0_img0 = preprocess(cap0_img0).unsqueeze(0).cuda()
        cap0_img1 = preprocess(cap0_img1).unsqueeze(0).cuda()
        cap1_img0 = preprocess(cap1_img0).unsqueeze(0).cuda()
        cap1_img1 = preprocess(cap1_img1).unsqueeze(0).cuda()

        with torch.no_grad():
            cap0_img0 = clip_model.encode_image(cap0_img0).squeeze().float()
            cap1_img0 = clip_model.encode_image(cap1_img0).squeeze().float()
            cap0_img1 = clip_model.encode_image(cap0_img1).squeeze().float()
            cap1_img1 = clip_model.encode_image(cap1_img1).squeeze().float()

    else:
        img_latent0 = torch.load(image_path + f'/ex_{i}_img_0_latent.pt', map_location=torch.device(device)).flatten().float()
        img_latent1 = torch.load(image_path + f'/ex_{i}_img_1_latent.pt', map_location=torch.device(device)).flatten().float()

        cap0_img0 = torch.load(caption_path + f'/ex_{i}_latent_0_cap_0.pt', map_location=torch.device(device)).flatten().float()
        cap1_img0 = torch.load(caption_path + f'/ex_{i}_latent_0_cap_1.pt', map_location=torch.device(device)).flatten().float()
        cap0_img1 = torch.load(caption_path + f'/ex_{i}_latent_1_cap_0.pt', map_location=torch.device(device)).flatten().float()
        cap1_img1 = torch.load(caption_path + f'/ex_{i}_latent_1_cap_1.pt', map_location=torch.device(device)).flatten().float()

        # euclidean distance  between image_latent and caption_image_latent
        #CORRECT:
    c0_i0 = torch.dot(img_latent0 - cap0_img0, img_latent0 - cap0_img0)
    c0_i1 = torch.dot(img_latent1 - cap0_img1, img_latent1 - cap0_img1)
    c1_i0 = torch.dot(img_latent0 - cap1_img0, img_latent0 - cap1_img0)
    c1_i1 = torch.dot(img_latent1 - cap1_img1, img_latent1 - cap1_img1)

    # WRONG:
    # c0_i0 = torch.dot(img_latent0 - cap0_img0_latent, img_latent0 - cap0_img0_latent)
    # c0_i1 = torch.dot(img_latent1 - cap1_img0_latent, img_latent1 - cap1_img0_latent)
    # c1_i0 = torch.dot(img_latent0 - cap0_img1_latent, img_latent0 - cap0_img1_latent)
    # c1_i1 = torch.dot(img_latent1 - cap1_img1_latent, img_latent1 - cap1_img1_latent)

    c0_i0, c0_i1, c1_i0, c1_i1 = -c0_i0, -c0_i1, -c1_i0, -c1_i1
    
    return c0_i0, c0_i1, c1_i0, c1_i1


if __name__ == "__main__":
    from tqdm import tqdm

    METRIC = 'dot'

    text_scores = 0
    image_scores = 0
    group_scores = 0
    caption_path='winoground/img2img_captions_latent_guidance_scale_7.5_strength_0.8_steps_50_seed_5'
    print("EVALUATING ON", caption_path)
    USE_CLIP = True
    ACROSS_SEEDS = False

    seeds = ['X', '1', '2', '3', '4']
    
    for i in tqdm(range(400)):
        c0_i0s = []
        c0_i1s = []
        c1_i0s = []
        c1_i1s = []
        if ACROSS_SEEDS:
            for seed in seeds:
                seeded_caption_path = caption_path[:-1] + seed
                c0_i0, c0_i1, c1_i0, c1_i1 = score(i, metric=METRIC, caption_path=seeded_caption_path, use_clip=USE_CLIP)
                c0_i0s.append(c0_i0)
                c0_i1s.append(c0_i1)
                c1_i0s.append(c1_i0)
                c1_i1s.append(c1_i1)
            # c0_i0 = max(c0_i0s)
            # c0_i1 = max(c0_i1s)
            # c1_i0 = max(c1_i0s)
            # c1_i1 = max(c1_i1s)
            #mean
            c0_i0 = sum(c0_i0s)/len(c0_i0s)
            c0_i1 = sum(c0_i1s)/len(c0_i1s)
            c1_i0 = sum(c1_i0s)/len(c1_i0s)
            c1_i1 = sum(c1_i1s)/len(c1_i1s)
        else:
            c0_i0, c0_i1, c1_i0, c1_i1 = score(i, metric=METRIC, caption_path=caption_path, use_clip=USE_CLIP)

        text_score = 1 if c0_i0 > c1_i0 and c1_i1 > c0_i1 else 0
        img_score = 1 if c0_i0 > c0_i1 and c1_i1 > c1_i0 else 0
        group_score = 1 if text_score and img_score else 0
        text_scores += text_score
        image_scores += img_score
        group_scores += group_score

        print(f'Group score: {(group_scores)/(i+1)}')
        print(f'Text score: {(text_scores)/(i+1)}')
        print(f'Image score: {(image_scores)/(i+1)}')