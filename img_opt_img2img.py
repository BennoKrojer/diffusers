import cv2
import matplotlib.pyplot as plt
# import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm.auto import tqdm
import os
import clip

# from diffusers import StableDiffusionPipeline
from src.diffusers.models.vae import DiagonalGaussianDistribution
from src.diffusers.schedulers import LMSDiscreteScheduler

from src.diffusers import StableDiffusionPipeline
import imageio

import argparse

from datasets import get_dataset
from torch.utils.data import DataLoader

from utils import evaluate_scores

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def decode_latents(pipe, latents):
    with torch.no_grad():
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        # image = pipe.vae.decode(latents.to(self.vae.dtype)).sample
        image = pipe.vae.decoder(pipe.vae.post_quant_conv(latents.to(pipe.vae.dtype)))

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
    return image


def img_train(latents, optim, pipe, prompt, n_iters, guidance_scale=40.0, sample_freq=10):
    try:
        # expand the latents if we are doing classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        with torch.no_grad():
            # get prompt text embeddings
            text_input = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.text_encoder.device))[0]

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance:
                uncond_input = pipe.tokenizer(
                    "", padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
                )
                uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.text_encoder.device))[0]

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        num_train_timesteps = pipe.scheduler.config.num_train_timesteps
        min_step = int(num_train_timesteps * 0.02)
        max_step = int(num_train_timesteps * 0.98)

        text_unet = None
        losses, losses_abs, losses_diff, losses_diff_x0, samples = [], [], [], [], []

        for i in range(n_iters):

            with torch.no_grad():

                # timesteps = torch.randint(min_step, max_step + 1, (latents.shape[0],), dtype=torch.long, device=pipe.device).sort()[0]
                timesteps = torch.randint(min_step, max_step + 1, (latents.shape[0],), dtype=torch.long, device=pipe.device)

                # add noise to latents using the timesteps
                noise = torch.randn(latents.shape, generator=None, device=pipe.device)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps).to(pipe.device)

                # predict the noise residual
                samples_unet = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
                timesteps_unet = torch.cat([timesteps] * 2) if do_classifier_free_guidance else timesteps
                text_unet = text_embeddings.repeat_interleave(len(noisy_latents), 0) if text_unet is None else text_unet
                noise_pred = []
                for s, t, text in zip(samples_unet, timesteps_unet, text_unet):
                    noise_pred.append(pipe.unet(sample=s[None, ...],
                                                timestep=t[None, ...],
                                                encoder_hidden_states=text[None, ...]).sample)

                noise_pred = torch.cat(noise_pred)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            optim.zero_grad()
            loss = ((noise_pred - noise) * latents).mean()
            loss.backward()
            optim.step()

            loss_abs = ((noise_pred - noise) * latents).abs().mean()
            loss_diffusion = ((noise_pred - noise)**2).mean()
            loss_diffusion_x0 = ((latents - (noisy_latents - torch.sqrt(1 - pipe.scheduler.alphas_cumprod[timesteps])[:, None, None, None] * noise_pred)/torch.sqrt(pipe.scheduler.alphas_cumprod[timesteps])[:, None, None, None])**2).mean()
            
            losses.append(loss.item())
            losses_abs.append(loss_abs.item())
            losses_diff.append(loss_diffusion.item())
            losses_diff_x0.append(loss_diffusion_x0.item())
            
            if i % sample_freq == 0:
                # print(i, loss.item())
                # media.show_images(latents.detach().cpu().permute(1, 0, 2, 3).reshape(-1, 64, 64), columns=len(latents))
                s = decode_latents(pipe, latents.detach())
                # media.show_images(s, height=100)
                samples.append(s)

    except KeyboardInterrupt:
        print("Ctrl+C!")

    return latents, [losses, losses_abs, losses_diff, losses_diff_x0], samples


def make_gif_from_imgs(frames, fname, caption, imsize=512, resize=1.0, fps=3, upto=None, repeat_first=1, skip=1,
             f=0, s=0.75, t=2):
    imgs = []
    for i, img in tqdm(enumerate(frames[:upto:skip]), total=len(frames[:upto:skip])):
        img = np.moveaxis(img, 0, 1).reshape(512, -1, 3)
        n_cols = img.shape[1]//imsize
        img = np.array(Image.fromarray((img*255).astype(np.uint8)).resize((int(img.shape[1]/resize), int(img.shape[0]/resize)), Image.Resampling.LANCZOS))
        # img = np.concatenate([img[:, int(i*imsize/resize):int((i+1)*imsize/resize), :] for i in range(0, n_cols, 2)], axis=1)
        text = f"{i*10:05d}"
        img = cv2.putText(img=img, text=text, org=(0, 20), fontFace=f, fontScale=s, color=(0,0,0), thickness=t)
        # add a small white stripe at the top that contains the caption text
        img = np.concatenate([np.ones((20, img.shape[1], 3))*255, img], axis=0).astype(np.uint8)
        img = cv2.putText(img=img, text=caption, org=(0, 15), fontFace=f, fontScale=s, color=(0,0,0), thickness=t)
        imgs.append(img)
        if i == len(frames[:upto:skip]) - 1:
            # save last frame as file too
            imageio.imwrite(f"{fname}.png", img, format='png')
            
    # Save gif
    imgs = [imgs[0]]*repeat_first + imgs
    imageio.mimwrite(f'{fname}.gif', imgs, fps=4)
    
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='flickr30k')
parser.add_argument('--n_iters', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--score_method', type=str, default='latent')
args = parser.parse_args()

torch.manual_seed(0)

#print 
model_id_or_path = "./stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id_or_path,
)
pipe = pipe.to(f"cuda:{args.cuda_device}")

if args.score_method == 'clip':
    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda')
    transform = preprocess
else:
    transform = None

dataset = get_dataset(args.task, f'datasets/{args.task}', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

if args.score_method == 'inception':
    import torch
    import torchvision
    from torchvision import transforms
    inception_model = torchvision.models.inception_v3(pretrained=True)
    inception_model.eval()
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

metrics = []
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    if i % 5 != 0:
        continue
    imgs, texts = batch[0], batch[1]
    imgs, imgs_resize = imgs[0], imgs[1]
    imgs, imgs_resize = [img.cuda() for img in imgs], [img.cuda() for img in imgs_resize]

    scores = []
    for txt_idx, text in enumerate(texts):        

        for img_idx, img in enumerate(imgs):
            resized_img = imgs_resize[img_idx].to(pipe.device)
            
            resized_latent_dist = pipe.vae.encode(resized_img).latent_dist
            resized_latents = resized_latent_dist.sample()
            resized_latents = 0.18215 * resized_latents
            latents_copy = resized_latents.detach().clone()
            latents_copy.requires_grad = True
            
            optim = torch.optim.Adam([latents_copy], lr=args.lr)
            fname = f'cache/{args.task}/img_opt_img2img_lr{args.lr}/{i}_{txt_idx}_{img_idx}'
            if not os.path.exists(f'cache/{args.task}/img_opt_img2img_lr{args.lr}/'):
                os.makedirs(f'cache/{args.task}/img_opt_img2img_lr{args.lr}/')
            latents, losses, samples = img_train(latents_copy, optim, pipe, list(text), args.n_iters, 40.0)
            make_gif_from_imgs(samples, fname, text[0], resize=1)

            
    #         if args.score_method == 'loss':
    #             all_losses = [torch.tensor(x) for x in losses[2]]
    #             score = sum(all_losses) / len(all_losses)
    #         elif args.score_method == 'latent':
    #             score = (latents - resized_latents)**2
    #             score = score.mean()
    #         elif args.score_method == 'pixel':
    #             resized_img = resized_img / 2.0 + 0.5
    #             resized_img = resized_img.permute(0, 2, 3, 1).cpu()
    #             acc_scores = []
    #             for i in range(len(samples)):
    #                 score = (resized_img - samples[-1])**2
    #                 score = score.mean()
    #                 acc_scores.append(score)
    #             score = sum(acc_scores) / len(acc_scores)
    #         elif args.score_method == 'clip':
    #             with torch.no_grad():
    #                 img_emb = clip_model.encode_image(img).squeeze().float()
    #                 acc_scores = []
    #                 for i in range(len(samples)):
    #                     sample = numpy_to_pil(samples[i])
    #                     sample = torch.stack([preprocess(sample[0])]).cuda()
    #                     sample_emb = clip_model.encode_image(sample).squeeze().float()
    #                     score = torch.norm(img_emb - sample_emb, dim=-1)
    #                     acc_scores.append(score)
    #                 score = sum(acc_scores) / len(acc_scores)
    #                 score = -score
    #         elif args.score_method == 'inception':
    #             with torch.no_grad():
    #                 img = transform(img[0].cpu())
    #                 img = img.unsqueeze(0)
    #                 img = img.to(pipe.device)
    #                 img = inception_model(img)
    #                 acc_scores = []
    #                 for i in range(len(samples)):
    #                     sample = transform(samples[i][0].cpu())
    #                     sample = sample.unsqueeze(0)
    #                     sample = sample.to(pipe.device)
    #                     sample = inception_model(sample)
    #                     score = torch.norm(img - sample, dim=-1)
    #                     acc_scores.append(score)
    #                 score = sum(acc_scores) / len(acc_scores)

    #         score = -score
    #         scores.append(score.detach())
            
    # scores = torch.stack(scores).unsqueeze(0)
    # scoring = evaluate_scores(args, scores, batch)
    # metrics.append(scoring)
    # accuracy = sum(metrics) / len(metrics)
    # print(f'Retrieval Accuracy: {accuracy}')