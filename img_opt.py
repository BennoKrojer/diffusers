import cv2
import matplotlib.pyplot as plt
# import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm.auto import tqdm

# from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline

import imageio

import os

import argparse

from datasets import get_dataset
from torch.utils.data import DataLoader

from utils import evaluate_scores


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

        for i in tqdm(range(n_iters)):

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


def make_gif_from_imgs(frames, fname, imsize=512, resize=1.0, fps=3, upto=None, repeat_first=1, skip=1,
             f=0, s=0.75, t=2):
    imgs = []
    for i, img in tqdm(enumerate(frames[:upto:skip]), total=len(frames[:upto:skip])):
        img = np.moveaxis(img, 0, 1).reshape(512, -1, 3)
        n_cols = img.shape[1]//imsize
        img = np.array(Image.fromarray((img*255).astype(np.uint8)).resize((int(img.shape[1]/resize), int(img.shape[0]/resize)), Image.Resampling.LANCZOS))
        # img = np.concatenate([img[:, int(i*imsize/resize):int((i+1)*imsize/resize), :] for i in range(0, n_cols, 2)], axis=1)
        text = f"{i*10:05d}"
        img = cv2.putText(img=img, text=text, org=(0, 20), fontFace=f, fontScale=s, color=(0,0,0), thickness=t)
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
parser.add_argument('--n_samples', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--cuda_device', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(0)

#print 
model_id = "stabilityai/stable-diffusion-2-1-base"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
model = model.to(accelerator.device)

random_latent = torch.randn((args.n_samples, 4, 64, 64), device=pipe.device, requires_grad=True)
dataset = get_dataset(args.task, f'datasets/{args.task}', transform=None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

metrics = []
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    imgs, texts = batch[0], batch[1]
    imgs, imgs_resize = imgs[0], imgs[1]
    imgs, imgs_resize = [img.cuda() for img in imgs], [img.cuda() for img in imgs_resize]

    scores = []
    for txt_idx, text in enumerate(texts):
        latents_copy = random_latent.clone().detach()
        latents_copy.requires_grad = True
        optim = torch.optim.Adam([latents_copy], lr=args.lr)
        fname = f"cache/{args.task}/img_opt_txt2img_lr{args.lr}/{i}_{txt_idx}"
        if not os.path.exists(f'cache/{args.task}/img_opt_txt2img_lr{args.lr}/'):
            os.makedirs(f'cache/{args.task}/img_opt_txt2img_lr{args.lr}/')
        latents, losses, samples = img_train(latents_copy, optim, pipe, list(text), args.n_iters, 40.0)
        make_gif_from_imgs(samples, fname, resize=4)

        for img_idx, img in enumerate(imgs):
            resized_img = imgs_resize[img_idx].to(pipe.device)
            resized_latent_dist = pipe.vae.encode(resized_img).latent_dist
            resized_latents = resized_latent_dist.sample()
            resized_latents = 0.18215 * resized_latents
            resized_latents = resized_latents.reshape(resized_latents.shape[0], -1)

            latents = latents.reshape(latents.shape[0], -1)

            score = torch.norm(latents - resized_latents, dim=1).mean()
            score = -score
            scores.append(score.detach())
            
    scores = torch.stack(scores).unsqueeze(0)
    scoring = evaluate_scores(args, scores, batch)
    metrics.append(scoring)
    accuracy = sum(metrics) / len(metrics)
    print(f'Retrieval Accuracy: {accuracy}')