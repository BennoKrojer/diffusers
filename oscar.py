#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from PIL import Image
from tqdm.auto import tqdm


# In[ ]:


logging.set_verbosity_error()


# In[ ]:


vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet', use_auth_token='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')
scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)


# In[ ]:


vae.to(device='cuda', dtype=torch.bfloat16)
text_encoder.to(device='cuda', dtype=torch.bfloat16)
unet.to(device='cuda', dtype=torch.bfloat16)


# In[ ]:


prompt = ["a victorian portrait of a squirrel eating a bagel"]
height = 512
width = 512
num_inference_steps = 50
guidance_scale = 7.5
batch_size = len(prompt)


# In[ ]:


text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
with torch.autocast('cuda'):
    text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([''] * batch_size, padding='max_length', max_length=max_length, return_tensors='pt')
    uncond_embeddings = text_encoder(uncond_input.input_ids.to('cuda'))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


    # In[ ]:


    # generator = torch.manual_seed(0)
    generator = None

    latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), generator=generator, device='cuda', dtype=torch.bfloat16)
    latents = latents * scheduler.init_noise_sigma


    # In[ ]:


    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample


    # In[ ]:


    latents = 1 / 0.18215 * latents
    images = vae.decode(latents).sample


    # In[ ]:


    images = (images / 2 + 0.5).clamp(0, 1)
    to_pil_image(images[0])
