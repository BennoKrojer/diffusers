import requests
import torch
from PIL import Image
from io import BytesIO
import os
from diffusers import AutoencoderKL
from torchvision.transforms.functional import to_pil_image


from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
model_id_or_path = "/home/krojerb/.cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/3beed0bcb34a3d281ce27bd8a6a1efbb68eada38"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    # revision="fp16", 
    # torch_dtype=torch.float16,
) # TODO try just modifying their code and saving the latents

# or download via git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
# and pass `model_id_or_path="./stable-diffusion-v1-5"`.
pipe = pipe.to(device)

# let's download an initial image
img_path = '/home/krojerb/img-gen-project/diffusers/animals/sketch-mountains-input.jpeg'#'/home/krojerb/img-gen-project/diffusers/winoground/images/ex_272_img_0.png'

init_image = Image.open(img_path).convert("RGB")
init_image = init_image.resize((768, 512))

prompt0 = "A fantasy landscape, trending on artstation"
# prompt1 = "using the guitar while the laptop is close by"

latent, images = pipe(prompt=prompt0, init_image=init_image, strength=0.8, guidance_scale=7.5)
fname = "fantasy mountain landscape_768_strengh=0.8"
images.images[0].save(f"./generated_images/{fname}.png")
torch.save(latent, f"./generated_images/{fname}_latent.pt")


vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae', use_auth_token='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')

def decode_latent(latent):
    image = vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    im = to_pil_image(image[0]) 
    return im

decode_latent(latent).save(f"./generated_images/{fname}_decoded.png")
