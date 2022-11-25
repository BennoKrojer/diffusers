import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
model_id_or_path = "./stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16", 
    torch_dtype=torch.float16,
)
# or download via git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
# and pass `model_id_or_path="./stable-diffusion-v1-5"`.
pipe = pipe.to(device)

# let's download an initial image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images

images[0].save("fantasy_landscape.png")