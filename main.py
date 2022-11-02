import torch
from src.diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
print("made it")
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
# print(prompt)
image = pipe(prompt).images[0]  
# print('saving')
# im.save(f"./generated_images/{prompt.replace(' ','_')}.png")