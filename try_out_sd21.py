from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1-base"

#seed
generator = torch.Generator(device=accelerator.device).manual_seed(seed)

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "'Two young guys with shaggy hair look at their hands while hanging out in the yard'"
# prompt = 'dog'
image = pipe(prompt).images[0]  
    
image.save("flickr30k_0.png")
