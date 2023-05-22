from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image, ImageDraw, ImageFont
import textwrap
import pandas as pd

# Read the csv file
df = pd.read_csv("drawbench.csv")
texts = df.iloc[:, 0].tolist()  # Get the first column of the csv file and convert it to a list

model_id = "stabilityai/stable-diffusion-2-1-base"

LORA = False
generator = torch.Generator("cuda").manual_seed(42)

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda:7")
if LORA:
    pipe.unet.load_attn_procs('mixed_neg1.0_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-4000')

# Define a function to calculate the height of multiline text
def multiline_textsize(text, font):
    total_height = 0  # total height of all lines
    line_width, line_height = font.getsize(text)
    total_height += line_height
    return line_width, total_height

for i, prompt in enumerate(texts):
    image = pipe(prompt).images[0]
    image = image.convert("RGB")  # Convert image to RGB mode if it isn't

    # Load a font
    font_size = 30  # Set larger font size
    font = ImageFont.load_default()

    # Wrap the text into multiple lines if it's too wide
    max_text_width = image.width - 20  # Set some padding
    lines = textwrap.wrap(prompt, width=40)  # Adjust the width as needed

    # Calculate total height of the text
    total_text_height = sum(font.getsize(line)[1] for line in lines)

    # Add a white rectangle at the bottom for the text
    img_draw = ImageDraw.Draw(image)
    img_draw.rectangle(
        [(0, image.height - total_text_height - 20), 
            (image.width, image.height)], 
        fill="white"
    )

    # Write the text into the rectangle
    y_text = image.height - total_text_height - 10
    for line in lines:
        width, height = font.getsize(line)
        img_draw.text(((image.width - width) / 2, y_text), line, font=font, fill="black")
        y_text += height

    image.save(f"generated_drawbench_seed42/{i}_{'finetuned' if LORA else 'zeroshot'}.png")
