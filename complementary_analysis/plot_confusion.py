import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles

# Define the file paths
file_diffusion_itm = 'vg_attribution_diffusion_classifier_2.1_seed0_steps100_subsetFalse_img_retrievalFalse_lora_mixed_grayFalse_predictions.json'
file_clip = 'vg_attribution_clip_baseline_rn50x64_seed0_steps250_subsetFalse_img_retrievalFalse_predictions.json'
file_blip = 'vg_attribution_blip_2.1_seed0_steps250_subsetFalse_img_retrievalFalse__grayFalse_predictions.json'

# Load the data
with open(file_diffusion_itm, 'r') as f:
    data_diffusion_itm = json.load(f)
with open(file_clip, 'r') as f:
    data_clip = json.load(f)
with open(file_blip, 'r') as f:
    data_blip = json.load(f)

# Create sets of correctly predicted examples for each model
correct_diffusion_itm = set([item['id'] for item in data_diffusion_itm if item['correct'] == 1])
correct_clip = set([item['id'] for item in data_clip if item['correct'] == 1])
correct_blip = set([item['id'] for item in data_blip if item['correct'] == 1])

# Create a Venn diagram for each pair of models and increase font size

plt.figure(figsize=(18, 6))

# Diffusion ITM and CLIP
plt.subplot(131)
venn = venn2([correct_diffusion_itm, correct_clip], ('Diffusion ITM', 'CLIP'))
venn_circles = venn2_circles([correct_diffusion_itm, correct_clip])

# Increase font size
for i, text in enumerate(venn.set_labels):
    text.set_fontsize(30)
    if i == 0:
        text.set_color('darkred')
    else:
        text.set_color('darkgreen')
for text in venn.subset_labels:
    text.set_fontsize(15)


# Diffusion ITM and BLIP
plt.subplot(132)
venn = venn2([correct_diffusion_itm, correct_blip], ('Diffusion ITM', 'BLIP'))
venn_circles = venn2_circles([correct_diffusion_itm, correct_blip])

# Increase font size
for i, text in enumerate(venn.set_labels):
    text.set_fontsize(30)
    if i == 0:
        text.set_color('darkred')
    else:
        text.set_color('darkgreen')
for text in venn.subset_labels:
    text.set_fontsize(15)

# CLIP and BLIP
plt.subplot(133)
venn = venn2([correct_clip, correct_blip], ('CLIP', 'BLIP'))
venn_circles = venn2_circles([correct_clip, correct_blip])

# Increase font size
for i, text in enumerate(venn.set_labels):
    text.set_fontsize(30)
    if i == 0:
        text.set_color('darkred')
    else:
        text.set_color('darkgreen')
for text in venn.subset_labels:
    text.set_fontsize(15)

plt.tight_layout()
plt.savefig('venn_diagrams_vg_attribution.png')
