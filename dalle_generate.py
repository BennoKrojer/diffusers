import json
import os
from pathlib import Path
from base64 import b64decode
import openai
openai.api_key = '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'

import datasets

winoground_data = json.load(open("datasets/winoground/data.json"))

for info in winoground_data:
    print(info["id"])
    idx = info["id"]
    if idx in [178, 319]: #inappropriate language
        continue
    caption0 = info["caption_0"]
    caption1 = info["caption_1"]

    if os.path.exists("cache/winoground/dalle_txt2img/" + str(idx) + "_1_seed_0.png"):
        continue

    response0 = openai.Image.create(prompt=caption0, n=5, size="512x512", response_format="b64_json")
    response1 = openai.Image.create(prompt=caption1, n=5, size="512x512", response_format="b64_json")
    response0 = json.dumps(response0)
    response0 = json.loads(response0)
    response1 = json.dumps(response1)
    response1 = json.loads(response1)
    for i in range(5):
        image_data = b64decode(response0["data"][i]["b64_json"])
        image_file = "cache/winoground/dalle_txt2img/" + str(idx) + "_0_seed_" + str(i) + ".png"
        with open(image_file, mode="wb") as png:
            png.write(image_data)

        image_data = b64decode(response1["data"][i]["b64_json"])
        image_file = "cache/winoground/dalle_txt2img/" + str(idx) + "_1_seed_" + str(i) + ".png"
        with open(image_file, mode="wb") as png:
            png.write(image_data)
    