import json
import requests as r
import mimetypes
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import numpy as np

ENDPOINT_URL="https://avcut4aqkmdzbn1d.eastus.azure.endpoints.huggingface.cloud" # url of your endpoint
HF_TOKEN="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" # organization token where you deployed your endpoint
path_to_raw_image = r"" # input your raw image path here in jpg format
path_to_processed_image = r"" # input the path where you wish to save processed image in jpg format
path_to_output_image = r"" # input the path where you wish to save output image in jpg format

image = Image.open(path_to_raw_image)
image = image.convert("RGB")
cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
image = Image.fromarray(cv2_img)
image.resize((850,850)).save(path_to_processed_image)

# image

def predict(path_to_image:str=None):
    with open(path_to_image, "rb") as i:
      b = i.read()
    headers= {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": mimetypes.guess_type(path_to_processed_image)[0]
    }
    response = r.post(ENDPOINT_URL, headers=headers, data=b)
    return response.json()

prediction = predict(path_to_image=path_to_processed_image)

# print(prediction)

# draw results on image
def draw_result(path_to_image, result):
    image = Image.open(path_to_image)
    image_size = image.size
    label2color = {
        "B-HEADER": "blue",
        "B-QUESTION": "red",
        "B-ANSWER": "green",
        "I-HEADER": "blue",
        "I-QUESTION": "red",
        "I-ANSWER": "green",
    }

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def normalize_bbox(bbox, size):
        return (
            float(bbox[0] / 850 * size[0]),
            float(bbox[1] / 850 * size[1]),
            float(bbox[2] / 850 * size[0]),
            float(bbox[3] / 850 * size[1]),
        )

    def normalize_textbox(bbox, size):
        return (
            float(bbox[0] / 850 * size[0]+10),
            float(bbox[1] / 850 * size[1]-10),
        )

    # debug
    for res in result:
        draw.rectangle(normalize_bbox(res["bbox"], image_size), outline="black")
        draw.rectangle(normalize_bbox(res["bbox"], image_size), outline=label2color[res["label"]])
        draw.text(normalize_textbox((res["bbox"][0] + 10, res["bbox"][1] - 10), image_size), text=res["label"], fill=label2color[res["label"]], font=font)
    return image

image2 = draw_result(path_to_raw_image, prediction["predictions"])
image2.save(path_to_output_image)