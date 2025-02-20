
import torch
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
import cv2

def get_bbox_pos(bbox, image_size):
    # what is the position of the bbox as a fraction of the image size
    x1, y1, x2, y2 = bbox
    x1_pct = x1 / image_size[0]
    y1_pct = y1 / image_size[1]
    x2_pct = x2 / image_size[0]
    y2_pct = y2 / image_size[1]
    return x1_pct, y1_pct, x2_pct, y2_pct

def mirror_image(image: Image.Image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def bbox_to_mask(bbox, image_size):
    mask = Image.new('L', (image_size[0], image_size[1]), 0)
    # Draw a white rectangle for the bounding box
    mask.paste(255, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
    return mask

def brighten_bbox(image, bbox, brightness=1.2):
    mask = bbox_to_mask(bbox, image.size)
    brightened_image = ImageEnhance.Brightness(image).enhance(brightness)
    result = Image.composite(brightened_image, image, mask)
    return result

def color_bbox(image, bbox, color=(255, 0, 0), intensity=0.2):
    mask = bbox_to_mask(bbox, image.size)
    mask = (np.asarray(mask) == 255).astype(np.uint8)
    mask = mask[:,:,None]
    colored_mask = mask * np.array(color)[None,None,:]
    image = np.asarray(image)
    image = image.astype(np.float32) * ( 1 - mask.astype(np.float32) * intensity) + colored_mask.astype(np.float32) * intensity
    return Image.fromarray(image.astype(np.uint8))

def thicken_bbox(image, bbox, color=(255, 0, 0), thickness=10):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for i in range(thickness):
        draw.rectangle(
            [bbox[0] - i, bbox[1] - i, bbox[2] + i, bbox[3] + i],
            outline=color
        )
    result = image
    return result
