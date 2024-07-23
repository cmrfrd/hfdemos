#!/usr/bin/env python3
import math

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image


def concat_nxn_imgs(imgs: list[Image.Image]) -> Image.Image:
    """Concatenate images into a single image in a grid nxn pattern."""
    assert math.sqrt(len(imgs)).is_integer(), f"Number of images must be a square number, not {len(imgs)}"
    n = int(math.sqrt(len(imgs)))

    img_width, img_height = imgs[0].size
    assert all(img.size == (img_width, img_height) for img in imgs), "All images must have the same size"
    new_img = Image.new("RGB", (img_width * n, img_height * n))
    for i, img in enumerate(imgs):
        new_img.paste(img, (img_width * (i % n), img_height * (i // n)))
    return new_img


pipe: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe.to("cuda:0")  # type: ignore

prompt0 = "A cute happy robot holding a sign that says 'Welcome to GPTuesday!'"
prompt1 = "A high quality 1080p picture of Michael Jordan eating a human heart, blood on his face."
prompt2 = "The word 'potato' on a piece of paper, sans-serif."
result = pipe(  # type: ignore
    prompt0,
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
    num_images_per_prompt=4,
)
final_img = concat_nxn_imgs(result.images)
final_img.save("data/test_text.png")
