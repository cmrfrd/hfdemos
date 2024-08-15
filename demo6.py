#!/usr/bin/env python3
import enum
import json
import math
import random
import types
from collections import defaultdict
from itertools import product
from pathlib import Path
from random import shuffle
from textwrap import dedent
from typing import Any, Dict, Iterable, List, TypeVar, Union, get_args, get_origin

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import FluxPipeline
from PIL import Image
from pydantic import BaseModel

MODEL_ID = "black-forest-labs/FLUX.1-dev"
BASE_PATH = Path("data/guesswho_prod_run_4")
BASE_PATH.mkdir(parents=True, exist_ok=True)
NUM_IMAGES = 64
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


def dict_update(base: dict, update: dict) -> dict:
    """Update the base dictionary and return it."""
    base.update(update)
    return base


def is_type_constructor(obj: Any) -> bool:
    """Check if the object is a type constructor."""
    if isinstance(obj, (type, types.GenericAlias)):
        return True
    if hasattr(obj, "__origin__") or get_origin(obj) is not None:
        return True
    return False


def is_optional_type(type_hint: Any) -> bool:
    """Check if the type hint is an optional type."""
    return get_origin(type_hint) is Union and type(None) in get_args(type_hint)


def get_inner_type(type_hint: Any) -> Any:
    """Get the inner type of a type hint."""
    if is_optional_type(type_hint):
        return get_args(type_hint)[0]
    return type_hint


class BaseEnum(str, enum.Enum):
    pass


class Gender(BaseEnum):
    male = "male"
    female = "female"


class Weight(BaseEnum):
    light = "skinny weight"
    medium = "midweight"
    heavy = "overweight"


class Age(BaseEnum):
    young = "young"
    middle = "middle"
    old = "old"


class HairColor(BaseEnum):
    black = "black"
    brown = "brown"
    blonde = "blonde"
    gray = "gray"
    red = "red"


class HairStyle(BaseEnum):
    bald = "bald"
    short = "short"
    long = "long"
    straight = "straight"
    wavy = "wavy"
    curly = "curly"
    ponytail = "ponytail"
    dreadlocks = "dreadlocks"
    bun = "bun"
    afro = "afro"
    buzzcut = "buzzcut"
    bob = "bob"
    bangs = "bangs"


class EyeColor(BaseEnum):
    brown = "brown"
    blue = "blue"
    green = "green"


class Ethnicity(BaseEnum):
    african = "African-American"
    asian = "Asian-American"
    caucasian = "Caucasian"
    hispanic = "Hispanic-American"
    middle_eastern = "Middle-Eastern-American"


class SkinTone(BaseEnum):
    fair = "fair"
    medium = "medium"
    olive = "olive"
    dark = "dark"


class FacialHair(BaseEnum):
    beard = "beard"
    mustache = "mustache"
    clean_shaven = "clean shaven"


class Accessories(BaseEnum):
    none = "no accessories"
    glasses = "glasses"
    hat = "a hat"
    cap = "a cap"
    beanie = "a beanie"
    headband = "a headband"
    bandana = "a bandana"
    scarf = "a scarf"
    earrings = "earrings"


T = TypeVar("T", bound=BaseEnum)


def to_bidi_dict(d: Dict[T, List[T]]) -> Dict[T, List[T]]:
    """Convert a dict of enum -> list of enums to a bidirectional dict."""
    bidi_d: Dict[T, List[T]] = defaultdict(list)
    for k, v in d.items():
        bidi_d[k] = v
        for vv in v:
            bidi_d[vv].append(k)
    return bidi_d


CONSTRAINTS: Dict[BaseEnum, List[BaseEnum]] = to_bidi_dict(
    dict_update(
        defaultdict(list),
        {
            ## Men with earings looks fruity
            Gender.male: [Accessories.earrings],
            ## Image gen struggles with generating females with facial hair
            Gender.female: [
                FacialHair.beard,
                FacialHair.mustache,
            ],
            ## hair exceptions
            HairColor.gray: [Age.young, Age.middle, HairStyle.buzzcut, HairStyle.bald],
            ## Dark caucasian is rare and hard to generate
            Ethnicity.caucasian: [SkinTone.dark, SkinTone.olive],
            ## Light skin tone african/hispanic results in weird artifacts
            Ethnicity.african: [SkinTone.fair, SkinTone.medium, HairStyle.bangs, HairStyle.bob, HairColor.red],
            Ethnicity.hispanic: [SkinTone.fair, SkinTone.dark],
            Ethnicity.middle_eastern: [SkinTone.fair, SkinTone.dark],
            Ethnicity.asian: [SkinTone.medium, SkinTone.olive, SkinTone.dark],
            ## Don't use uncommon hair styles based on the ethnicity
            HairStyle.dreadlocks: [Ethnicity.asian, Ethnicity.middle_eastern],
            HairStyle.afro: [Ethnicity.asian, Ethnicity.middle_eastern],
            ## blue eye constraints
            EyeColor.blue: [Ethnicity.african, Ethnicity.asian, Ethnicity.hispanic, Ethnicity.middle_eastern],
        },
    )
)


class Attributes(BaseModel):
    gender: Gender
    age: Age
    weight: Weight
    hair_color: HairColor
    hair_style: HairStyle
    eye_color: EyeColor
    ethnicity: Ethnicity
    skin_tone: SkinTone
    facial_hair: FacialHair
    accessories: Accessories

    def prompt(self: "Attributes") -> str:
        """Generate a prompt from the attributes."""
        a = self
        prompt = dedent(
            f"""
            guess who game character,
            person,
            disney style,
            3d animated,
            light teal background,
            BREAK
            {a.gender.value} gender,
            {a.age.value} aged,
            {a.weight.value},
            BREAK
            {a.ethnicity.value} ethnicity
            with {a.skin_tone.value} skin tone,
            BREAK
            {a.hair_style.value} hair style,
            {a.hair_color.value} hair color,
            {f"{a.facial_hair.value} facial hair" if a.facial_hair != FacialHair.clean_shaven else ""}
            BREAK
            {a.eye_color.value} eyes,
            {f"wearing {a.accessories.value}" if a.accessories != Accessories.none else ""}
            """
        )

        prompt = prompt.replace("\n", " ").strip()
        return prompt

    @classmethod
    def all(cls) -> Iterable["Attributes"]:  # noqa: D102
        attr_names = sorted(cls.__annotations__.keys())
        attr_classes = [get_inner_type(cls.__annotations__[name]) for name in attr_names]
        all_possible_attrs = [list(a) for a in attr_classes]
        for items in product(*all_possible_attrs):
            asstributes = cls(**dict(zip(attr_names, items)))
            if satisfies_constraints(asstributes):
                yield asstributes


def satisfies_constraints(attributes: Attributes) -> bool:
    all_attrs = set(attributes.model_dump().values())
    for attr, constraints in CONSTRAINTS.items():
        if attr in all_attrs:
            for constraint in constraints:
                if constraint in all_attrs:
                    return False
    return True


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


def generate_images(worker: int, world_size: int) -> None:
    dist.init_process_group("nccl", rank=worker, world_size=world_size)

    shuffled_attrs = list(Attributes.all())
    shuffle(shuffled_attrs)
    all_attrs = shuffled_attrs[:NUM_IMAGES]

    device = torch.device(f"cuda:{worker}")
    print(f"Initializing worker {worker} on device {device}")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(worker)

    indices = range(worker, len(all_attrs), world_size)
    gpu_attrs = [all_attrs[i] for i in indices]

    for i, attributes in zip(indices, gpu_attrs):
        prompt = attributes.prompt()
        print(f"Worker {worker} generating image {i} for: {prompt}")
        result = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=6.5,
            num_inference_steps=40,
            max_sequence_length=512,
            generator=torch.Generator(device).manual_seed(SEED),
            num_images_per_prompt=9,
        )
        final_img = concat_nxn_imgs(result.images)
        final_img.save(BASE_PATH / f"{i}.png")
        (BASE_PATH / f"{i}.json").write_text(json.dumps(attributes.model_dump(), indent=4))

    dist.destroy_process_group()


if __name__ == "__main__":
    FluxPipeline.from_pretrained(MODEL_ID)  # ensure the model is downloaded
    world_size = torch.cuda.device_count()
    print(f"Intitializing with world size: {world_size}")
    mp.spawn(generate_images, args=(world_size,), nprocs=world_size)  # type: ignore
