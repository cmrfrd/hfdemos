#!/usr/bin/env python3
import enum
import json
import math
import pprint
import random
import types
from collections import Counter, defaultdict
from itertools import product
from operator import mul
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, Iterable, List, Type, TypeVar, Union, get_args, get_origin

import pulp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import FluxPipeline
from diffusers.utils.logging import disable_progress_bar
from loguru import logger
from PIL import Image
from pydantic import BaseModel

disable_progress_bar()
T = TypeVar("T")
E = TypeVar("E")


def assert_fn(n: int, fn: Callable[[int], bool]) -> int:
    """Assert that n satisfies the function fn."""
    assert fn(n), f"{n} does not satisfy the function {fn}"
    return n


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


def invert_dict(d: Dict[T, E]) -> Dict[E, T]:
    """Invert a dictionary."""
    return {v: k for k, v in d.items()}


def random_color() -> str:
    class Colors(str, enum.Enum):
        red = "red"
        orange = "orange"
        yellow = "yellow"
        green = "green"
        blue = "blue"
        purple = "purple"
        pink = "pink"
        brown = "brown"
        black = "black"
        white = "white"
        gray = "gray"
        cyan = "cyan"
        magenta = "magenta"
        lime = "lime"
        indigo = "indigo"

    return random.choice(list(Colors)).value


def to_bidi_dict(d: Dict[T, List[T]]) -> Dict[T, List[T]]:
    """Convert a dict of enum -> list of enums to a bidirectional dict."""
    bidi_d: Dict[T, List[T]] = defaultdict(list)
    for k, v in d.items():
        bidi_d[k] = v
        for vv in v:
            bidi_d[vv].append(k)
    return bidi_d


class BaseEnum(str, enum.Enum): ...


class Gender(BaseEnum):
    male = "male"
    female = "female"


class Weight(BaseEnum):
    light = "skinny weight"
    medium = "midweight"
    heavy = "overweight"


class Age(BaseEnum):
    adult = "adult"
    elderly = "elderly"


class HairColor(BaseEnum):
    black = "black"
    brown = "brown"
    blonde = "blonde"
    gray = "gray"
    red = "red"


class HairStyle(BaseEnum):
    short = "short"
    long = "long"
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
    hazel = "hazel"


class SkinTone(BaseEnum):
    pale = "pale"
    normal = "normal"
    tan = "tan"
    dark = "dark"


class FacialHair(BaseEnum):
    none = "none"
    beard = "beard"
    mustache = "mustache"
    clean_shaven = "clean shaven"


class Accessories(BaseEnum):
    none = "none"
    glasses = "glasses"
    hat = "hat"
    cap = "cap"
    beanie = "beanie"
    headband = "headband"
    bandana = "bandana"
    scarf = "scarf"
    earrings = "earrings"


## Pairings that are not allowed
MANUAL_CONSTRAINTS: Dict[BaseEnum, List[BaseEnum]] = to_bidi_dict(
    dict_update(
        defaultdict(list),
        {
            ## Male characters with earings sometimes get generated as women
            Gender.male: [Accessories.earrings],
            ## Image gen struggles with generating females with facial hair
            Gender.female: [
                FacialHair.beard,
                FacialHair.mustache,
                FacialHair.clean_shaven,
            ],
            Gender.male: [
                FacialHair.none,
            ],
            ## Gray hair should be associated with old age
            HairColor.gray: [Age.adult],
        },
    )
)

## Ignore even distribution enums
IGNORE_EVEN_DISTRIBUTION: List[Type[BaseEnum]] = [FacialHair, HairStyle]


class Attributes(BaseModel):
    gender: Gender
    age: Age
    weight: Weight
    hair_color: HairColor
    hair_style: HairStyle
    eye_color: EyeColor
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
            close up,
            light teal background,
            BREAK
            {a.gender.value} gender,
            {a.age.value} aged,
            {a.weight.value},
            {a.skin_tone.value} skin tone,
            BREAK
            {a.hair_style.value} hair style,
            {a.hair_color.value} hair color,
            {f"{a.facial_hair.value}" if a.facial_hair != FacialHair.none else ""}
            BREAK
            {a.eye_color.value} eyes,
            {f"wearing {random_color()} {a.accessories.value}" if a.accessories != Accessories.none else ""}
            """
        )

        prompt = prompt.replace("\n", " ").strip()
        return prompt

    @classmethod
    def size(cls) -> int:
        """Get the product of the size of all attributes."""
        attr_names = cls.__annotations__.keys()
        attr_classes = [get_inner_type(cls.__annotations__[name]) for name in attr_names]
        return int(math.prod([len(a) for a in attr_classes]))

    @classmethod
    def all(cls) -> Iterable["Attributes"]:  # noqa: D102
        attr_names = sorted(cls.__annotations__.keys())
        attr_classes = [get_inner_type(cls.__annotations__[name]) for name in attr_names]
        all_possible_attrs = [list(a) for a in attr_classes]
        for items in product(*all_possible_attrs):
            attributes = cls(**dict(zip(attr_names, items)))
            if attrs_satisfies_constraints(attributes):
                yield attributes

    @classmethod
    def make_game(cls, n: int) -> List["Attributes"]:
        """Make a game with n people, ensuring fair distribution of attributes."""
        if n <= 0:
            raise ValueError("n must be greater than 0")
        if n > cls.size():
            raise ValueError("Cannot make a game with more people than there are unique attribute combinations")

        attr_names = list(cls.__annotations__.keys())
        attr_map = cls.__annotations__
        valid_game_problem = pulp.LpProblem("valid_game_problem", pulp.LpMinimize)

        ## Create the people as just a list of variables
        ## dehumanize the people!
        people: List[Dict[str, Dict[BaseEnum, pulp.LpVariable]]] = []
        for p_index in range(n):
            person = {}
            for attr_name in random.sample(attr_names, len(attr_names)):
                raw_attr_values = list(get_inner_type(attr_map[attr_name]))
                all_attr_values = random.sample(raw_attr_values, len(raw_attr_values))
                person[attr_name] = pulp.LpVariable.dicts(
                    f"person_{p_index}_{attr_name}",
                    [a.value for a in all_attr_values],
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpInteger,
                )

                ## constraint 1: each person has exactly one attribute for each attribute type
                valid_game_problem += pulp.lpSum(person[attr_name].values()) == 1
            people.append(person)

        ## constraint 2. The attributes should be 'evenly' distributed as much as possible
        ## Create a budget for each fine grain attribute
        ## then create a constraint based on the budget
        budgets: Dict[BaseEnum, int] = defaultdict(int)
        for attr_name in attr_names:
            attr_class = get_inner_type(cls.__annotations__[attr_name])
            for i in range(n):
                attr = list(attr_class)[i % len(attr_class)]
                budgets[attr] += 1

        for attr_name in attr_names:
            attr_class = get_inner_type(cls.__annotations__[attr_name])
            for v in list(attr_class):
                assert v in budgets, f"{v} not in budgets"
                if attr_class in IGNORE_EVEN_DISTRIBUTION:
                    valid_game_problem += pulp.lpSum([p[attr_name][v] for p in people]) >= 1
                    # valid_game_problem += pulp.lpSum([p[attr_name][v] for p in people]) <= budgets[v]
                else:
                    valid_game_problem += pulp.lpSum([p[attr_name][v] for p in people]) <= budgets[v]

        ## 3. Add manual constraints
        ## source and dest are the attributes that should not be together
        ##
        ## Convert enum->enum to attr_name + value -> attr_name + value
        attr_class_to_attr_name = invert_dict(attr_map)
        for attr, constraints in MANUAL_CONSTRAINTS.items():
            for constraint in constraints:
                source_attr_name, dest_attr_name = (
                    attr_class_to_attr_name[attr.__class__],
                    attr_class_to_attr_name[constraint.__class__],
                )
                source_attr_value, dest_attr_value = attr.value, constraint.value

                ## ensure that for all people, the pair is not selected
                ## or in this case that the sum of the pair is less than 2
                for person in people:
                    valid_game_problem += (
                        pulp.lpSum(
                            [person[source_attr_name][source_attr_value], person[dest_attr_name][dest_attr_value]]
                        )
                        <= 1
                    )

        valid_game_problem.solve()

        ## Now the solver has provided which attributes to use for each person
        ## collect them all and return the game
        game: List[Attributes] = []
        for person in people:
            raw_attrs: Dict[str, BaseEnum] = {}
            for attr_name in attr_names:
                result = max(person[attr_name].items(), key=lambda x: x[1].value())
                k, _ = result
                raw_attrs[attr_name] = k
            game.append(cls(**raw_attrs))  # type: ignore
        assert len(game) == n, f"Game size is {len(game)}, not {n}"
        return game


def attrs_satisfies_constraints(attributes: Attributes) -> bool:
    """Check if the attributes satisfy the constraints."""
    all_attrs = set(attributes.model_dump().values())
    for attr in all_attrs:
        if attr in MANUAL_CONSTRAINTS:
            for constraint in MANUAL_CONSTRAINTS[attr]:
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


def concat_mxn_imgs(imgs: list[Image.Image], n: int, m: int) -> Image.Image:
    """Concatenate images into a single image in a grid mxn pattern."""
    assert len(imgs) == n * m, f"Number of images must be {n}x{m}, not {len(imgs)}"
    img_width, img_height = imgs[0].size
    assert all(img.size == (img_width, img_height) for img in imgs), "All images must have the same size"
    new_img = Image.new("RGB", (img_width * m, img_height * n))
    for i, img in enumerate(imgs):
        new_img.paste(img, (img_width * (i % m), img_height * (i // m)))
    return new_img


def generate_images(worker: int, world_size: int, attrs: List[Attributes]) -> None:
    dist.init_process_group("nccl", rank=worker, world_size=world_size)

    device = torch.device(f"cuda:{worker}")
    logger.info(f"Initializing worker {worker} on device {device}")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(worker)

    indices = range(worker, len(attrs), world_size)
    all_attrs = [attrs[i] for i in indices]

    for i, attributes in zip(indices, all_attrs):
        prompt = attributes.prompt()
        logger.info(f"Worker {worker} generating image {i} for: {prompt}")
        result = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=5.0,
            num_inference_steps=40,
            max_sequence_length=512,
            generator=torch.Generator(device).manual_seed(SEED),
            num_images_per_prompt=NUM_SAMPLES,
        )

        ## Create all assets for person 'i'
        person_base_path = BASE_PATH / f"{i}"
        person_base_path.mkdir(parents=True, exist_ok=True)

        ## Write all samples individually
        for j, img in enumerate(result.images):
            img.save(person_base_path / f"sample_{j}.webp")

        ## Concatenate all images into a single image
        final_img = concat_nxn_imgs(result.images)
        final_img.save(person_base_path / "all.webp")

        ## Write the attributes to a json file
        (person_base_path / f"{i}.json").write_text(json.dumps(attributes.model_dump(), indent=4))

    dist.destroy_process_group()


def perc_of_attributes(attrs: List[Attributes]) -> Dict[str, Dict[str, float]]:
    """Get the distribution of attributes."""
    attr_names = sorted(Attributes.__annotations__.keys())
    attr_counts: Dict[str, Counter] = {attr_name: Counter() for attr_name in attr_names}
    for attr in attrs:
        for attr_name in attr_names:
            attr_counts[attr_name][attr.__getattribute__(attr_name)] += 1
    print(attr_counts)

    result = {}
    for attr_name, counter in attr_counts.items():
        total = sum(counter.values())
        result[attr_name] = {k: v / total for k, v in counter.items()}
    return result


MODEL_ID = "black-forest-labs/FLUX.1-dev"
BASE_PATH = Path("data/guesswho_run_game_4")
BASE_PATH.mkdir(parents=True, exist_ok=True)
NUM_IMAGES = 1024
NUM_SAMPLES = assert_fn(9, lambda n: math.sqrt(n).is_integer())
GAME_LAYOUT = (6, 6)
GAME_SIZE = mul(*GAME_LAYOUT)
SEED = 42

if __name__ == "__main__":
    logger.info("Generating images ...")

    world_size = torch.cuda.device_count()
    logger.info(f"Intitializing with world size: {world_size}")

    ## Generate a game
    game = Attributes.make_game(GAME_SIZE)
    pprint.pprint(perc_of_attributes(game))

    ## Generate images
    # mp.spawn(generate_images, args=(world_size, game), nprocs=world_size)  # type: ignore
    mp.spawn

    ## Concat all images into a single image
    all_imgs: List[Image.Image] = []
    for i in range(GAME_SIZE):
        person_base_path = BASE_PATH / f"{i}"
        num = random.randint(0, NUM_SAMPLES - 1)
        all_imgs.append(Image.open(person_base_path / f"sample_{num}.webp"))
    final_img = concat_mxn_imgs(all_imgs, *GAME_LAYOUT)
    final_img.save(BASE_PATH / "game.webp")

# # Some code for listing attrs
# shuffled_attrs = list(Attributes.all())
# shuffle(shuffled_attrs)
# shuffle(shuffled_attrs)
# for attrs in shuffled_attrs:
#     print(attrs.prompt())
#     input()

# ## Generate the full cartesian product of attributes
# shuffled_attrs = list(Attributes.all())
# shuffle(shuffled_attrs)
# shuffle(shuffled_attrs)
# all_attrs = shuffled_attrs[:NUM_IMAGES] if NUM_IMAGES > 0 else shuffled_attrs
# logger.info(f"There are {len(shuffled_attrs)} possible attributes sets, generating a {NUM_IMAGES} image subset")
