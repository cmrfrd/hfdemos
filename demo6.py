#!/usr/bin/env python3
import enum
import json
import math
import pprint
import random
import types
import typing
from collections import Counter, defaultdict
from itertools import combinations, product
from operator import mul
from pathlib import Path
from textwrap import dedent
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import cvxpy as cp
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
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


def to_bidi_dict(d: Dict[T, List[T]]) -> Dict[T, List[T]]:
    """Convert a dict of enum -> list of enums to a bidirectional dict."""
    bidi_d: Dict[T, List[T]] = defaultdict(list)
    for k, v in d.items():
        bidi_d[k] = v
        for vv in v:
            bidi_d[vv].append(k)
    return bidi_d


def categorical_cmap(nc: int, nsc: int, cmap: str, continuous: bool = False) -> matplotlib.colors.ListedColormap:
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * nsc : (i + 1) * nsc, :] = rgb
    return matplotlib.colors.ListedColormap(cols)


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


def entropy(probabilities: List[float]) -> float:
    """Calculate the entropy from a list of probabilities."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


class BaseEnum(enum.Enum):
    def __str__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}.{self.name}"

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}.{self.name}"

    def to_json(self) -> Any:  # noqa: D102
        return self.value

    @classmethod
    def from_json(cls: Type["BaseEnum"], value: Any) -> "BaseEnum":  # noqa: D102
        return cls(value)


class EnumEncoder(json.JSONEncoder):
    def default(self: "EnumEncoder", obj: Any) -> Any:  # noqa: D102
        if isinstance(obj, BaseEnum):
            return obj.to_json()
        return super().default(obj)


class BaseAttributes(BaseModel):
    NAME: ClassVar[str]
    ANTI_AFFINITY_CONSTRAINTS: ClassVar[Dict[BaseEnum, List[BaseEnum]]]

    @classmethod
    def __init_subclass__(cls: Type["BaseAttributes"], **kwargs: Any) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "NAME"):
            raise TypeError(f"Class {cls.__name__} must define 'NAME'")
        if not hasattr(cls, "ANTI_AFFINITY_CONSTRAINTS"):
            raise TypeError(f"Class {cls.__name__} must define 'ANTI_AFFINITY_CONSTRAINTS'")

    def prompt(self) -> str:
        """Generate a prompt from the attributes."""
        raise NotImplementedError("Subclass must implement this method")

    def attrs_satisfies_constraints(self) -> bool:
        """Check if the attributes satisfy the anti-affinity constraints."""
        cls = self.__class__
        attr_dump = {self.__getattribute__(attr) for attr in cls.get_attrs().keys()}
        assert all(isinstance(a, BaseEnum) for a in attr_dump), "Attributes must be of type BaseEnum"
        for attr, constraints in cls.ANTI_AFFINITY_CONSTRAINTS.items():
            for constraint in constraints:
                pair = (attr, constraint)
                if (pair[0] in attr_dump) and (pair[1] in attr_dump):
                    return False
        return True

    @classmethod
    def get_attrs(cls) -> Dict[str, BaseEnum]:  # noqa: D102
        attr_dict = {}
        for attr_name, attr_class in cls.__annotations__.items():
            if isinstance(attr_class, (typing._GenericAlias, typing._SpecialGenericAlias)):
                continue
            if issubclass(attr_class, BaseEnum):
                attr_dict[attr_name] = attr_class
        assert len(attr_dict) > 0, "No attributes found"
        return attr_dict

    @classmethod
    def size(cls) -> int:
        """Get the product of the size of all attributes."""
        attr_names = cls.get_attrs().keys()
        attr_classes = [get_inner_type(cls.get_attrs()[name]) for name in attr_names]
        print(attr_classes)
        return int(math.prod([len(a) for a in attr_classes]))

    @classmethod
    def all(cls) -> Iterable["BaseAttributes"]:  # noqa: D102
        attr_names = sorted(cls.get_attrs().keys())
        attr_classes = [get_inner_type(cls.get_attrs()[name]) for name in attr_names]
        all_possible_attrs = [list(a) for a in attr_classes]
        for items in product(*all_possible_attrs):
            attributes = cls(**dict(zip(attr_names, items)))
            if attributes.attrs_satisfies_constraints():
                yield attributes

    @classmethod
    def random_sample(cls, n: int) -> List["BaseAttributes"]:  # noqa: D102
        attr_names = random.sample(list(cls.get_attrs().keys()), len(cls.get_attrs().keys()))
        attr_classes = [get_inner_type(cls.get_attrs()[name]) for name in attr_names]
        all_possible_attrs = [list(a) for a in attr_classes]
        valid_attrs = []
        while len(valid_attrs) < n:
            items = [random.choice(attrs) for attrs in all_possible_attrs]
            attributes = cls(**dict(zip(attr_names, items)))
            if attributes.attrs_satisfies_constraints() and attributes not in valid_attrs:
                valid_attrs.append(attributes)
        return valid_attrs

    @classmethod
    def create_game(cls, n: int) -> List["BaseAttributes"]:
        """Make a game with n collections, ensuring fair distribution of attributes."""
        assert n > 0, "n must be greater than 0"
        assert n <= cls.size(), "Cannot make a game with more people than there are unique attribute combinations"

        attr_names = random.sample(list(cls.get_attrs().keys()), len(cls.get_attrs().keys()))

        # set_param("verbose", 5)
        # set_param("parallel.enable", True)

        # opt = Optimize()
        # enum_to_vars = {}

        # count_pairs = 0
        # for attr_name in attr_names:
        #     attr_class = cls.get_attrs()[attr_name]
        #     count_pairs += len(attr_class)
        # all_state = BitVec("all_state", n * count_pairs)

        # # how to index
        # # person i
        # # attribute j
        # def get_person(i: int) -> BitVec:
        #     return Extract((i + 1) * count_pairs - 1, i * count_pairs, all_state)

        # def get_person_attr_class(i: int, attr_class: Type[BaseEnum]) -> BitVec:
        #     person = get_person(i)  # bitvec of size count_pairs
        #     start_index = 0
        #     for attr_name_iter in attr_names:
        #         if attr_name_iter == invert_dict(cls.get_attrs())[attr_class]:
        #             break
        #         start_index += len(cls.get_attrs()[attr_name_iter])
        #     return Extract(start_index + len(attr_class) - 1, start_index, person)

        # def get_person_attr_class_attr(i: int, attr_class: Type[BaseEnum], attr_value: BaseEnum) -> BitVec:
        #     person_attr_class = get_person_attr_class(i, attr_class)
        #     index = list(attr_class).index(attr_value)
        #     return Extract(index, index, person_attr_class)

        # # Enforce that each collection has exactly one value for each attribute
        # for i in range(n):  # Iterate over each person
        #     for attr_name in attr_names:
        #         person_attr = get_person_attr_class(i, cls.get_attrs()[attr_name])
        #         opt.add(Sum(person_attr) == 1)
        # print("Enforced oneof")

        # # Enforce anti-affinity constraints
        # for attr, constraints in cls.ANTI_AFFINITY_CONSTRAINTS.items():
        #     for constraint in constraints:
        #         for i in range(n):
        #             # Ensure both attributes are not selected together in the same collection
        #             print(attr, constraint, i)
        #             opt.add(
        #                 Not(
        #                     And(
        #                         get_person_attr_class_attr(i, attr.__class__, attr) == 1,
        #                         get_person_attr_class_attr(i, constraint.__class__, constraint) == 1,
        #                     )
        #                 )
        #             )
        # print("Enforced anti-affinity")

        # # Enforce even distribution of each attribute's values
        # objective_terms = []
        # for attr_name in attr_names:
        #     attr_class = cls.get_attrs()[attr_name]

        #     # Convert BitVec sums to Int
        #     sum_each_enum_elem = []
        #     for attr_value in attr_class:
        #         all_extracted = []
        #         for i in range(n):
        #             extracted = get_person_attr_class_attr(i, attr_class, attr_value)
        #             all_extracted.append(If(extracted == 1, 1, 0))
        #         sum_each_enum_elem.append(Sum(all_extracted))
        #     print(sum_each_enum_elem)
        #     # Sum([If(Extract(j, j, vars[i]) == 1, 1, 0) for i in range(n)]) for j in range(num_values)

        #     upper = Int(f"upper_{attr_name}")
        #     lower = Int(f"lower_{attr_name}")
        #     opt.add(upper - lower >= 0)
        #     objective_terms.append(upper - lower)
        #     for j in range(len(sum_each_enum_elem)):
        #         opt.add(lower <= sum_each_enum_elem[j])
        #         opt.add(sum_each_enum_elem[j] <= upper)
        # print("Enforced even distribution")

        # # Set the total objective to minimize
        # total_objective = Sum(objective_terms)
        # opt.minimize(total_objective)

        # # Solve the problem
        # if opt.check() == sat:
        #     model = opt.model()
        #     # Extract and return the solutions as BaseAttributes instances
        #     # (Assuming BaseAttributes can be constructed from the model)
        #     result = []
        #     for i in range(n):
        #         attributes = {}
        #         for attr_name in attr_names:
        #             attr_class = cls.get_attrs()[attr_name]
        #             vars = enum_to_vars[attr_class]
        #             num_values = len(attr_class)
        #             for j in range(num_values):
        #                 var = vars[i][j]
        #                 if is_true(model.evaluate(var)):
        #                     attribute_value = sorted(attr_class, key=lambda x: x.name)[j]
        #                     attributes[attr_name] = attribute_value
        #                     break
        #         result.append(cls(**attributes))
        #     return result
        # else:
        #     raise ValueError("Problem is infeasible")

        objective = cp.Minimize(0)
        all_constraints: List[cp.Constraint] = []
        debug_vars: Dict[str, cp.Variable] = {}

        ## Map "enum" -> "constraint variables"
        enum_to_vars: Dict[Type[BaseEnum], cp.Variable] = {}

        ## Create all the variables. One per element, per enum, per collection.
        for attr_name in attr_names:
            attr_class: Type[BaseEnum] = cls.get_attrs()[attr_name]
            enum_to_vars[attr_class] = cp.Variable(name=f"{str(attr_class)}", shape=(n, len(attr_class)), boolean=True)

        ## Enforce the base constraints:
        ## - every variable must be 0 or 1
        ## - the sum of all variables for an enum must be 1 (oneof constraint)
        for _, var in enum_to_vars.items():
            all_constraints += [cp.sum(var, axis=1) == 1]

        ## Enforce anti-affinity: certain attributes should not be together
        ## apply this for all collections
        for attr, constraints in cls.ANTI_AFFINITY_CONSTRAINTS.items():
            for constraint in constraints:
                source_index = sorted(attr.__class__, key=lambda x: x.name).index(attr)
                dest_index = sorted(constraint.__class__, key=lambda x: x.name).index(constraint)

                source_vars = enum_to_vars[attr.__class__]
                dest_vars = enum_to_vars[constraint.__class__]

                all_constraints.append(source_vars[:, source_index] + dest_vars[:, dest_index] <= 1)

        ## Enforce the sum of each enum is as even as possible
        for attr_name in attr_names:
            attr_class = cls.get_attrs()[attr_name]
            sum_each_enum_elem = cp.sum(enum_to_vars[attr_class], axis=0)

            upper = cp.Variable(name=f"upper_{attr_name}", shape=1, integer=True)
            lower = cp.Variable(name=f"lower_{attr_name}", shape=1, integer=True)
            all_constraints.append(upper - lower >= 0)
            all_constraints.append(lower <= sum_each_enum_elem)
            all_constraints.append(sum_each_enum_elem <= upper)
            objective += cp.Minimize((upper - lower) * 100_000)

        # ## Enforce that the count of collections with enum pairs is as even as possible
        # for i, (a, b) in list(enumerate(combinations(attr_names, 2))):
        #     a_class = get_inner_type(cls.__annotations__[a])
        #     b_class = get_inner_type(cls.__annotations__[b])
        #     product_ab = list(product(list(a_class), list(b_class)))

        #     max_c = cp.Variable(name=f"max_c_{a}_{b}", shape=1, integer=True)
        #     c = cp.Variable(name=f"c_{a}_{b}", shape=(len(product_ab), n), boolean=True)
        #     for i_a in range(len(a_class)):
        #         for i_b in range(len(b_class)):
        #             all_constraints += [
        #                 c[i_a * len(b_class) + i_b, :] <= enum_to_vars[a_class][:, i_a],
        #                 c[i_a * len(b_class) + i_b, :] <= enum_to_vars[b_class][:, i_b],
        #                 enum_to_vars[a_class][:, i_a] + enum_to_vars[b_class][:, i_b]
        #                 <= c[i_a * len(b_class) + i_b, :] + 1,
        #             ]
        #     all_constraints += [cp.sum(c, axis=1) <= max_c]
        #     objective += cp.Minimize(max_c)

        ## Run the solver and assert failurs
        problem = cp.Problem(objective, all_constraints)
        problem.solve(
            verbose=True,
            solver=cp.CBC,
            canon_backend=cp.SCIPY_CANON_BACKEND,
            maximumSeconds=60 * 5,
        )

        ## Print debug vars if they exist
        if debug_vars:
            print(f"Debug vars: {debug_vars}")
            for k, v in debug_vars.items():
                print(f"{k}:\n{v.value}")

        match problem.status:
            case cp.OPTIMAL:
                pass
            case cp.INFEASIBLE:
                raise ValueError("Problem is infeasible")
            case cp.UNBOUNDED:
                raise ValueError("Problem is unbounded")
            case _:
                print(problem.status)

        game: List[BaseAttributes] = []
        for i in range(n):
            raw_attrs: Dict[str, BaseEnum] = {}
            for attr_name, attr_class in cls.get_attrs().items():
                v = enum_to_vars[attr_class].value[i, :]  # Get the i-th row
                max_index = np.argmax(v)
                enum_value = sorted(get_inner_type(attr_class), key=lambda x: x.name)[max_index]
                raw_attrs[attr_name] = enum_value
            game.append(cls(**raw_attrs))  # type: ignore

        for i, elem in enumerate(game):
            assert elem.attrs_satisfies_constraints(), f"Person {i} does not satisfy constraints:\n\n{elem}"
        assert len(game) == n, f"Game size is {len(game)}, not {n}"
        return game


class Gender(BaseEnum):
    male = "male"
    female = "female"


class Weight(BaseEnum):
    light = "skinny weight"
    medium = "normal weight"
    heavy = "overweight"


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
    bald = "bald"
    wavy = "wavy"
    ponytail = "ponytail"
    braided = "braided"


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
    stubble = "stubble"
    beard = "beard"
    mustache = "mustache"
    clean_shaven = "clean shaven"


class Accessories(BaseEnum):
    none = "none"
    beanie = "beanie"
    glasses = "glasses"
    earrings = "earrings"
    hat = "wide brim hat"
    cap = "cap"


class MainColors(BaseEnum):
    red = "red"
    orange = "orange"
    yellow = "yellow"
    green = "green"
    blue = "blue"
    violet = "violet"


class AccentColors(BaseEnum):
    red = "red"
    orange = "orange"
    yellow = "yellow"
    green = "green"
    blue = "blue"
    violet = "violet"


class Attributes(BaseAttributes):
    gender: Gender
    weight: Weight
    hair_color: HairColor
    hair_style: HairStyle
    eye_color: EyeColor
    skin_tone: SkinTone
    facial_hair: FacialHair
    accessories: Accessories
    color: MainColors
    accent: AccentColors

    NAME: ClassVar[str] = "Guess Who"
    ANTI_AFFINITY_CONSTRAINTS: ClassVar[Dict[BaseEnum, List[BaseEnum]]] = to_bidi_dict(
        dict_update(
            defaultdict(list),
            {
                Gender.female: [
                    FacialHair.beard,
                    FacialHair.mustache,
                    FacialHair.clean_shaven,
                    FacialHair.stubble,
                    HairStyle.bald,
                ],
                Gender.male: [FacialHair.none, Accessories.earrings, HairStyle.braided],
                SkinTone.dark: [HairColor.red, HairColor.blonde],
            },
        )
    )

    def prompt(self: "Attributes") -> str:
        """Generate a prompt from the attributes."""
        a = self
        # prompt = dedent(
        #     f"""
        #     guess who game character,
        #     person,
        #     disney style,
        #     3d animated,
        #     close up,
        #     light teal background,
        #     BREAK
        #     {a.gender.value} gender,
        #     {a.weight.value},
        #     {a.skin_tone.value} skin tone,
        #     BREAK
        #     {a.hair_style.value} hair style,
        #     {a.hair_color.value} hair color,
        #     {f"{a.facial_hair.value}" if a.facial_hair != FacialHair.none else ""}
        #     BREAK
        #     {a.eye_color.value} eyes,
        #     {f"wearing {a.color.value} {a.accessories.value}" if a.accessories != Accessories.none else ""},
        #     {f"with {a.accent.value} clothes"}
        #     """
        # )

        # prompt = dedent(
        #     f"""
        #     guess who game character,
        #     adult person,
        #     hand drawn watercolor style,
        #     high contrast color palette,
        #     close up,
        #     bright yellow background,
        #     BREAK
        #     {a.gender.value} gender,
        #     {a.weight.value},
        #     {a.skin_tone.value} skin tone,
        #     BREAK
        #     {a.hair_style.value} hair style,
        #     {a.hair_color.value} hair color,
        #     {f"{a.facial_hair.value}" if a.facial_hair != FacialHair.none else ""}
        #     BREAK
        #     {a.eye_color.value} eyes,
        #     {f"wearing {a.color.value} {a.accessories.value}" if a.accessories != Accessories.none else ""},
        #     {f"with {a.accent.value} clothes"}
        #     """
        # )

        # prompt = dedent(
        #     f"""
        #     ultra-realistic dslr portrait photo,
        #     full head shot photo,
        #     blurred background,
        #     BREAK
        #     {a.gender.value} gender person,
        #     {a.weight.value},
        #     {a.skin_tone.value} skin tone,
        #     BREAK
        #     {a.hair_style.value} hair style,
        #     {a.hair_color.value} hair color,
        #     {f"{a.facial_hair.value}" if a.facial_hair != FacialHair.none else ""}
        #     BREAK
        #     {a.eye_color.value} eyes,
        #     {f"wearing {a.color.value} {a.accessories.value}" if a.accessories != Accessories.none else ""},
        #     {f"with {a.accent.value} clothes"}
        #     BREAK
        #     high detail photo realistic style
        #     """
        # )

        prompt = dedent(
            f"""
            realistic renaissance portrait of a {a.gender.value} person,
            high detail realistic,
            in the style of Giotto di Bondone, Sandro Botticelli, Leonardo da Vinci, Michelangelo Buonarroti,
            BREAK
            {a.gender.value} gender person,
            {a.weight.value},
            {a.skin_tone.value} skin tone,
            BREAK
            {a.hair_style.value} hair style,
            {a.hair_color.value} hair color,
            {f"{a.facial_hair.value}" if a.facial_hair != FacialHair.none else ""}
            BREAK
            {a.eye_color.value} eyes,
            {f"wearing {a.color.value} {a.accessories.value}" if a.accessories != Accessories.none else ""},
            {f"with {a.accent.value} clothes"}
            """
        )
        prompt = prompt.replace("\n", " ").strip()
        return prompt


class NumberOfEyes(BaseEnum):
    two = "two"
    three = "three"
    four = "four"


class MonsterColor(BaseEnum):
    red = "deep red"
    blue = "blue"
    green = "green"
    pink = "pink"
    yellow = "yellow"


class MonsterAppearance(BaseEnum):
    cute = "cute"
    ugly = "ugly"


class MonsterSkin(BaseEnum):
    smooth = "smooth"
    hairy = "hairy"
    polka_dots = "polka dots"
    snake_skin = "snake"


class Crown(BaseEnum):
    horns = "horns"
    antennae = "antennae"
    none = "none"


class Monsters(BaseAttributes):
    number_of_eyes: NumberOfEyes
    color: MonsterColor
    appearance: MonsterAppearance
    skin: MonsterSkin
    crown: Crown

    NAME: ClassVar[str] = "Guess Who Monster"
    ANTI_AFFINITY_CONSTRAINTS: ClassVar[Dict[BaseEnum, List[BaseEnum]]] = to_bidi_dict(
        dict_update(
            defaultdict(list),
            {},
        )
    )

    def prompt(self: "Monsters") -> str:
        """Generate a prompt from the attributes."""
        a = self
        prompt = dedent(
            f"""
            A {a.appearance.value} looking monster with {a.number_of_eyes.value} eye(s),
            guess who game character portrait,
            hand drawn watercolor style,
            high contrast color palette,
            close up,
            light teal background,
            BREAK
            {a.color.value} skin color,
            {a.skin.value} skin,
            {a.crown.value if a.crown != Crown.none else ""}
            """
        )
        prompt = prompt.replace("\n", " ").strip()
        return prompt


class PickleShape(BaseEnum):
    uncut = "uncut"
    sliced = "sliced"
    diced = "diced"
    spear = "spear"


class PickleTexture(BaseEnum):
    smooth = "smooth"
    bumpy = "bumpy"


class PickleColor(BaseEnum):
    yellowish = "yellowish"
    dark_green = "dark green"
    light_green = "light green"


class PickleGirth(BaseEnum):
    skinny = "skinny"
    normal = "normal"
    thick = "thick"


class Pickles(BaseAttributes):
    shape: PickleShape
    texture: PickleTexture
    color: PickleColor
    girth: PickleGirth

    NAME: ClassVar[str] = "Guess Who Pickle"
    ANTI_AFFINITY_CONSTRAINTS: ClassVar[Dict[BaseEnum, List[BaseEnum]]] = to_bidi_dict(
        dict_update(
            defaultdict(list),
            {},
        )
    )

    def prompt(self: "Monsters") -> str:
        """Generate a prompt from the attributes."""
        a = self
        prompt = dedent(
            f"""
            A {a.shape.value} {a.texture.value} pickle with {a.girth.value} girth,
            hand drawn watercolor style,
            high contrast color palette,
            close up,
            light teal background,
            BREAK
            {a.color.value} color,
            accurate representation of a pickle,
            ultra realistic,
            """
        )
        prompt = prompt.replace("\n", " ").strip()
        return prompt


def generate_images(worker: int, world_size: int, attrs: List[BaseAttributes]) -> None:
    dist.init_process_group("nccl", rank=worker, world_size=world_size)

    device = torch.device(f"cuda:{worker}")
    logger.info(f"Initializing worker {worker} on device {device}")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(worker)

    indices = range(worker, len(attrs), world_size)
    all_attrs = [attrs[i] for i in indices]

    for i, attributes in zip(indices, all_attrs):
        ## Create all assets for person 'i'
        person_base_path = BASE_PATH / f"{i}"
        person_base_path.mkdir(parents=True, exist_ok=True)

        ## Write the attributes to a json file
        (person_base_path / f"{i}.json").write_text(attributes.model_dump_json(indent=4))

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

        ## Write all samples individually
        for j, img in enumerate(result.images):
            img.save(person_base_path / f"sample_{j}.webp")

        ## Concatenate all images into a single image
        final_img = concat_nxn_imgs(result.images)
        final_img.save(person_base_path / "all.webp")

    dist.destroy_process_group()


def perc_of_attributes(attrs: List[BaseAttributes]) -> Dict[str, Dict[str, float]]:
    """Get the distribution of attributes."""
    cls = attrs[0].__class__
    attr_names = sorted(cls.get_attrs().keys())
    attr_counts: Dict[str, Counter] = {attr_name: Counter() for attr_name in attr_names}
    for attr in attrs:
        for attr_name in attr_names:
            attr_counts[attr_name][attr.__getattribute__(attr_name)] += 1

    result = {}
    for attr_name, counter in attr_counts.items():
        total = sum(counter.values())
        result[attr_name] = {k: v / total for k, v in counter.items()}
    return result


def plot_attribute_dist(attrs: List[BaseAttributes]) -> None:
    """Plot the distribution of attributes."""
    data = perc_of_attributes(attrs)

    fig, ax = plt.subplots(figsize=(12, 8))

    categories = list(data.keys())
    num_subcategories = [len(data[cat]) for cat in categories]
    max_subcategories = max(num_subcategories)

    cmap = categorical_cmap(len(categories), max_subcategories, cmap="tab20")
    x = np.arange(len(categories))
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylim(0, 1)

    legend_labels = []
    legend_handles = []
    bar_width = 0.8 / max_subcategories
    for i, cat in enumerate(categories):
        subcategories = list(data[cat].keys())
        subcategory_values = list(data[cat].values())
        dotted_line_height = 1 / len(subcategories)

        for j, subcat in enumerate(subcategories):
            bar = ax.bar(
                i + j * bar_width, subcategory_values[j], bar_width, label=subcat, color=cmap(i * max_subcategories + j)
            )
            if j == 0:
                legend_handles.append(bar[0])
                legend_labels.append(cat)

        dotted_line_handle = ax.axhline(
            y=dotted_line_height,
            color="black",
            linestyle="--",
            linewidth=2,
            xmin=i / len(categories) + 0.03,
            xmax=(i + 1) / len(categories) - 0.03,
        )

    legend_handles.append(dotted_line_handle)
    legend_labels.append("Theoretical Optimal")

    # Add labels and title
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_title('Distribution of "Guess Who" Attributes', fontsize=16)
    ax.legend(handles=legend_handles, labels=legend_labels, loc="upper right", bbox_to_anchor=(1, 1))
    plt.subplots_adjust(bottom=0.2, right=0.8)
    plt.savefig("data/attribute_distribution.png")


def plot_pairwise_attribute_dist(collections: List[BaseAttributes]) -> None:
    """Plot the distribution of pairwise attributes."""
    cls = collections[0].__class__
    attr_names = cls.get_attrs().keys()
    all_counts: Dict[Tuple[BaseEnum, ...], int] = {}
    attr_name_pairs = list(combinations(attr_names, 2))

    for collection in collections:
        for a, b in attr_name_pairs:
            attr_a = collection.__getattribute__(a)
            attr_b = collection.__getattribute__(b)
            key: List[BaseEnum] = sorted([attr_a, attr_b], key=lambda x: x.value)
            if tuple(key) not in all_counts:
                all_counts[tuple(key)] = 1
            else:
                all_counts[tuple(key)] += 1

    labels = [v for attr_name in attr_names for v in list(cls.get_attrs()[attr_name])]
    data = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for pair, count in all_counts.items():
        data[labels.index(pair[0]), labels.index(pair[1])] += count
        data[labels.index(pair[1]), labels.index(pair[0])] += count
    data[np.triu_indices(len(labels))] = 0
    data[np.diag_indices(len(labels))] = -1
    for attr_class in (cast(Type[BaseEnum], e) for e in cls.get_attrs().values()):
        for x, y in combinations(attr_class, 2):
            data[labels.index(x), labels.index(x)] = -1
            data[labels.index(y), labels.index(y)] = -1
            data[labels.index(x), labels.index(y)] = -1
            data[labels.index(y), labels.index(x)] = -1

    cmap = plt.cm.viridis
    cmap.set_bad(color="red")

    _, ax = plt.subplots(figsize=(16, 12))

    masked_data = np.ma.masked_where(data == -1, data)
    _ = ax.imshow(masked_data, cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=float(np.max(data))))

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Pairwise Distribution of Attributes", fontsize=14)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = data[i, j]
            # Use a threshold value to switch text color
            text_color = "black" if value >= 5 else "white"
            ax.text(j, i, str(int(value)), ha="center", va="center", color=text_color, fontsize=8)

    # Draw outline around each cell in the lower triangular portion
    section_boundaries = {}
    start_idx = 0
    for attr_name in attr_names:
        end_idx = start_idx + len(list(cls.get_attrs()[attr_name]))
        section_boundaries[attr_name] = (start_idx, end_idx)
        start_idx = end_idx

    for attr1, attr2 in combinations(attr_names, 2):
        start1, end1 = section_boundaries[attr1]
        start2, end2 = section_boundaries[attr2]
        if start1 < start2:  # Ensure we are in the lower triangle
            rect = patches.Rectangle(
                (start1 - 0.5, start2 - 0.5),
                end1 - start1,
                end2 - start2,
                linewidth=2,
                edgecolor="yellow",
                facecolor="none",
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig("data/pairwise_attribute_distribution.png")


def plot_gini_coefficients(attrs: List[BaseAttributes]) -> None:
    """Plot the Gini coefficients of the attributes."""
    cls = attrs[0].__class__

    def gini_coefficient(sequence) -> float:
        assert len(sequence) > 0, "Sequence must not be empty"
        n = len(sequence)
        cumulative_sum = np.cumsum(np.sort(sequence), dtype=float)
        assert cumulative_sum[-1] != 0, "Sum of sequence must not be zero"
        gini = (n + 1 - 2 * np.sum(cumulative_sum) / cumulative_sum[-1]) / n
        return gini

    attr_names = cls.get_attrs().keys()
    gini_coeffs = {}
    sequences = {}
    for attr_name in attr_names:
        attr_values = [getattr(collection, attr_name) for collection in attrs]
        attr_class = cls.get_attrs()[attr_name]
        sequences[attr_name] = [list(attr_class).index(value) + 1 for value in attr_values]
        gini_coeffs[attr_name] = gini_coefficient(sequences[attr_name])

    plt.figure(figsize=[10, 8])
    for attr_name, seq in sequences.items():
        sorted_sequence = np.sort(seq)
        n = len(sorted_sequence)
        cumulative_sum = np.cumsum(sorted_sequence, dtype=float)
        cumulative_sum /= cumulative_sum[-1]
        x = np.linspace(0, 1, n)
        gini = gini_coefficient(seq)
        plt.plot(x, cumulative_sum, label=f"{attr_name} (Gini: {gini:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Line of Equality")
    plt.title(
        f"Lorenz Curves + Gini Coefficients for attributes | Avg. Gini: {np.mean(list(gini_coeffs.values())):.2f}"
    )
    plt.xlabel("Cumulative Share of Collections")
    plt.ylabel("Cumulative Share of Attribute")
    plt.legend()
    plt.grid(True)
    plt.savefig("data/gini_coefficients.png")


def plot_entropy(attrs: List[BaseAttributes]) -> None:
    """Plot the entropy of the attributes."""
    attr_class = attrs[0].__class__
    attr_names = attr_class.get_attrs().keys()

    ## Entropy of the list of input attrs
    entropy_for_attrs = {}
    for attr_name in attr_names:
        attr_values = [getattr(collection, attr_name) for collection in attrs]
        total_count = len(attr_values)
        probabilities = [count / total_count for count in Counter(attr_values).values()]
        entropy_value = entropy(probabilities)
        entropy_for_attrs[attr_name] = entropy_value
    total_entropy_attrs = sum(entropy_for_attrs.values())

    ## Entropy for random attrs
    samples = 256
    total_entropy_random_attrs = 0
    for _ in range(samples):
        entropy_for_random_attrs = {}
        random_attrs = list(attr_class.random_sample(len(attrs)))
        for attr_name in attr_names:
            attr_values = [getattr(collection, attr_name) for collection in random_attrs]
            total_count = len(attr_values)
            probabilities = [count / total_count for count in Counter(attr_values).values()]
            entropy_value = entropy(probabilities)
            entropy_for_random_attrs[attr_name] = entropy_value
        total_entropy_random_attrs += sum(entropy_for_random_attrs.values())
    avg_random_entropy = total_entropy_random_attrs / samples

    ## plot horizontal lines for each entropy value
    plt.figure(figsize=(10, 8))
    plt.axhline(y=total_entropy_attrs, color="r", linestyle="--", label="Optimized Game Total Entropy")
    plt.axhline(
        y=avg_random_entropy,
        color="g",
        linestyle="--",
        label=f"Random Game Avg. Entropy, {samples} samples",
    )

    # Add a vertical line to show the difference
    difference = abs(total_entropy_attrs - avg_random_entropy)
    midpoint = (total_entropy_attrs + avg_random_entropy) / 2
    plt.vlines(
        x=len(attr_names) / 2,
        ymin=min(total_entropy_attrs, avg_random_entropy),
        ymax=max(total_entropy_attrs, avg_random_entropy),
        colors="b",
        linestyles="solid",
        label=f"Difference: {difference:.2f}",
    )
    plt.text(
        len(attr_names) / 2 + 0.1, midpoint, f"{difference:.2f}", verticalalignment="center", horizontalalignment="left"
    )
    plt.legend()
    plt.xlabel("Attributes")
    plt.ylabel("Entropy")
    plt.title("Entropy Comparison: Optimized vs Random Game")
    plt.xticks(range(len(attr_names)), attr_names, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("data/entropy_comparison.png")


def plot_gini_impurity(attrs: List[BaseAttributes]) -> None:
    """Plot the Gini impurity of the attributes."""
    cls = attrs[0].__class__
    attr_names = cls.get_attrs().keys()

    def gini_impurity(sequence) -> float:
        assert len(sequence) > 0, "Sequence must not be empty"
        return 1 - sum(p**2 for p in sequence)

    ## gini impurity of the list of input attrs
    gini_impurity_for_attrs = {}
    for attr_name in attr_names:
        attr_values = [getattr(collection, attr_name) for collection in attrs]
        probabilities = [count / len(attr_values) for count in Counter(attr_values).values()]
        gini_impurity_value = gini_impurity(probabilities)
        gini_impurity_for_attrs[attr_name] = gini_impurity_value

    ## gini impurity of random attrs
    gini_impurity_for_random_attrs = {}
    samples = 256
    for attr_name in attr_names:
        gini_impurity_samples: List[float] = []
        for _ in range(samples):
            random_attrs = list(cls.random_sample(len(attrs)))
            attr_values = [getattr(collection, attr_name) for collection in random_attrs]
            probabilities = [count / len(attr_values) for count in Counter(attr_values).values()]
            gini_impurity_value = gini_impurity(probabilities)
            gini_impurity_samples.append(gini_impurity_value)
        gini_impurity_for_random_attrs[attr_name] = sum(gini_impurity_samples) / samples

    ## plot bar chart of gini impurity for attrs / random attrs
    plt.figure(figsize=(10, 8))
    width = 0.35
    x = np.arange(len(gini_impurity_for_attrs))
    plt.bar(x - width / 2, list(gini_impurity_for_attrs.values()), width, label="Optimized Game", alpha=0.7)
    plt.bar(x + width / 2, list(gini_impurity_for_random_attrs.values()), width, label="Random Game", alpha=0.7)
    plt.xlabel("Attributes")
    plt.ylabel("Gini Impurity")
    plt.title("Gini Impurity Comparison: Optimized vs Random Game")
    plt.xticks(x, list(gini_impurity_for_attrs.keys()), rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/gini_impurity_comparison.png")


MODEL_ID = "black-forest-labs/FLUX.1-dev"
DATA_PATH = Path("data/")
DATA_PATH.mkdir(parents=True, exist_ok=True)
BASE_PATH = DATA_PATH / "guesswho_run_game_14"
BASE_PATH.mkdir(parents=True, exist_ok=True)
NUM_SAMPLES = assert_fn(4, lambda n: math.sqrt(n).is_integer())
GAME_LAYOUT = (6, 6)
GAME_SIZE = mul(*GAME_LAYOUT)
SEED = 42

if __name__ == "__main__":
    logger.info("Generating images ...")

    world_size = torch.cuda.device_count()
    logger.info(f"Intitializing with world size: {world_size}")

    ## Generate a game
    game = Attributes.create_game(GAME_SIZE)
    pprint.pprint(perc_of_attributes(game))
    plot_attribute_dist(game)
    plot_pairwise_attribute_dist(game)
    plot_gini_coefficients(game)
    plot_entropy(game)
    plot_gini_impurity(game)

    ## Generate images
    # mp.spawn(generate_images, args=(world_size, game), nprocs=world_size)  # type: ignore
    mp.spawn

    # Concat all images into a single image
    all_imgs: List[Image.Image] = []
    for i in range(GAME_SIZE):
        person_base_path = BASE_PATH / f"{i}"
        num = random.randint(0, NUM_SAMPLES - 1)
        all_imgs.append(Image.open(person_base_path / f"sample_{num}.webp"))
    final_img = concat_mxn_imgs(all_imgs, *GAME_LAYOUT)
    final_img.save(BASE_PATH / "game.webp")
