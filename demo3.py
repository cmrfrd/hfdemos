from textwrap import dedent
from typing import Generator, Iterator, TypeVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline

T = TypeVar("T")


def take_up_to_last(iterator: Iterator[T]) -> Generator[T, None, None]:
    try:
        last = next(iterator)
    except (StopIteration, TypeError):
        return

    for item in iterator:
        yield last
        last = item


torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True)


messages = [
    {"role": "system", "content": "Answer the users question about the book provided."},
    {
        "role": "user",
        "content": dedent("""
    The book is "A Casino Oddysey in Cyberspace".
    """),
    },
    {
        "role": "user",
        "content": dedent("""
    Can you tell me the main lesson from the book? Don't output anything else, just the main lesson.
    """),
    },
]
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, streamer=streamer)
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
    "streamer": streamer,
}
for o in take_up_to_last(pipe(messages, **generation_args)):
    print(o)


messages = [
    {"role": "system", "content": "Answer the users question about the book provided."},
    {
        "role": "user",
        "content": dedent(f"""
    The book is "A Casino Oddysey in Cyberspace".

    Here is the book:

    --- BEGIN BOOK ---
    {open("./data/a_casino_oddyssey_in_cyberspace.txt", "r").read()}
    --- END BOOK ---
"""),
    },
    {
        "role": "user",
        "content": dedent("""
    Can you tell me the main lesson from the book? Don't output anything else, just the main lesson.
"""),
    },
]

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, streamer=streamer)
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
    "streamer": streamer,
}
for o in take_up_to_last(pipe(messages, **generation_args)):
    print(o)
