from textwrap import dedent
from typing import Generator, Iterator, TypeVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline

torch.random.manual_seed(0)
T = TypeVar("T")


def take_up_to_last(iterator: Iterator[T]) -> Generator[T, None, None]:
    try:
        last = next(iterator)
    except (StopIteration, TypeError):
        return

    for item in iterator:
        yield last
        last = item


model_id = "gradientai/Llama-3-70B-Instruct-Gradient-1048k"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True)


messages = [
    {"role": "system", "content": "Answer the users question about the book provided."},
    {
        "role": "user",
        "content": dedent(f"""
    The book is "Atlas Shrugged".

    Here is the book:

    --- BEGIN BOOK ---
    {open("./data/atlas_shrugged.txt", "r").read()[:100_000]}
    --- END BOOK ---
"""),
    },
    {
        "role": "user",
        "content": dedent("""
    Please tell me what was the family business of Francisco d'Anconia?
"""),
    },
]

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, streamer=streamer, cache=False)
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
    "streamer": streamer,
}
for o in take_up_to_last(pipe(messages, **generation_args)):
    print(o)
