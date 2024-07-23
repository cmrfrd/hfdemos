import warnings
from typing import Generator, Iterator, TypeVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer, pipeline

warnings.filterwarnings("ignore")

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


model_id = "meta-llama/Meta-Llama-3.1-405B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
streamer = TextStreamer(tokenizer, skip_prompt=True)


messages = [
    {"role": "system", "content": "Hey, have you heard of GPTuesday?"},
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
