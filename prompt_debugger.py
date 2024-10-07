from pathlib import Path
from typing import List, Tuple

import torch
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel

asdf: LlamaModel

llama_weights_path = Path("data/llama/")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(llama_weights_path)
model = AutoModelForCausalLM.from_pretrained(
    llama_weights_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
).to(device)

# Set the model to evaluation mode
model.eval()

SPECIAL_TOKENS = (
    [str(t.content) for t in tokenizer.added_tokens_decoder.values()]
    + [str(t) for t in tokenizer.special_tokens_map.values()]
    + [str(t) for t in tokenizer.additional_special_tokens]
    + [str(t) for t in tokenizer.all_special_tokens]
)

messages: List[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": "You are a friendly chatbot.",
    },
    {
        "role": "user",
        "content": "Can you tell me a random inspirational quote?",
    },
]


class Token(BaseModel):
    content: str
    token_id: int


class AttentionOutputs(BaseModel):
    input_tokens: List[Token]
    output_tokens: List[Token]
    attentions: Tuple[torch.Tensor, ...]

    class Config:  # noqa: D106
        arbitrary_types_allowed = True


def get_attention_outputs(messages: List[ChatCompletionMessageParam]) -> AttentionOutputs:
    """From a chat completion prompt, get attention outputs.

    The purpose of this function is to get the attention outputs from a causal language model.
    This way we can 'see' into the model to understand which tokens it attends to so we
    can visualize the models understanding of the prompt.
    """
    ## Do model inference from messages
    inputs_tokens_tensor = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    output_tokens_tensor = model.generate(inputs_tokens_tensor, max_new_tokens=32)
    output_tokens = output_tokens_tensor.flatten().cpu().tolist()
    tokenized_prompt = tokenizer.batch_decode(output_tokens)

    ## Do another forward pass to get attentions intermediate outputs
    with torch.no_grad():
        model_all_outputs = model(output_tokens_tensor, output_attentions=True)
    attentions = model_all_outputs.attentions

    ## Determine input and output tokens
    all_tokens = [
        Token(content=content, token_id=token_id) for token_id, content in zip(output_tokens, tokenized_prompt)
    ]
    num_input_tokens = inputs_tokens_tensor.shape[1]
    input_tokens = all_tokens[:num_input_tokens]
    output_tokens = all_tokens[num_input_tokens:]

    return AttentionOutputs(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        attentions=attentions,
    )


def filter_tokens(attention_outputs: AttentionOutputs, tokens: List[str]) -> AttentionOutputs:
    """Filter out tokens from the attention outputs.

    This function filters out tokens from the attention outputs. This is useful for
    removing special tokens from the attention outputs.
    """
    ## Filter 'token' from the input and output
    new_input_tokens = list(filter(lambda t: t.content not in tokens, attention_outputs.input_tokens))
    new_output_tokens = list(filter(lambda t: t.content not in tokens, attention_outputs.output_tokens))

    ## Get the indices of the tokens to keep (via tensor slicing)
    token_idxs = [
        i for i, t in enumerate(attention_outputs.input_tokens + attention_outputs.output_tokens) if t.content in tokens
    ]
    total_num_tokens = len(attention_outputs.input_tokens) + len(attention_outputs.output_tokens)

    new_attentions: List[torch.Tensor] = []
    for attn in attention_outputs.attentions:
        # attn shape (batch, num_heads, sequence_length, sequence_length)
        # slice the tokens to explicity keep
        keep_idxs = [i for i in range(total_num_tokens) if i not in token_idxs]
        new_attn = attn[:, :, keep_idxs, :][:, :, :, keep_idxs]
        print(attn.shape, new_attn.shape)
        new_attentions.append(new_attn)
    new_attentions_tuple = tuple(new_attentions)

    return AttentionOutputs(
        input_tokens=new_input_tokens,
        output_tokens=new_output_tokens,
        attentions=new_attentions_tuple,
    )


class HeatMapParams(BaseModel):
    input_tokens: List[str]
    output_tokens: List[str]
    attention_values: torch.Tensor

    class Config:  # noqa: D106
        arbitrary_types_allowed = True


def create_attention_heatmap(attention_outputs: AttentionOutputs) -> HeatMapParams:
    """Create an attention heatmap from the attention outputs."""
    attentions = attention_outputs.attentions
    attn_outputs_all = torch.cat(attentions, dim=0)
    attn_outputs_avg = attn_outputs_all.mean(dim=0).mean(dim=0)
    attn_outputs_avg_norm = attn_outputs_avg / attn_outputs_avg.max()
    final_attn_output = attn_outputs_avg_norm.cpu().to(torch.float32)

    return HeatMapParams(
        input_tokens=[t.content for t in attention_outputs.input_tokens],
        output_tokens=[t.content for t in attention_outputs.output_tokens],
        attention_values=final_attn_output,
    )


attn_outputs = get_attention_outputs(messages)
attn_outputs_filtered = filter_tokens(attn_outputs, SPECIAL_TOKENS)
heatmap_params = create_attention_heatmap(attn_outputs_filtered)


# plot heatmap
import matplotlib.pyplot as plt

num_input_tokens = len(heatmap_params.input_tokens)
total_num_tokens = len(heatmap_params.input_tokens) + len(heatmap_params.output_tokens)
data = heatmap_params.attention_values[:, :]
print(f"Num input tokens: {num_input_tokens}")
print(f"Num output tokens: {len(heatmap_params.output_tokens)}")
print(f"Attn shape: {data.shape}")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))  # Adjust figure size as needed
im = ax.imshow(data.numpy(), cmap="hot_r", interpolation="nearest")

# Set tick locations
ax.set_xticks(list(range(total_num_tokens)))
ax.set_yticks(list(range(total_num_tokens)))

# Set tick labels
ax.set_xticklabels(heatmap_params.input_tokens + heatmap_params.output_tokens, rotation=90, ha="right")
ax.set_yticklabels(heatmap_params.input_tokens + heatmap_params.output_tokens)

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Add colorbar
plt.colorbar(im)

# Save the figure
plt.savefig("data/attention_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
