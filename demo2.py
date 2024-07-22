from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat", device_map="cuda:1", trust_remote_code=True, bf16=True
).eval()
query = tokenizer.from_list_format(
    [
        {"image": "./imgs/lean_docs.png"},
        {"text": "Can you please OCR this image?"},
    ]
)
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
