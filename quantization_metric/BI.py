from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from alphalora.expert_number import calculate_expert
import json
from tqdm import tqdm
import torch
from lsaq_quant_assign import quantize_llama_like
import os 

from datasets import load_dataset

def compute_bi(x_in, x_out):
    x_in = x_in.view(-1, x_in.size(-1))
    x_out = x_out.view(-1, x_out.size(-1))
    cos_sim = torch.nn.functional.cosine_similarity(x_in, x_out, dim=-1)
    return (1 - cos_sim.mean()).item()


dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
# print(dataset[:8])
# model_names = ['Qwen3-8B', 'Qwen3-4B', 'Mistral-7B-Instruct-v0.3','Llama-3.2-3B-Instruct']
model_names = ['Llama-2-7b-hf']
for model_name in model_names:
    model_id = f"../models/{model_name}"
    # Llama-3.2-3B-Instruct
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    num_layers = len(model.model.layers)
    bi_sums = torch.zeros(num_layers, device=model.device)
    count = 0
    batch_size = 8
    max_len = 128

    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        
        texts = dataset[i: i + batch_size]["text"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)

        hidden_states_in, hidden_states_out = {}, {}

        def make_hook(layer_id):
            def hook(module, input, output):
                hidden_states_in[layer_id] = input[0].detach()
                hidden_states_out[layer_id] = output.detach()
            return hook

        hooks = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(model.model.layers)]

        with torch.no_grad():
            _ = model(**inputs)

        for h in hooks: h.remove()  # 移除 hook，避免重复注册

        # 累积每层 BI
        for l in range(num_layers):
            bi_sums[l] += compute_bi(hidden_states_in[l], hidden_states_out[l])
        count += 1

    # break

bi_scores = (bi_sums / count).cpu().tolist()
sorted_bi = sorted(enumerate(bi_scores), key=lambda x: x[1])
print(sorted_bi)
with open(f"BI/BI_{os.path.basename(model_id)}.json", "w") as f:
    json.dump(bi_scores, f, indent=4)
