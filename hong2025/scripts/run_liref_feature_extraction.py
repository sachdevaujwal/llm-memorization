# hong2025/scripts/run_liref_feature_extraction.py

import torch
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from pathlib import Path

torch.set_grad_enabled(False)

# Set seeds for reproducibility
random.seed(8888)
torch.manual_seed(8888)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(8888)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model selection
#model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Publicly accessible
model_name = "tiiuae/falcon-7b-instruct"
dataset_path = Path("hong2025/data/raw/dataset")
output_path = Path("hong2025/outputs")
output_path.mkdir(parents=True, exist_ok=True)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if model.config.model_type.lower() in ['llama', 'mistral', 'yi']:
    tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    tokenizer.pad_token = '<|endoftext|>'
tokenizer.padding_side = "left"
model.to(device)

def generate_hidden_states(texts, batch_size=4):
    all_hidden_states = {}
    layers_to_cache = list(range(model.config.num_hidden_layers))

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding="longest").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        for layer in layers_to_cache:
            layer_hs = outputs.hidden_states[layer][:, -1, :].cpu()
            if layer not in all_hidden_states:
                all_hidden_states[layer] = layer_hs
            else:
                all_hidden_states[layer] = torch.cat((all_hidden_states[layer], layer_hs), dim=0)
        torch.cuda.empty_cache()
    return all_hidden_states

# Load GSM8K subset and extract
gsm8k_path = dataset_path / "gsm8k" / "main"
if gsm8k_path.exists():
    gsm8k = load_from_disk(str(gsm8k_path))['test']
    gsm8k_qs = ["Q: " + ex['question'] + "\nA: " for ex in list(gsm8k)[:100]]
    print("Running on GSM8K (main/test)...")
    hs_gsm8k = generate_hidden_states(gsm8k_qs)
    torch.save(hs_gsm8k, output_path / f"mistral7b-gsm8k_hs.pt")
    print("Saved GSM8K hidden states.")
else:
    print("GSM8K not found at expected path.")
