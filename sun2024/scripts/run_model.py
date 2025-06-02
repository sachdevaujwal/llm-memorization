# sun2024/scripts/run_model.py

import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Settings
model_name = "distilgpt2"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(dev)
model.eval()

# Directories
input_dir = "sun2024/data/processed"
output_dir = "sun2024/outputs/tables"
os.makedirs(output_dir, exist_ok=True)

def build_prompt(history, test_query):
    prompt = ""
    for turn in history:
        prompt += f"User: {turn['user']}\nAgent: {turn['agent']}\n"
    prompt += f"User: {test_query}\nAgent:"
    return prompt

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_k=50)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("Agent:")[-1].strip()

for file in os.listdir(input_dir):
    if not file.endswith("_processed.json"):
        continue

    with open(os.path.join(input_dir, file), "r") as f:
        data = json.load(f)

    prompt = build_prompt(data["history"], data["test_query"])
    agent_response = generate_response(prompt)

    result = {
        "domain": file.replace("_processed.json", ""),
        "user_query": data["test_query"],
        "agent_response": agent_response
    }

    output_path = os.path.join(output_dir, result["domain"] + "_response.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Generated response for {result['domain']} -> {output_path}")

print("Model execution complete.")
