import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from peft import PeftModel
from datasets import load_dataset
import numpy as np

def compute_limem_model_score(data_file, base_model_path, adapter_path="", output_path="outputs/limem_model"):
    os.makedirs(output_path, exist_ok=True)

    # Load data
    kk_dataset = load_dataset("json", data_files={"test": [data_file]})
    statements = kk_dataset["test"]["quiz"]
    labels = kk_dataset["test"]["correct"]

    # Load model and tokenizer on CPU
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True).to("cpu")

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()

    # Setup layer-wise activation capture
    mlp_activations = {i: [] for i in range(len(model.model.layers))}
    def get_mlp_activation_hook(layer_idx):
        def hook(module, input, output):
            mlp_activations[layer_idx].append(output.detach().cpu().numpy())
        return hook
    for i, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_hook(get_mlp_activation_hook(i))

    dataset = {i: [] for i in range(len(model.model.layers))}
    labelset = {i: [] for i in range(len(model.model.layers))}

    # Create prompts and run forward
    for text, label in zip(statements, labels):
        prompt = f"### Question: {text}\n### Answer:\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
        for i in range(len(model.model.layers)):
            mlp_activations[i] = []
        with torch.no_grad():
            _ = model(input_ids)
        for i in range(len(model.model.layers)):
            if mlp_activations[i]:
                dataset[i].append(mlp_activations[i][0])
                labelset[i].append(label)

    results = {}
    for i in range(len(model.model.layers)):
        X = [x.sum(axis=(0,1)) for x in dataset[i]]
        y = labelset[i]
        if len(set(y)) < 2:
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=10000)
        clf.fit(X_train, y_train)
        test_probs = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, test_probs[:, 1])
        acc = accuracy_score(y_test, clf.predict(X_test))
        results[f"layer_{i}"] = {"auc": auc, "accuracy": acc}

    # Save
    out_file = os.path.join(output_path, "limem_model_score.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved model-based scores to {out_file}")
    return results
