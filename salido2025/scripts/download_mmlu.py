from datasets import load_dataset

dataset = load_dataset("cais/mmlu", "all", split="test")
subset = dataset.select(range(30))  # Use more if needed
subset.to_json("noto2024/data/real/mmlu_test.jsonl")
print("MMLU subset saved.")
