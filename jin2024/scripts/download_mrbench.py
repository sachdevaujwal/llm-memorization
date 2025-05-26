from datasets import load_dataset

dataset = load_dataset("kaist-ai/mrbench")

# Save a subset to disk for inspection (optional)
dataset["train"].to_json("jin2024/data/raw/mrbench_train.jsonl")
dataset["validation"].to_json("jin2024/data/raw/mrbench_validation.jsonl")
