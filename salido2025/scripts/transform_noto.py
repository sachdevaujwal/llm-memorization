import json
from pathlib import Path

input_path = Path("noto2024/data/real/mmlu_test.jsonl")
output_path = Path("noto2024/data/sample_mcq_noto.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
    for line in infile:
        item = json.loads(line)
        question = item["question"]
        choices = item["choices"]
        correct_idx = item["answer"]

        for idx, choice in enumerate(choices):
            label = 1 if idx == correct_idx else 0
            formatted = {
                "question": question,
                "choice": choice,
                "label": label,
                "original_idx": idx
            }
            outfile.write(json.dumps(formatted) + "\n")

print(f"NotO transformation saved to {output_path}")
