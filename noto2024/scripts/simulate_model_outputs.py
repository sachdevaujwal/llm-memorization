import json
from pathlib import Path
import random

input_path = Path("noto2024/data/real/mmlu_test.jsonl")
output_path = Path("noto2024/outputs/model_outputs.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
    for line in infile:
        item = json.loads(line)
        choices = item["choices"]
        correct = item["answer"]  # MMLU uses 'answer' as the correct index


        # Simulate prediction — random int from 0 to len(choices) for NotO
        pred = random.randint(0, len(choices))

        result = {
            "question": item["question"],
            "choices": choices,
            "answer_idx": correct,
            "predicted_idx": pred,
        }
        json.dump(result, outfile)
        outfile.write("\n")

print(f"Simulated predictions saved to {output_path}")
