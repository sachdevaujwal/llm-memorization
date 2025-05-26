import json
from pathlib import Path
import random

input_path = Path("jin2024/data/processed/all_tasks.jsonl")
output_path = Path("jin2024/outputs/simulated_responses.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
    for line in infile:
        item = json.loads(line)
        prediction = item["answer"] if random.random() > 0.5 else "Incorrect answer"
        correct = prediction.strip().lower() == item["answer"].strip().lower()

        result = {
            "question": item["question"],
            "answer": item["answer"],
            "task_type": item["task_type"],
            "prediction": prediction,
            "correct": correct
        }

        outfile.write(json.dumps(result) + "\n")

print(f"Simulated responses written to {output_path}")
