import os
import pandas as pd
from pathlib import Path
import json

# Paths
input_dir = Path("jin2024/data/raw")
output_dir = Path("jin2024/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Define mappings for task classification
TASK_MAP = {
    "extraction": "memory",
    "memorization": "memory",
    "deduction": "reasoning",
    "induction": "reasoning"
}

def preprocess():
    examples = []

    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue

        task_name = filename.replace(".csv", "").lower()
        task_type = TASK_MAP.get(task_name)

        if task_type is None:
            print(f"Skipping unknown task type: {filename}")
            continue

        df = pd.read_csv(input_dir / filename)

        for _, row in df.iterrows():
            question = row.get("question")
            answer = row.get("answer")

            if pd.isna(question) or pd.isna(answer):
                continue

            examples.append({
                "question": question,
                "answer": answer,
                "task_name": task_name,
                "task_type": task_type
            })

    # Save as JSONL
    output_path = output_dir / "all_tasks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item) + "\n")

    print(f"Processed {len(examples)} examples to {output_path}")

if __name__ == "__main__":
    preprocess()
