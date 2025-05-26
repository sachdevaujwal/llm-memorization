import json
from pathlib import Path
import random

# Paths
input_path = Path("jin2024/data/processed/all_tasks.jsonl")
output_path = Path("jin2024/outputs/simulated_responses.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Simulate responses
responses = []
with input_path.open("r", encoding="utf-8") as infile:
    for line in infile:
        item = json.loads(line)
        question = item["question"]
        answer = item["answer"]
        task_type = item["task_type"]

        # Simulated prediction: 80% correct if reasoning, 50% if memory (for demo purposes)
        if task_type == "reasoning":
            correct = random.random() < 0.8
        else:
            correct = random.random() < 0.5

        prediction = answer if correct else "Incorrect answer"

        responses.append({
            "question": question,
            "answer": answer,
            "task_type": task_type,
            "prediction": prediction,
            "correct": prediction == answer
        })

# Save
with output_path.open("w", encoding="utf-8") as out:
    for response in responses:
        out.write(json.dumps(response) + "\n")

print(f"Simulated responses written to {output_path}")
