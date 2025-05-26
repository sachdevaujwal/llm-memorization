import json
from pathlib import Path
from collections import defaultdict

input_path = Path("jin2024/outputs/simulated_responses.jsonl")
output_path = Path("jin2024/outputs/scores.json")

task_scores = defaultdict(lambda: {"total": 0, "correct": 0})

with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        task = item["task_type"]
        task_scores[task]["total"] += 1
        if item["correct"]:
            task_scores[task]["correct"] += 1

# Calculate accuracy per task
for task in task_scores:
    total = task_scores[task]["total"]
    correct = task_scores[task]["correct"]
    task_scores[task]["accuracy"] = round(correct / total, 3) if total > 0 else 0.0

# Save results
with output_path.open("w", encoding="utf-8") as out:
    json.dump(task_scores, out, indent=2)

print(f"Scoring complete. Results saved to {output_path}")
