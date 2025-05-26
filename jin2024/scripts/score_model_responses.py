import json
from pathlib import Path
from collections import defaultdict

input_path = Path("jin2024/outputs/simulated_responses.jsonl")
output_path = Path("jin2024/outputs/scores.json")

counts = defaultdict(lambda: {"correct": 0, "total": 0})

with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        task = item["task_type"]
        counts[task]["total"] += 1
        if item["correct"]:
            counts[task]["correct"] += 1

# Compute accuracy
results = {}
for task_type, stats in counts.items():
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    results[task_type] = {
        "total": stats["total"],
        "correct": stats["correct"],
        "accuracy": round(acc, 3)
    }

with output_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Scores saved to {output_path}")
