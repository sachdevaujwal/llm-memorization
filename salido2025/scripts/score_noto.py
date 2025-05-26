import json
from pathlib import Path

input_path = Path("noto2024/outputs/model_outputs.jsonl")
output_path = Path("noto2024/outputs/score_results.json")

total = 0
correct = 0
noto_triggered = 0

with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        total += 1
        if item["predicted_idx"] == item["answer_idx"]:
            correct += 1
        if item["predicted_idx"] == "None of the others":
            noto_triggered += 1

results = {
    "total": total,
    "correct_predictions": correct,
    "accuracy": correct / total if total > 0 else 0.0,
    "noto_triggered_count": noto_triggered,
    "noto_triggered_rate": noto_triggered / total if total > 0 else 0.0,
}

with output_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Scoring complete. Results saved to {output_path}")
