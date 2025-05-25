import json
from pathlib import Path

input_path = Path("wu2023/data/processed/model_outputs.jsonl")
output_path = Path("wu2023/outputs/consistency_score.json")

consistencies = []

with input_path.open("r", encoding="utf-8") as infile:
    for line in infile:
        item = json.loads(line)
        # Placeholder logic: score is 1 if both responses are different
        consistent = int(item["response_original"] != item["response_cf"])
        consistencies.append(consistent)

average_score = sum(consistencies) / len(consistencies) if consistencies else 0.0

result = {
    "num_examples": len(consistencies),
    "consistency_score": average_score
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"Consistency score: {average_score:.3f}")
print(f"Results saved to {output_path}")
