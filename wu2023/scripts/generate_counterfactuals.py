import json
from pathlib import Path

input_path = Path("wu2023/data/processed/input_questions.jsonl")
output_path = Path("wu2023/data/processed/generated_cf.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

def generate_cf(question, answer):
    words = question.split()
    if len(words) > 2:
        replaced = words[2]
        cf_question = question.replace(replaced, "XYZ", 1)
        cf_entity = f"{replaced} -> XYZ"
    else:
        cf_question = question + " (counterfactual?)"
        cf_entity = "N/A -> XYZ"
    
    return {
        "original_question": question,
        "original_answer": answer,
        "cf_question": cf_question,
        "cf_entity": cf_entity
    }

with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
    for line in infile:
        item = json.loads(line)
        original_q = item["question"]
        original_a = item["answer"]

        cf_item = generate_cf(original_q, original_a)
        outfile.write(json.dumps(cf_item) + "\n")

print(f"Counterfactuals written to {output_path}")
