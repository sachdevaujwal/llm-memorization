import json
import random
import os

ENTITY_PAIRS = [
    ("Barack Obama", "George W. Bush"),
    ("Paris", "Tokyo"),
    ("Einstein", "Newton"),
    ("Python", "Java"),
    ("Amazon", "eBay"),
]

def replace_entity(text, entity_a, entity_b):
    return text.replace(entity_a, entity_b)

def generate_counterfactuals(input_path, output_path):
    with open(input_path, "r") as f:
        examples = [json.loads(line) for line in f]

    results = []
    for ex in examples:
        question = ex["question"]
        answer = ex.get("answer", "")
        for e1, e2 in ENTITY_PAIRS:
            if e1 in question:
                cf_question = replace_entity(question, e1, e2)
                results.append({
                    "original_question": question,
                    "original_answer": answer,
                    "cf_question": cf_question,
                    "cf_entity": f"{e1} -> {e2}"
                })
                break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)} counterfactuals to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    generate_counterfactuals(args.input, args.output)
