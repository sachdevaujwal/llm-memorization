import json
from pathlib import Path
from time import sleep
from random import randint

# Input/output paths
input_path = Path("wu2023/data/processed/generated_cf.jsonl")
output_path = Path("wu2023/data/processed/model_outputs.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Simulated model call — replace with your actual call (e.g., OpenAI or local model)
def fake_model_answer(question):
    return f"This is a simulated response to: '{question}'"

with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
    for line in infile:
        item = json.loads(line)

        original_q = item["original_question"]
        cf_q = item["cf_question"]

        response_original = fake_model_answer(original_q)
        response_cf = fake_model_answer(cf_q)

        item["response_original"] = response_original
        item["response_cf"] = response_cf

        outfile.write(json.dumps(item) + "\n")
        sleep(randint(1, 2))  # Simulate a small delay
