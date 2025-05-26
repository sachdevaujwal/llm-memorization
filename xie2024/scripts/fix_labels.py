import os
import json

INPUT_DIR = "outputs/test_run/TinyLlama/TinyLlama-1.1B-Chat-v1.0_0shot/test_config_token1024_cot_test"

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(INPUT_DIR, filename)
        updated_lines = []

        with open(file_path, "r") as infile:
            for line in infile:
                data = json.loads(line)
                if "correct" in data:
                    data["robust_metric"] = int(data["correct"])
                updated_lines.append(json.dumps(data))

        with open(file_path, "w") as outfile:
            for line in updated_lines:
                outfile.write(line + "\n")

print("All files updated with 'robust_metric'")
