import os
import json

raw_dir = "sun2024/data/raw"
processed_dir = "sun2024/data/processed"

os.makedirs(processed_dir, exist_ok=True)

for filename in os.listdir(raw_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(raw_dir, filename)
        with open(input_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        # Preprocess: Here we just copy, but you can insert logic if needed
        processed_data = {
            "history": data.get("history", []),
            "test_query": data.get("test_query", "")
        }

        output_filename = filename.replace(".json", "_processed.json")
        output_path = os.path.join(processed_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(processed_data, outfile, indent=2)

        print(f"Processed: {input_path} -> {output_path}")

print("Preprocessing complete.")
