import os
import torch
import csv

def summarize_outputs(repo_root=".", output_file="summary_outputs.csv"):
    summary = []
    for root, _, files in os.walk(repo_root):
        for file in files:
            if file.endswith(".pt"):
                pt_path = os.path.join(root, file)
                try:
                    data = torch.load(pt_path, map_location="cpu")
                    if isinstance(data, dict):
                        first_val = next(iter(data.values()))
                        if isinstance(first_val, torch.Tensor):
                            num_layers = len(data)
                            shape = first_val.shape
                            summary.append([pt_path, num_layers, str(shape)])
                            print(f"[✓] Parsed: {pt_path}")
                        else:
                            print(f"[!] Dict in {pt_path} but not tensor values.")
                    else:
                        print(f"[!] Skipping non-dict file: {pt_path}")
                except Exception as e:
                    print(f"[✗] Failed to load {pt_path}: {e}")

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "num_layers", "shape"])
        writer.writerows(summary)

    print(f"\n Summary written to {output_file} with {len(summary)} entries.")

if __name__ == "__main__":
    summarize_outputs()
