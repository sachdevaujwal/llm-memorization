# hong2025/scripts/analyze_liref_directions.py

import torch
import numpy as np
from pathlib import Path

# Config
model_name = "mistral7b"
input_path = Path(f"hong2025/outputs/{model_name}-gsm8k_hs.pt")
output_path = Path(f"hong2025/outputs/{model_name}-gsm8k_liref_directions.pt")

# Load cached hidden states
print(f"Loading hidden states from: {input_path}")
hs_cache = torch.load(input_path)

# Compute linear reasoning directions (mean vector per layer)
liref_directions = {}
for layer, hidden_states in hs_cache.items():
    direction = hidden_states.mean(dim=0)
    liref_directions[layer] = direction
    print(f"Computed direction for layer {layer}, shape: {direction.shape}")

# Save
torch.save(liref_directions, output_path)
print(f"LiReF directions saved to: {output_path}")
