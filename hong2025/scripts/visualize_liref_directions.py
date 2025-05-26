# hong2025/scripts/visualize_liref_directions.py

import torch
import matplotlib.pyplot as plt

# Load the LiReF directions
directions_path = "hong2025/outputs/mistral7b-gsm8k_liref_directions.pt"
print(f"Loading LiReF directions from: {directions_path}")
directions = torch.load(directions_path)

# Compute L2 norm of direction vector for each layer
norms = [tensor.norm().item() for _, tensor in sorted(directions.items())]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(norms, marker='o')
plt.title("LiReF Direction Norms per Layer")
plt.xlabel("Layer Index")
plt.ylabel("L2 Norm")
plt.grid(True)

# Save plot
output_path = "hong2025/outputs/liref_direction_norms.png"
plt.savefig(output_path)
print(f"Plot saved to: {output_path}")
plt.close()
