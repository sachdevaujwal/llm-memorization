import torch

# Adjusted paths
pt_2_path = "./hong2025/outputs/mistral7b-gsm8k_hs.pt"
pt_3_path = "./hong2025/outputs/mistral7b-gsm8k_liref_directions.pt"

# Load and inspect hong2025_2.pt
data_2 = torch.load(pt_2_path, map_location=torch.device('cpu'))
print(f"{pt_2_path} loaded. Type: {type(data_2)}")
if isinstance(data_2, dict):
    print("Keys:", list(data_2.keys()))
elif isinstance(data_2, torch.Tensor):
    print("Tensor shape:", data_2.shape)

# Load and inspect hong2025_3.pt
data_3 = torch.load(pt_3_path, map_location=torch.device('cpu'))
print(f"\n{pt_3_path} loaded. Type: {type(data_3)}")
if isinstance(data_3, dict):
    print("Keys:", list(data_3.keys()))
elif isinstance(data_3, torch.Tensor):
    print("Tensor shape:", data_3.shape)
