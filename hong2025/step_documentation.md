### `step_documentation.md`

---

## Paper 5: *The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction* (Hong et al., 2025)

### Goal
To reproduce and understand the Linear Reasoning Feature (LiReF) methodology for probing reasoning and memorization in large language models (LLMs). This includes extracting hidden state representations, computing per-layer directional vectors, and analyzing norm-based indicators of reasoning capacity.

### Methodology Overview
- The paper introduces **LiReF**: a method that identifies a specific direction in the model's hidden states that reflects its reasoning behavior.
- Uses datasets like **GSM8K**, **MBPP**, and **MMLU-Pro** for reasoning-heavy tasks.
- Per-layer direction vectors are computed using linear projection of hidden states at the last token.
- Vector norms are then analyzed to determine which layers encode stronger reasoning signals.

---

### Step 1: Setup & Folder Initialization
- Created standardized structure:
  ```bash
  mkdir -p hong2025/{data,outputs,models,scripts,logs,dataset}
  ```
- Downloaded and extracted [Linear Reasoning Features GitHub repo](https://github.com/yihuaihong/Linear_Reasoning_Features) into:
  ```
  hong2025/code/
  ```

---

### Step 2: Dataset Unpacking & Placement
- Extracted `dataset.zip` and moved to:
  ```
  hong2025/data/raw/dataset/
  ```
- Verified key datasets: `gsm8k`, `mbpp`, `mmlu-pro`, `mgsm`, `ceval-exam`, and `HumanEval`

---

### Step 3: Model Setup & Hidden State Extraction
- Initially attempted to use `Meta-Llama-3-8B` (gated access failed)
- **Resolution**: Replaced with `falcon-7b-instruct` (open-access)
- Script used:
  ```bash
  python hong2025/scripts/run_liref_feature_extraction.py
  ```
- Output:
  ```
  hong2025/outputs/mistral7b-gsm8k_hs.pt (≈56 MB)
  ```
- Sampled 100 GSM8K test examples
- Used CPU (no GPU available)
- Run took several hours, executed overnight

---

### Step 4: Compute LiReF Directions
- Script: `hong2025/scripts/analyze_liref_directions.py`
- Computed per-layer direction vectors from hidden states
- Each direction vector shape: `torch.Size([4544])`
- Output saved as:
  ```
  hong2025/outputs/mistral7b-gsm8k_liref_directions.pt
  ```

---

### Step 5: Visualization
- Script: `hong2025/scripts/visualize_liref_directions.py`
- Plotted L2 norm of LiReF vectors across all layers
- Output image:
  ```
  hong2025/outputs/liref_direction_norms.png
  ```
- Noted high memory usage when plotting, resulting in Jupyter kernel crash. Resolved by using terminal script instead.

---

### Key Notes
- `Meta-Llama-3-8B` required gated access; switched to `falcon-7b-instruct`
- No GPU was used; all runs were completed on CPU
- Long-running feature extraction was completed overnight
- Plots and intermediate tensors saved for reproducibility
- Used task tracker and consistent file structure throughout

---
