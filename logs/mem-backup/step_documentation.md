# LLM Memorization and Reasoning on Knights & Knaves Tasks

This project investigates how Large Language Models (LLMs) solve logical reasoning problems, particularly in the context of **Knights & Knaves** puzzles. It tests whether LLMs rely more on **memorization** or **reasoning**, using specially designed datasets and analysis techniques.

---

## 1. Dataset Construction

We use two main types of datasets:

- **Original (Clean) KK Puzzles**  
  Logical puzzles involving 2–8 characters, each of whom is either a knight (who tells the truth) or a knave (who always lies).

- **Perturbed Variants**  
  Several perturbations are applied to test model generalization:
  - `perturbed_leaf`: changes character names
  - `perturbed_statement`: modifies statements without altering logical structure
  - `random_pair`: includes unrelated distractors
  - `flip_role`: switches character labels
  - `uncommon_name`: introduces rare/unseen names
  - `reorder_statement`: shuffles statement order

Datasets are split by character count: `people2_num100.jsonl`, ..., `people8_num100.jsonl`.

---

## 2. Evaluate LLMs on KK Tasks

We evaluate LLMs on the above puzzles with both **0-shot** and **CoT (Chain-of-Thought)** prompting.  

To run evaluations:

```bash
python eval_kk.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --config test_config \
  --split test \
  --data_dir data/test/clean \
  --save_dir outputs/test_run \
  --limit 10 \
  --ntrain 0 \
  --max_token 1024 \
  --cot


The predictions and CoT responses are saved in outputs/test_run/{model_name}/....

## 3. Analyzing LLM Reasoning vs Memorization

To probe model behavior, we analyze its internal activations and output predictions using two key approaches:

### A. Model-Based Score (LIMEM Model Score)
This score evaluates the linear separability of MLP activations at each transformer layer. It helps us understand whether a model memorizes the task structure.

Run this with:
```bash
python run_limem_model.py
```

Output: outputs/limem_model/limem_model_score.json

### B. Puzzle-Based Score (LIMEM Puzzle Score)

This uses logistic regression over puzzle text features (e.g. character count, TF-IDF) to evaluate predictability of correct vs incorrect model responses.


Run this with:
```bash
python run_limem_puzzle.py --no_balance_label
```

Output saved to: result/results_<num_people>_unbalanced.jsonl

## 4. Code Strcuture

.
├── dataset/              # Dataset generation and prompt formatting
│   └── prompt.py
├── data/                 # Clean and perturbed puzzles
├── outputs/              # Evaluation results and scores
│   └── limem_model/
├── code/
│   └── xie2024/
│       └── scoring.py    # LIMEM model scoring logic
├── mem_cls_model.py      # Activation-based analysis
├── mem_cls_puzzle.py     # Text-based memorization classifier
├── run_limem_model.py    # Wrapper for model score
├── run_limem_puzzle.py   # Wrapper for puzzle score
├── eval_kk.py            # LLM evaluation logic
└── step_documentation.md # 📄 This documentation

## 5. Summary of Key Outputs

After running all experiments, you should have the following:

### ✅ LLM Evaluation Outputs
- Generated model responses:  
  `outputs/test_run/<model_name>/<config>/people<num>_num100.jsonl`

These include:
- `predicts`: model-generated reasoning
- `correct`: boolean for whether prediction matched ground truth
- `response`: full prompt and answer trace

### LIMEM Model Score
- Layer-wise linear separability of model activations:  
  `outputs/limem_model/limem_model_score.json`

Format:
```json
{
  "layer_0": {"auc": 0.72, "accuracy": 0.68},
  ...
}
```

### LIMEM Puzzle Score

Text-feature-based classification results:
result/results_<num_people>_unbalanced.jsonl

{
  "method": "charlength",
  "text_field": "quiz",
  "train_accuracy": ...,
  "test_accuracy": ...,
  "note": "Only one class present..." (if skipped)
}


## 6. Interpretation Guidance

High model-based scores suggest that the model internally separates solvable vs unsolvable puzzles based on structure.

High puzzle-based scores suggest LLM responses may be predictable from superficial features, indicating memorization.

Comparing clean vs perturbed versions helps isolate reasoning from rote recall.

## 7. Final Notes

All scripts are configured to run out-of-the-box with TinyLlama. You can switch to other HuggingFace models by changing the --model flag.

Ensure you install dependencies with:

pip install -r requirements.txt
