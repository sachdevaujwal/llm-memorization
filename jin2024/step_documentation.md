# Jin et al. (2024): Disentangling Memory and Reasoning Ability in LLMs

This project reproduces and extends experiments from Jin et al. (2024) — *"Disentangling Memory and Reasoning Ability in Large Language Models"*. The paper investigates how well LLMs can separate true reasoning from memorized knowledge, using benchmark datasets covering both skill types.

---

## 1. Motivation

The paper argues that while LLMs often succeed on downstream tasks, it’s unclear whether this success reflects true reasoning or just memorization. Jin et al. design experiments to disentangle these two abilities by testing models on specially curated task types.

---

## 2. Dataset and Task Types

The paper uses tasks classified into two categories:

- **Memory Tasks**: factual recall (e.g., "Who is the president of the US?")
- **Reasoning Tasks**: involve logical or semantic inference beyond recall

### Categories:
- **Memory**
  - `extraction`
  - `memorization`
- **Reasoning**
  - `deduction`
  - `induction`

Each task is stored in a CSV file under `jin2024/data/raw/`.

**Note:** As of May 2025, the original MRBench dataset referenced in the paper is **no longer available on Hugging Face or GitHub**. Therefore, we used **sample CSVs** modeled on the paper’s descriptions to demonstrate preprocessing and scoring logic.

---

## 3. Preprocessing

All CSVs are converted into a unified JSONL format for evaluation. Each line contains:
- `question`
- `answer`
- `task_type` ("memory" or "reasoning")

Run preprocessing:

```bash
python jin2024/scripts/preprocess.py
```

Output: `jin2024/data/processed/all_tasks.jsonl`

---

## 4. Simulate Model Responses

We simulate predictions by randomly returning correct or incorrect answers. This is for demonstration and will later be replaced by actual model responses.

```bash
python jin2024/scripts/simulate_responses.py
```

Output: `jin2024/outputs/simulated_responses.jsonl`

---

## 5. Scoring

To score how well the model performs on memory vs. reasoning tasks:

```bash
python jin2024/scripts/score_by_tasktype.py
```

Output: `jin2024/outputs/scores.json`

Sample output:
```json
{
  "memory": {"total": 3, "correct": 1, "accuracy": 0.333}
}
```

---

## 6. Outputs

- `jin2024/data/raw/`: sample CSVs used to mimic MRBench
- `jin2024/data/processed/all_tasks.jsonl`: preprocessed input
- `jin2024/outputs/simulated_responses.jsonl`: simulated predictions
- `jin2024/outputs/scores.json`: evaluation metrics by task type

---

This concludes our pipeline replication for Jin et al. (2024). We note the real dataset's inaccessibility and rely on synthetically generated samples to demonstrate the intended workflow.


## 7. Simulation and Results

We simulated predictions on the preprocessed MRBench subset and computed accuracy per task type. All samples were synthetic or sourced from the GitHub repository since the full dataset was unavailable on HuggingFace.

Final scores:

json
Copy
Edit
{
  "memory": {
    "total": 3,
    "correct": 2,
    "accuracy": 0.667
  }
}
Note: Real data was not accessible via HuggingFace Hub, so we used public JSONL files from the MRBench GitHub repo.