NotO (2024): None of the Others

This project reproduces the experiments from NotO (2024) — "None of the Others: A General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks". The paper introduces a counterfactual technique to test whether an LLM uses genuine reasoning or memorized patterns when answering multiple-choice questions.

1. Motivation

Standard multiple-choice evaluations can mask shallow memorization. NotO introduces a novel way to reveal it by inserting an additional distractor option labeled “None of the others is correct.” If a model picks this new option for a question it originally answered correctly, it likely did not truly reason through the answer.

2. Folder Structure

All experiments and scripts are located under the noto2024/ directory:

noto2024/
├── data/
│   ├── real/                  # Real dataset (MMLU used)
│   │   └── mmlu_test.jsonl
│   └── sample_mcq.jsonl       # (optional) original dummy questions
├── dataset/
├── scripts/
│   ├── transform_noto.py      # Adds NotO choice to dataset
│   ├── simulate_model_outputs.py # Random or fixed predictions
│   └── score_noto.py          # Scoring script
├── outputs/
│   ├── model_outputs.jsonl    # Simulated predictions
│   └── score_results.json     # Final results
├── logs/

3. Step-by-Step Execution

3.1 Prepare Dataset

We used 30 samples from the publicly available MMLU dev set for testing:

wc -l noto2024/data/real/mmlu_test.jsonl
head -n 2 noto2024/data/real/mmlu_test.jsonl

3.2 Transform to NotO Format

Adds a new distractor choice to simulate the NotO intervention:

python noto2024/scripts/transform_noto.py

Output saved to:

noto2024/data/sample_mcq_noto.jsonl

3.3 Simulate Model Predictions

For prototyping, we randomly select an answer (can be replaced with actual LLM outputs):

python noto2024/scripts/simulate_model_outputs.py

Output saved to:

noto2024/outputs/model_outputs.jsonl

3.4 Score with NotO Metric

Evaluates both accuracy and NotO-triggered rate:

python noto2024/scripts/score_noto.py

Results saved to:

noto2024/outputs/score_results.json

4. Final Results

{
  "total": 30,
  "correct_predictions": 11,
  "accuracy": 0.3667,
  "noto_triggered_count": 0,
  "noto_triggered_rate": 0.0
}

These results are based on a simulation over MMLU-style real questions. In production, this workflow can be swapped with actual LLM responses to evaluate memorization vs reasoning at scale.