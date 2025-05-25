Wu et al. (2023): Reasoning or Reciting?

This project replicates and extends the findings from Wu et al. (2023) — "Reasoning or Reciting: Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks". The core objective is to evaluate whether LLMs genuinely reason through problems or rely on shallow pattern-matching and memorized heuristics.

1. Motivation

The authors argue that many LLMs appear to "reason" correctly because their training data contains similar problems. To expose this, they propose counterfactual tasks — problems that closely resemble familiar ones but contain subtle changes that disrupt common heuristics. By doing this, they isolate true reasoning ability from rote memorization.

2. Core Task Types

The paper designs five counterfactual tasks:

Last Letter Conjunction (LLC)Given two words, output the last letter of each and then the conjunction of those letters.

Input: "ant cat" → Output: "t and t"

Object-Attribute Counterfactual (OA-CF)Given a statement like "The apple is blue", models must correctly recognize if it is counterfactual or factual.

Input: "The banana is blue" → Should flag as counterfactual

Object-Relation Counterfactual (OR-CF)Identify if the stated object-relation is true or not.

"A hammer is eaten" → Should be counterfactual

Object-Counting (OC)Reason about how many instances of an object are described in the input.

"There are two cats on the mat and one on the chair." → Output: 3

Object Substitution (OS)Identify whether substituting one object with another changes factuality.

"An elephant is smaller than a mouse" → False

3. Goal

The goal is to assess how model performance drops when shifted from canonical to counterfactual forms, and whether prompting (e.g. CoT) helps models recover.

We will implement this benchmark, run evaluations, and document model performance.

4.  Dataset Reconstruction

The authors evaluate on synthetic examples designed to test reasoning. Since their datasets are not directly released in a centralized form, we reconstruct them as follows:

Each task type (LLC, OA-CF, OR-CF, OC, OS) has its own JSONL file.

Each example contains fields like:

"input": the prompt given to the LLM

"target": the expected output (answer)

"type": task type (e.g., "LLC", "OC")

"label": (optional) used for classification tasks

We will place the reconstructed data under:

wu2023/data/
├── llc.jsonl
├── oa_cf.jsonl
├── or_cf.jsonl
├── oc.jsonl
└── os.jsonl

If variants are created (e.g., with negations or distractors), we’ll include them as *_variant.jsonl.

5. Run LLM Evaluations

We evaluate open LLMs (e.g., Mistral, TinyLlama) using two prompting formats:

Direct Prompting (answer-only)

Chain-of-Thought Prompting (rationale + answer)

To run evaluation:

python wu2023/eval_tasks.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data wu2023/data/llc.jsonl \
  --save_path outputs/wu2023/llc_tinyllama_direct.jsonl \
  --prompt_format direct

Or for CoT prompting:

python wu2023/eval_tasks.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data wu2023/data/llc.jsonl \
  --save_path outputs/wu2023/llc_tinyllama_cot.jsonl \
  --prompt_format cot

The --prompt_format flag controls the instruction structure. CoT mode uses templates like:

Q: If all cats are animals and Felix is a cat, is Felix an animal?
A: Let us think step by step

Predictions are saved with fields: input, prediction, correct, and optional rationale.

6. Scoring Model Performance

To evaluate model accuracy, use the scoring script:

python wu2023/score_predictions.py \
  --data_path outputs/wu2023/llc_tinyllama_direct.jsonl

The script computes:

Accuracy

Majority class baseline

Performance by question type (e.g., counterfactual, factual)

Expected output:

{
  "overall_accuracy": 0.72,
  "majority_baseline": 0.61,
  "by_type": {
    "factual": 0.85,
    "counterfactual": 0.59,
    "negation": 0.71
  }
}

7. Counterfactual Consistency Check (Reproduction)

To validate whether models respond differently to counterfactuals, we performed a simplified version of Wu et al.'s consistency check. We:

Created 30 factual QA pairs in input_questions.jsonl

Generated synthetic counterfactuals using token substitution

Queried a model on both factual and counterfactual versions

Checked whether the outputs differ

Run scripts:

python wu2023/scripts/generate_counterfactuals.py
python wu2023/scripts/get_model_responses.py
python wu2023/scripts/score_consistency.py

Final result:

{
  "num_examples": 30,
  "consistency_score": 1.0
}

This shows the model always changed its answer in response to counterfactual inputs — a strong indicator that our counterfactuals are effective.

8. Outputs Summary

All results are saved under wu2023/outputs/

consistency_score.json: final consistency score

model_outputs.jsonl: model responses to original and CF inputs

generated_cf.jsonl: generated counterfactual input pairs