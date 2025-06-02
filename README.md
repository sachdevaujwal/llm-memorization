Overview
This project implements and reproduces key methods from six foundational studies exploring the boundary between memorization and reasoning in large language models (LLMs). The objective is to create a modular, extensible, and well-documented codebase that enables future cross-paper and cross-dataset evaluations.

Goals
Reproduce core techniques from each paper for identifying reasoning or memorization patterns

Standardize dataset formatting and model evaluation across studies

Document implementation assumptions, challenges, and deviations from reported results

Enable future comparative and extension research

Target Papers
1. Xie et al. (2024)
On Memorization of Large Language Models in Logical Reasoning
Introduces perturbed logic puzzle variants to test memorization sensitivity.

2. Wu et al. (2023)
Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks
Uses counterfactual examples to distinguish reasoning failures from memorization.

3. Jin et al. (2024)
Disentangling Memory and Reasoning Ability in Large Language Models
Explores dual-task setups and training-controlled experiments.

4. Salido et al. (2025)
None of the Others: A General Technique to Distinguish Reasoning from Memorization
Proposes distractor-logic for probing model reliability on MCQ tasks.

5. Hong et al. (2025)
The Reasoning-Memorization Interplay Is Mediated by a Single Direction
Identifies a directional vector in activation space (LiReF) that modulates reasoning.

6. Sun et al. (2024)
How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior
Investigates how different memory conditions affect LLM agent responses across multiple domains using synthetic episodic histories.

Repository Structure
bash
Copy
Edit
llm-memorization/
├── xie2024/       # Paper 1: Xie et al. (2024)
├── wu2023/        # Paper 2: Wu et al. (2023)
├── jin2024/       # Paper 3: Jin et al. (2024)
├── salido2025/    # Paper 4: Salido et al. (2025)
├── hong2025/      # Paper 5: Hong et al. (2025)
├── sun2024/       # Paper 6: Sun et al. (2024)
├── shared/        # Shared scripts, visualizations, utilities
├── utils/         # Helper utilities (e.g., memory_manager.py)
├── environment.yml
└── README.md      # This file
Getting Started
Clone this repository

bash
'''
git clone https://github.com/YOUR_USERNAME/llm-memorization.git  
cd llm-memorization
''' 

Set up the environment

bash
'''conda env create -f environment.yml  
conda activate llm-mem
'''

Reproduce a paper
Navigate into any paper folder (e.g., hong2025/, sun2024/) and follow its README.md.

Deliverables
Each paper folder includes:

* A modular implementation of the proposed method

* Reproduction results (where applicable)

* Annotated documentation of assumptions and challenges

* Process notes in step_documentation.md and reproduction_log.md


