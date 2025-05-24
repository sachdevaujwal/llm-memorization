# Xie et al. (2024) – On Memorization of Large Language Models in Logical Reasoning

**arXiv**: https://arxiv.org/abs/2410.23123  
**PDF Downloaded**: Yes  
**Project Page**: https://memkklogic.github.io  
**Code Repository**: https://github.com/AlphaPav/mem-kk-logic  
**Code Available**: Yes 
**Dataset Available**: Yes (Included in the repository under the `data/` directory)

---

## Method Summary

- Introduces a new logical reasoning benchmark based on Knights and Knaves (K&K) puzzles.
- Measures memorization using a **Local Inconsistency-based Memorization Score (LiMem)**:
  
  `LiMem(f ; D) = Accuracy × (1 − Consistency Ratio under perturbations)`

- Perturbation types:
  - Statement-level and leaf-level (math structure)
  - Role name swaps, uncommon names, reordering, and role flipping (language structure)
- Provides modules for:
  - Puzzle generation at varying difficulty (2–8 characters)
  - Automatic solution and Chain-of-Thought (CoT) generation
  - Perturbation synthesis

---

## Reproduction Plan

- [x] Locate and confirm access to the code and dataset
- [ ] Clone and set up the repository locally
- [ ] Generate example puzzles and perturbations
- [ ] Run evaluation and LiMem scoring scripts
- [ ] Integrate into modular repo format (`code/xie2024/`)
- [ ] Reproduce accuracy/memorization plots

---

## Notes

- Fine-tuning is done using Llama3-8B and GPT4o-mini.
- Model performance is evaluated under direct answer prediction and CoT prompting.
- Code appears actively maintained and well-commented.

---

