# Paper 6: Sun et al. (2024) – How Memory Management Impacts LLM Agents

## Goal
To implement and replicate the experimental setup that tests how long-context memory and user history affect LLM agents' ability to generate context-aware responses. The study focuses on experience-following behavior across domains.

## Method
- Domains: `food`, `movie`, `schedule`, `shopping`, `travel`
- For each domain:
  - A synthetic history and test query were generated
  - The model `distilgpt2` was prompted to produce a response
- The history remained constant to simulate memory presence, and test queries probed adaptation

## Scripts
- `generate_data.py`: Creates 5 sample domain files with consistent user history and test queries
- `preprocess_data.py`: Adds structured metadata (e.g., token count, length)
- `run_model.py`: Loads `distilgpt2`, constructs prompts, and generates responses

## Output
Each file in `sun2024/outputs/tables/*_response.json` contains:
- The domain
- The user’s test query
- The model's generated response

## Observations
- All model runs completed without errors
- Responses were valid but often lacked deep context-following behavior, consistent with expectations for smaller LLMs without memory augmentation
- No additional finetuning or memory injection was performed

## Next Steps
- Extend this setup to test with memory-augmented models or agent frameworks (e.g., LangChain, LlamaIndex)
- Compare response quality against human-written responses or larger LLMs
