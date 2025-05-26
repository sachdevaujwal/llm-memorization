#!/bin/bash
mkdir -p jin2024/data/raw

# Download valid JSONL files from the GitHub repository
curl -L -o jin2024/data/raw/strategyqa.jsonl https://raw.githubusercontent.com/kaistAI/MRBench/main/data/memory/strategyqa.jsonl
curl -L -o jin2024/data/raw/csqa.jsonl https://raw.githubusercontent.com/kaistAI/MRBench/main/data/memory/csqa.jsonl
curl -L -o jin2024/data/raw/sports_understanding.jsonl https://raw.githubusercontent.com/kaistAI/MRBench/main/data/reasoning/sports_understanding.jsonl
