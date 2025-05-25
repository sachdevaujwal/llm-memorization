import os
import sys
from mem_cls_puzzle import main as puzzle_main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="outputs/test_run/TinyLlama/TinyLlama-1.1B-Chat-v1.0_0shot/test_config_token1024_cot_test/people4_num100.jsonl")
    parser.add_argument("--method", type=str, default="charlength")  # Options: tfidf, bow, wordlength, charlength, combine
    parser.add_argument("--text_field", type=str, default="quiz")  # Options include quiz, solution_text, cot_repeat_steps, statements
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--output_dir", type=str, default="outputs/test_run/limem_puzzle/")
    parser.add_argument("--no_balance_label", action="store_true")  # Add this flag if you want unbalanced labels
    args = parser.parse_args()

    # Compose argument list for mem_cls_puzzle.py
    sys.argv = [
        "mem_cls_puzzle.py",
        "--input_file", args.input_file,
        "--method", args.method,
        "--text_field", args.text_field,
        "--train_split", str(args.train_split),
        "--output_dir", args.output_dir
    ]
    if args.no_balance_label:
        sys.argv.append("--no_balance_label")

    os.makedirs(args.output_dir, exist_ok=True)
    puzzle_main()
