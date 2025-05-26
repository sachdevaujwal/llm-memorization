from code.xie2024.scoring import compute_limem_model_score

compute_limem_model_score(
    data_file="outputs/test_run/TinyLlama/TinyLlama-1.1B-Chat-v1.0_0shot/test_config_token1024_cot_test/people4_num100.jsonl",
    base_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_path="",
    output_path="outputs/test_run/limem_model"
)
