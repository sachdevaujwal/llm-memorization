import os
import json

# Domains from the paper setup
domains = ['food', 'movie', 'travel', 'schedule', 'shopping']

# Data template
template = {
    "history": [
        {"user": "I prefer vegetarian meals.", "agent": "Got it, noted."},
        {"user": "Last time you added chicken, please avoid it.", "agent": "Apologies, I'll avoid that."}
    ],
    "test_query": "I'd like to order dinner again. What do you recommend?"
}

# Ensure output directory exists
raw_dir = "sun2024/data/raw"
os.makedirs(raw_dir, exist_ok=True)

# Generate one file per domain
for domain in domains:
    domain_path = os.path.join(raw_dir, f"{domain}.json")
    with open(domain_path, 'w') as f:
        json.dump(template, f, indent=2)
print(f"Wrote data for domain: {domain} -> {domain_path}")


print("Data generation complete.")
