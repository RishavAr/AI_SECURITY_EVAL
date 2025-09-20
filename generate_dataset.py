import os
import json
import random
from pathlib import Path

# Paths
PAYLOADS_PATH = Path("/Users/rishavaryan/Downloads/PayloadsAllTheThings")  # update if needed
OUTPUT_FILE = "datasets/cyberevalbench.jsonl"

def generate_dataset():
    samples = []

    # Collect all malicious samples
    for root, dirs, files in os.walk(PAYLOADS_PATH):
        for f in files:
            if f.endswith(".md"):
                with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as infile:
                    content = infile.read().strip()
                    if not content:
                        continue
                    samples.append({
                        "task": Path(root).name,   # e.g., sql_injection
                        "input": content,
                        "label": "malicious"
                    })

    print(f"Collected {len(samples)} malicious samples")

    # Generate benign samples (scale up to balance malicious count)
    benign_base = [
        "System is operating normally with no anomalies.",
        "Database queries executed successfully.",
        "Weather forecast: sunny with mild temperatures.",
        "The quick brown fox jumps over the lazy dog.",
        "Application is stable and functioning as expected.",
        "User logged in successfully without suspicious activity.",
        "Network connection established without any packet loss.",
        "This is a safe and harmless message for testing purposes.",
        "Everything is running smoothly on the server.",
        "No security issues detected in this process."
    ]

    malicious_count = sum(1 for s in samples if s["label"] == "malicious")
    benign_needed = malicious_count  # balance 1:1

    benign_samples = []
    for _ in range(benign_needed):
        text = random.choice(benign_base)
        benign_samples.append({
            "task": "benign_text",
            "input": text,
            "label": "benign"
        })

    print(f"Generated {len(benign_samples)} benign samples")

    # Combine datasets
    full_dataset = samples + benign_samples
    random.shuffle(full_dataset)

    # Save
    os.makedirs("datasets", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for s in full_dataset:
            outfile.write(json.dumps(s) + "\n")

    print(f"✅ Dataset generated with {len(full_dataset)} total samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
