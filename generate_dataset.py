import os
import json
import random
from pathlib import Path

# Paths
PAYLOADS_PATH = Path("/Users/rishavaryan/Downloads/PayloadsAllTheThings")  # update if needed
OUTPUT_FILE = "datasets/cyberevalbench.jsonl"

def generate_dataset():
    samples = []

    # Collect malicious samples from PayloadsAllTheThings
    for root, _, files in os.walk(PAYLOADS_PATH):
        for f in files:
            if f.endswith(".md"):
                with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as infile:
                    content = infile.read().strip()
                    if content:
                        samples.append({
                            "task": Path(root).name,
                            "input": content,
                            "label": "malicious"
                        })

    malicious_count = len(samples)
    print(f"Collected {malicious_count} malicious samples")

    # Generate benign samples (balanced 1:1)
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

    benign_samples = [{
        "task": "benign_text",
        "input": random.choice(benign_base),
        "label": "benign"
    } for _ in range(malicious_count)]

    print(f"Generated {len(benign_samples)} benign samples")

    # Combine and shuffle
    full_dataset = samples + benign_samples
    random.shuffle(full_dataset)

    os.makedirs("datasets", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for s in full_dataset:
            outfile.write(json.dumps(s) + "\n")

    print(f"✅ Dataset generated with {len(full_dataset)} total samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
