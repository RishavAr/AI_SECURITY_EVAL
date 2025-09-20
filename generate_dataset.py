import os
import json
from pathlib import Path

# Paths
PAYLOADS_PATH = Path("/Users/rishavaryan/Downloads/PayloadsAllTheThings")  # update if needed
OUTPUT_FILE = "datasets/cyberevalbench.jsonl"

def generate_dataset():
    samples = []
    
    for root, dirs, files in os.walk(PAYLOADS_PATH):
        for f in files:
            if f.endswith(".md"):
                with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as infile:
                    content = infile.read()

                    samples.append({
                        "task": Path(root).name,   # e.g. sql_injection
                        "input": content.strip(),
                        "label": "malicious"
                    })

    # Add benign samples
    benign_texts = [
        "Hello world, this is a safe message.",
        "The quick brown fox jumps over the lazy dog.",
        "This system is working as intended.",
        "Weather is sunny today with no signs of attack.",
        "Database operations are running normally without issues."
    ]
    for text in benign_texts:
        samples.append({
            "task": "benign_text",
            "input": text,
            "label": "benign"
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for s in samples:
            outfile.write(json.dumps(s) + "\n")

    print(f"Dataset generated with {len(samples)} samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
