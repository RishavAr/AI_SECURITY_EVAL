import os
import json
from pathlib import Path
from tqdm import tqdm

# Paths
PAYLOADS_PATH = Path("/Users/rishavaryan/Downloads/PayloadsAllTheThings")  # update path if needed
OUTPUT_FILE = "datasets/cyberevalbench.jsonl"

def generate_dataset():
    samples = []
    malicious_count = 0

    # Step 1: Collect malicious payloads from ALL file types
    for root, _, files in os.walk(PAYLOADS_PATH):
        for f in files:
            if f.startswith("."):  # skip hidden files like .gitignore
                continue
            if f.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):  # skip images
                continue

            try:
                with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as infile:
                    content = infile.read().strip()
                    if content:
                        samples.append({
                            "task": Path(root).name,   # folder name = vulnerability category
                            "input": content,
                            "label": "malicious"
                        })
                        malicious_count += 1
            except Exception as e:
                print(f"Skipping {f}: {e}")

    print(f"Collected {malicious_count} malicious samples")

    # Step 2: Add benign samples (balanced with malicious count)
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

    benign_samples = []
    for i in tqdm(range(malicious_count), desc="Generating benign"):
        benign_samples.append({
            "task": "benign_text",
            "input": benign_base[i % len(benign_base)],
            "label": "benign"
        })

    samples.extend(benign_samples)
    print(f"Generated {len(benign_samples)} benign samples")

    # Step 3: Save dataset
    os.makedirs(Path(OUTPUT_FILE).parent, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for s in samples:
            outfile.write(json.dumps(s) + "\n")

    print(f" Dataset generated with {len(samples)} total samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
