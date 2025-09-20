import os
import json

# Path to PayloadsAllTheThings
PAYLOADS_PATH = "/Users/rishavaryan/Downloads/PayloadsAllTheThings"

# Output dataset file
OUTPUT_FILE = "datasets/cyberevalbench.jsonl"

# Vulnerability categories to extract
CATEGORIES = {
    "SQL Injection": "sql_injection",
    "XSS Injection": "xss",
    "Command Injection": "command_injection",
    "Directory Traversal": "directory_traversal",
    "Server Side Request Forgery": "ssrf"
}

def extract_payloads():
    dataset = []
    for folder, task in CATEGORIES.items():
        folder_path = os.path.join(PAYLOADS_PATH, folder)
        if not os.path.exists(folder_path):
            print(f"️ Skipping {folder}, not found")
            continue

        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.endswith(".txt") or f.endswith(".md"):
                    file_path = os.path.join(root, f)
                    with open(file_path, "r", errors="ignore") as infile:
                        for line in infile:
                            line = line.strip()
                            if not line or line.startswith("#") or line.startswith("//"):
                                continue
                            dataset.append({
                                "task": task,
                                "input": line,
                                "expected": "malicious"
                            })

    # Add some benign samples
    benign_samples = [
        {"task": "sql_injection", "input": "SELECT * FROM users WHERE id=42", "expected": "benign"},
        {"task": "xss", "input": "<p>Hello World</p>", "expected": "benign"},
        {"task": "command_injection", "input": "echo Hello", "expected": "benign"},
        {"task": "directory_traversal", "input": "/images/logo.png", "expected": "benign"},
        {"task": "ssrf", "input": "https://www.stanford.edu", "expected": "benign"}
    ]
    dataset.extend(benign_samples)

    os.makedirs("datasets", exist_ok=True)
    with open(OUTPUT_FILE, "w") as out:
        for entry in dataset:
            out.write(json.dumps(entry) + "\n")

    print(f" Dataset generated with {len(dataset)} samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_payloads()


