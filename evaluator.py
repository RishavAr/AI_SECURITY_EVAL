from models import query_model
import json

def run_evaluation(limit=100, backend="openai"):
    results = []
    with open("datasets/cyberevalbench.jsonl") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            sample = json.loads(line)
            task, text = sample["task"], sample["input"]

            prompt = f"Task: {task}\nInput: {text}\nClassify as 'malicious' or 'benign'."
            prediction = query_model(prompt, backend=backend)

            results.append({
                "task": task,
                "input": text,
                "prediction": prediction
            })
    return results
