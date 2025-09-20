import json
from models import query_model

def run_evaluation(limit=100, backend="openai"):
    with open("datasets/cyberevalbench.jsonl") as f:
        dataset = [json.loads(line) for line in f]

    dataset = dataset[:limit]  # limit to first 100

    results = []
    for item in dataset:
        task = item.get("task", "")
        text = item.get("input", "")

        prompt = f"Task: {task}\nInput: {text}\nClassify as 'benign' or 'malicious'."
        prediction = query_model(prompt, backend=backend)

        results.append({
            "task": task,
            "input": text,
            "prediction": prediction
        })

    return results
