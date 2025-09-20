import json
from models import query_model
from scoring import compute_score

def run_evaluation(dataset_path= "datasets/cyberevalbench.jsonl"):
    results = []
    with open(dataset_path, "r") as f:
        for line in f:
            case = json.loads(line)
            task, text, expected = case["task"], case["input"], case["expected"]

            # Prompt model
            prompt = f"Task: {task}\nInput: {text}\nClassify as 'malicious' or 'benign'."
            prediction = query_model(prompt)

            results.append({
                "task": task,
                "input": text,
                "expected": expected,
                "prediction": prediction
            })
    
    compute_score(results)
    return results

if __name__ == "__main__":
    run_evaluation()
