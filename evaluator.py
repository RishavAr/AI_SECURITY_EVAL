import json
from sklearn.metrics import classification_report
from models import query_model

def run_evaluation(dataset_path="datasets/cyberevalbench.jsonl", model_choice="gpt-4o-mini", limit=None):
    results = []
    y_true, y_pred = [], []

    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            text, label = item["input"], item["label"]

            prediction = query_model(text, model_choice=model_choice)

            results.append({
                "task": item.get("task", "unknown"),
                "input": text,
                "label": label,
                "prediction": prediction,
            })

            y_true.append(label)
            y_pred.append(prediction)

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return results, report
