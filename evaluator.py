import json
from sklearn.metrics import classification_report
from models import query_model

def run_evaluation(dataset_path, model_choice="openai", limit=50):
    """
    Run evaluation on a JSONL dataset.
    """
    results = []
    y_true, y_pred = [], []

    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            text, label = item["input"], item["label"]

            prediction = query_model(text, model_choice=model_choice)
            y_true.append(label.lower())
            y_pred.append(prediction.lower())

            results.append({
                "text": text,
                "label": label,
                "prediction": prediction
            })

    if not y_true:
        return results, {"error": "No labeled data to evaluate"}

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    return results, report
