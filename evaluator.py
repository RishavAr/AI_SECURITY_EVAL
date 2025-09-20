import json
import pandas as pd
from sklearn.metrics import classification_report
from models import query_gpt

def run_evaluation(dataset_path, model_choice, limit=50):
    results = []
    y_true, y_pred = [], []

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[:limit]:
        sample = json.loads(line)
        text = sample["input"]
        label = sample["label"]

        prediction = query_gpt(text, model_choice=model_choice)

        results.append({
            "task": sample.get("task", ""),
            "text": text[:200] + "...",  # truncate for readability
            "label": label,
            "prediction": prediction
        })

        y_true.append(label)
        y_pred.append(prediction)

    if len(set(y_true)) > 1:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    else:
        report = {"error": "Only one class present in dataset"}

    return pd.DataFrame(results), report
