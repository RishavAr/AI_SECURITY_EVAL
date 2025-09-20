import json
import pandas as pd
from sklearn.metrics import classification_report
from models import query_gpt

def run_evaluation(dataset_path, model_choice, limit=50):
    results = []
    y_true, y_pred = [], []

    # Load dataset
    with open(dataset_path, "r") as f:
        lines = f.readlines()

    for line in lines[:limit]:
        sample = json.loads(line)
        text = sample["text"]
        label = sample["label"]  # ground truth

        prediction = query_gpt(text, model_choice=model_choice)

        results.append({
            "text": text,
            "label": label,
            "prediction": prediction
        })

        y_true.append(label)
        y_pred.append(prediction)

    # Generate classification report
    if len(set(y_true)) > 1:  # avoid errors if only 1 class
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    else:
        report = {"error": "No labeled data to evaluate"}

    return pd.DataFrame(results), report
