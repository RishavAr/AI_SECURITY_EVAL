import json
from sklearn.metrics import classification_report, accuracy_score
from models import query_gpt

def run_evaluation(dataset_path="datasets/cyberevalbench.jsonl", model_choice="gpt-4o-mini", limit=100):
    results, y_true, y_pred = [], [], []

    with open(dataset_path, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            if i >= limit:
                break
            item = json.loads(line)
            text, label = item["input"], item["label"]

            prediction = query_gpt(text, model_choice=model_choice)

            y_true.append(label)
            y_pred.append(prediction)
            results.append({"input": text, "true": label, "pred": prediction})

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return results, {"classification_report": report, "accuracy": acc}
