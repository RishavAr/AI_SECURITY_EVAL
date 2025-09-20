import json
from sklearn.metrics import classification_report
from models import query_gpt

def run_evaluation(dataset_path, model_choice="gpt4o", limit=50):
    results = []
    gold, preds = [], []

    with open(dataset_path, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            if i >= limit:
                break
            item = json.loads(line)
            text, label = item["input"], item["label"]

            prediction = query_gpt(text, model_choice=model_choice)

            results.append({
                "text": text,
                "label": label,
                "prediction": prediction
            })
            gold.append(label)
            preds.append(prediction)

    report = classification_report(gold, preds, output_dict=True, zero_division=0)
    return results, report
