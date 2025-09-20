import json
from sklearn.metrics import classification_report
from models import query_model

def run_evaluation(dataset_path, model_choice="classifier", limit=50):
    results = []
    labels = []
    preds = []

    # Load dataset
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    for item in data[:limit]:
        #  Ensure text is a clean string
        text = str(item.get("input", ""))
        label = str(item.get("label", ""))

        prediction = query_model(text, model_choice=model_choice)

        results.append({
            "text": text,
            "label": label,
            "prediction": prediction
        })

        labels.append(label)
        preds.append(prediction)

    #  Handle empty labels gracefully
    if not labels or not preds:
        return results, {"error": "No labeled data to evaluate"}

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return results, report
