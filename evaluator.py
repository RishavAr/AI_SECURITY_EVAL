import json
from models import query_model
from sklearn.metrics import classification_report, accuracy_score

def run_evaluation(dataset_path="datasets/cyberevalbench.jsonl", model_choice="classifier", limit=None):
    """
    Run evaluation on a dataset using the chosen model.
    Supports both Hugging Face classifiers and LLMs (Mistral/LLaMA).
    """

    results = []
    y_true = []
    y_pred = []

    # Load dataset
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    if limit:
        data = data[:limit]

    for item in data:
        text, label = item["input"], item["label"]

        # Query model
        prediction = query_model(text, model_choice=model_choice)

        # --- Normalize prediction for LLMs ---
        if model_choice in ["mistral", "llama"]:
            if isinstance(prediction, str):
                if "malicious" in prediction.lower():
                    prediction = "malicious"
                elif "benign" in prediction.lower():
                    prediction = "benign"
                else:
                    prediction = "benign"  # default fallback

        # --- Normalize prediction for classifiers ---
        elif model_choice == "classifier":
            if prediction.lower() in ["positive", "malicious"]:
                prediction = "malicious"
            else:
                prediction = "benign"

        # Collect results
        y_true.append(label)
        y_pred.append(prediction)
        results.append({"text": text, "label": label, "prediction": prediction})

    # Compute evaluation report
    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        report["accuracy"] = accuracy
    else:
        report = {"error": "No labeled data to evaluate"}

    return results, report
