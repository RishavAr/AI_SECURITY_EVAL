from sklearn.metrics import classification_report

def compute_score(results):
    y_true = [r["expected"] for r in results]
    y_pred = [r["prediction"].lower() for r in results]

    print("\n=== Evaluation Report ===")
    print(classification_report(y_true, y_pred, zero_division=0))
