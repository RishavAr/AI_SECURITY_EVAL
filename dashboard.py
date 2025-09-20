import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from evaluator import run_evaluation

st.set_page_config(page_title="CyberEvalBench", layout="wide")

st.title(" LLM – AI Security Evaluator")

# Sidebar controls
st.sidebar.header("Settings")
sample_limit = st.sidebar.number_input("Number of samples", min_value=10, max_value=31694, value=100, step=10)
dataset_path = "datasets/cyberevalbench.jsonl"

# Models to run
model_choices = ["classifier", "mistral", "llama"]
selected_models = st.sidebar.multiselect(
    "Choose models to evaluate",
    model_choices,
    default=model_choices
)

if st.sidebar.button("Run Evaluation"):
    all_reports = {}
    all_results = {}

    for model in selected_models:
        st.subheader(f"Running **{model}** on {sample_limit} samples…")
        results, report = run_evaluation(
            dataset_path=dataset_path,
            model_choice=model,
            limit=sample_limit
        )

        # Save outputs
        all_results[model] = results
        all_reports[model] = report

        # Show classification report
        if "error" not in report:
            st.json(report)
        else:
            st.error(report["error"])

    # Compare Accuracy Across Models
    accuracies = {m: r.get("accuracy", 0) for m, r in all_reports.items() if "error" not in r}
    if accuracies:
        st.subheader(" Accuracy Comparison")
        acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])

        st.dataframe(acc_df)

        fig, ax = plt.subplots()
        acc_df.plot(kind="bar", x="Model", y="Accuracy", ax=ax, legend=False, color="skyblue")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison")
        st.pyplot(fig)

    #  Show detailed predictions
    st.subheader(" Detailed Predictions")
    for model, results in all_results.items():
        if results:
            st.markdown(f"### {model} Results")
            df = pd.DataFrame(results)
            st.dataframe(df.head(50))  # show first 50 for readability
