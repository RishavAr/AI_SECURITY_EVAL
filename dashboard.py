import streamlit as st
import json
from evaluator import run_evaluation

st.title("LLM – AI Security Evaluator (GPT Models Only)")

# Sidebar config
st.sidebar.header("Settings")
sample_limit = st.sidebar.number_input("Number of samples", min_value=1, max_value=500, value=50, step=1)
model_choice = st.sidebar.selectbox("Choose model to evaluate", ["gpt35", "gpt4o", "gpt4omini"])

dataset_path = "datasets/cyberevalbench.jsonl"

if st.sidebar.button("Run Evaluation"):
    st.write(f"Running {model_choice} on {sample_limit} samples…")

    results, report = run_evaluation(
        dataset_path=dataset_path,
        model_choice=model_choice,
        limit=sample_limit
    )

    # Show classification report
    st.subheader("Classification Report")
    st.json(report)

    # Show accuracy
    accuracy = report["accuracy"]
    st.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%")

    # Show predictions
    st.subheader("Predictions")
    st.dataframe(results)
