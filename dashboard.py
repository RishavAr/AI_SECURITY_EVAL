import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from evaluator import run_evaluation

st.set_page_config(page_title="LLM", layout="wide")
st.title("LLM – AI Security Evaluator")

# Sidebar controls
st.sidebar.header(" Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["gpt-4o-mini", "mistral", "llama"]
)
sample_limit = st.sidebar.slider("Number of samples", min_value=10, max_value=500, value=50, step=10)

# Run evaluation
if st.sidebar.button("Run Evaluation"):
    st.info(f"Running {model_choice} on {sample_limit} samples…")
    results, report = run_evaluation(
        dataset_path="datasets/cyberevalbench.jsonl",
        model_choice=model_choice,
        limit=sample_limit,
    )

    df = pd.DataFrame(results)
    st.subheader(" Predictions")
    st.dataframe(df)

    # Classification Report
    st.subheader("Classification Report")
    st.json(report)

    # Accuracy plot
    accuracy = report["accuracy"]
    fig, ax = plt.subplots()
    ax.bar(["Accuracy"], [accuracy])
    ax.set_ylim([0, 1])
    st.pyplot(fig)
