import streamlit as st
from evaluator import run_evaluation

st.set_page_config(page_title="AI Security Evaluator", layout="wide")
st.title("LLM – AI Security Evaluator (GPT only)")

# Sidebar Settings
st.sidebar.header("Settings")
sample_limit = st.sidebar.number_input(
    "Number of samples", min_value=1, max_value=500, value=50
)

# Only GPT models available
model_choice = st.sidebar.selectbox(
    "Choose model to evaluate",
    ["gpt35", "gpt40", "gpt4omini"]
)

if st.sidebar.button("Run Evaluation"):
    dataset_path = "datasets/cyberevalbench.jsonl"

    with st.spinner(f"Running {model_choice} on {sample_limit} samples…"):
        results, report = run_evaluation(
            dataset_path=dataset_path,
            model_choice=model_choice,
            limit=sample_limit,
        )

    st.subheader("Classification Report")
    st.json(report)

    st.subheader(" Accuracy")
    st.write(f"{report['accuracy'] * 100:.2f}%")

    st.subheader(" Predictions")
    st.dataframe(results)
