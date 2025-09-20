import streamlit as st
import pandas as pd
from evaluator import run_evaluation

st.set_page_config(page_title="CyberEvalBench", layout="wide")

st.title("LLM – AI Security Evaluator")

# Dataset path
dataset_path = "datasets/cyberevalbench.jsonl"

# UI controls
sample_limit = st.number_input("Number of samples", min_value=1, value=50, step=1)

model_choice = st.selectbox(
    "Choose models to evaluate",
    ["classifier", "mistral", "llama", "openai"]  # Added OpenAI here
)

# Run evaluation
if st.button("Run Evaluation"):
    with st.spinner(f"Running {model_choice} on {sample_limit} samples…"):
        results, report = run_evaluation(
            dataset_path=dataset_path,
            model_choice=model_choice,
            limit=sample_limit
        )

        # Show classification report
        st.subheader("Classification Report")
        st.json(report)

        # Show dataframe
        df = pd.DataFrame(results)
        st.dataframe(df)

        # Accuracy metric
        if "accuracy" in report:
            st.metric("Accuracy", f"{report['accuracy']*100:.2f}%")
