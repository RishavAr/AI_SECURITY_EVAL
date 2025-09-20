import streamlit as st
import pandas as pd
from evaluator import run_evaluation

st.set_page_config(page_title="CyberEvalBench", layout="wide")

st.title("🔐 LLM – AI Security Evaluator (GPT Models Only)")

dataset_path = "datasets/cyberevalbench.jsonl"

# Sidebar settings
sample_limit = st.sidebar.number_input("Number of samples", min_value=10, max_value=500, value=50, step=10)
model_choice = st.sidebar.selectbox("Choose model", ["gpt35", "gpt4o", "gpt4omini"])

if st.button("Run Evaluation"):
    with st.spinner(f"Running {model_choice} on {sample_limit} samples…"):
        results, report = run_evaluation(dataset_path, model_choice=model_choice, limit=sample_limit)

        df = pd.DataFrame(results)
        st.subheader("🔎 Predictions")
        st.dataframe(df)

        st.subheader("📊 Classification Report")
        st.json(report)

        st.metric("Accuracy", f"{report['accuracy']*100:.2f}%")
