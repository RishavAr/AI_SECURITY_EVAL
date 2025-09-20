# dashboard.py
import streamlit as st
from evaluator import run_evaluation

st.title("LLM – AI Security Evaluator")

# Sidebar settings
st.sidebar.header("Settings")
sample_limit = st.sidebar.number_input("Number of samples", min_value=1, max_value=500, value=50)

# Dropdown with all model options
model_choice = st.sidebar.selectbox(
    "Choose model to evaluate",
    [
        "classifier",   # DistilBERT baseline
        "mistral",      # Mistral (OpenAI hosted)
        "llama",        # Llama-3 (OpenAI hosted)
        "gpt35",        # GPT-3.5
        "gpt40",        # GPT-4o
        "gpt4omini"     # GPT-4o-mini (best performer in your runs)
    ]
)

if st.sidebar.button("Run Evaluation"):
    dataset_path = "datasets/cyberevalbench.jsonl"
    with st.spinner(f"Running {model_choice} on {sample_limit} samples…"):
        results, report = run_evaluation(
            dataset_path=dataset_path,
            model_choice=model_choice,
            limit=sample_limit
        )

    st.subheader("Classification Report")
    st.json(report)

    st.subheader("Accuracy")
    st.write(f"{report['accuracy']*100:.2f}%")

    st.subheader("Predictions")
    st.dataframe(results)
