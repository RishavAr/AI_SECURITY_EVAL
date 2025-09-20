import streamlit as st
import pandas as pd
from evaluator import run_evaluation

st.set_page_config(page_title="AI Security Evaluator", layout="wide")

st.title(" LLM – AI Security Evaluator")

# Dataset path
dataset_path = "datasets/cyberevalbench.jsonl"

# Sidebar settings
st.sidebar.header("Settings")
sample_limit = st.sidebar.number_input(
    "Number of samples", min_value=10, max_value=31694, value=50, step=10
)

# Model selector (only OpenAI models)
model_choice = st.sidebar.selectbox(
    "Choose model to evaluate",
    ["gpt_4o_mini", "gpt4o", "gpt3.5"]
)

# Run evaluation button
if st.sidebar.button("Run Evaluation"):
    st.write(f"### Running {model_choice} on {sample_limit} samples…")

    results, report = run_evaluation(
        dataset_path=dataset_path,
        model_choice=model_choice,
        limit=sample_limit
    )

    if "error" in report:
        st.error(report["error"])
    else:
        # Show classification report
        st.subheader("📊 Classification Report")
        st.json(report)

        # Accuracy
        accuracy = report.get("accuracy", 0)
        st.metric(label="Accuracy", value=f"{accuracy*100:.2f}%")

        # Show predictions
        df = pd.DataFrame(results)
        st.subheader("🔎 Predictions")
        st.dataframe(df)
