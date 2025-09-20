import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from evaluator import run_evaluation

# Page setup
st.set_page_config(page_title="CyberEvalBench", layout="wide")
st.title("CyberEvalBench: LLM Security Evaluation")

# Sidebar controls
st.sidebar.header("Settings")
backend = st.sidebar.selectbox("Choose Model Backend", ["openai", "mistral", "llama"])
limit = st.sidebar.slider("Number of samples to evaluate", 10, 100, 50)

st.markdown(
    """
    This dashboard evaluates LLMs on cybersecurity prompts.  
    Choose your backend (OpenAI GPT, Hugging Face Mistral, or LLaMA) and run up to 100 tasks.  
    """
)

# Run evaluation
if st.button("Run Evaluation"):
    with st.spinner(f"Evaluating {limit} samples with {backend}..."):
        try:
            results = run_evaluation(limit=limit, backend=backend)
            df = pd.DataFrame(results)

            # Display table
            st.subheader(" Evaluation Results")
            st.dataframe(df, use_container_width=True)

            # Plot prediction counts
            if "prediction" in df.columns:
                counts = df["prediction"].value_counts()
                fig, ax = plt.subplots()
                counts.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"])
                ax.set_title("Prediction Distribution")
                ax.set_ylabel("Count")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during evaluation: {e}")
