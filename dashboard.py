import streamlit as st
import pandas as pd
import evaluator

st.title("CyberEvalBench Results")

results = evaluator.run_evaluation()
df = pd.DataFrame(results)
st.dataframe(df)

task_summary = df.groupby("task").apply(
    lambda g: (g["expected"] == g["prediction"]).mean()
).reset_index(name="accuracy")

st.bar_chart(task_summary.set_index("task"))
