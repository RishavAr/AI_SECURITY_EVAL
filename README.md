\
# 🔐 AI Security Eval – Benchmarking LLMs on Cybersecurity Payloads

## 📌 Overview

AI Security Eval is a **Streamlit-based evaluation platform** for testing **Large Language Models (LLMs)** against real-world cybersecurity payloads.

It leverages the [PayloadsAllTheThings](https://github.com/swisskyrepo/PayloadsAllTheThings) repository to simulate **malicious attack payloads** and generates a balanced dataset with benign samples. The system then evaluates different LLMs (OpenAI GPT family, Hugging Face models) on their ability to classify malicious vs. benign inputs.

---

## ⚙️ Features

* 🔄 **Automatic dataset generation** from PayloadsAllTheThings.
* 📊 **Balanced malicious + benign dataset** for fair evaluation.
* 🧪 **Evaluation framework** with accuracy, precision, recall, and F1-score.
* 📈 **Streamlit Dashboard** for interactive visualization.
* 🤖 **Multi-model support**:

  * GPT-3.5
  * GPT-4o
  * GPT-4o-mini

---

## 📂 Dataset Generation

We generate datasets by extracting payloads from PayloadsAllTheThings:

```bash
python3 generate_dataset.py
```

This script:

* Collects **malicious payloads** from `.md` files.
* Adds automatically generated **benign samples**.
* Stores everything in `datasets/cyberevalbench.jsonl`.

Example:

```json
{"task": "sql_injection", "input": "' OR 1=1 --", "label": "malicious"}
{"task": "benign_text", "input": "Hello world, this is a safe message.", "label": "benign"}
```

---

## 🚀 Live Dashboard

 [Streamlit](https://ai-security-eval.streamlit.app)

Deployment (Streamlit Cloud):

* Add your **OpenAI API key** in the app settings.
* Push changes to GitHub, Streamlit will auto-deploy.

---

## 📊 Results – Model Performance

We evaluated multiple GPT models on the **cybersecurity dataset** (PayloadsAllTheThings + benign samples).
Below is the comparison table (50-sample evaluation runs):

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| GPT-3.5 Turbo | 0.76     | 0.74      | 0.77   | 0.75     |
| GPT-4o        | 0.84     | 0.83      | 0.85   | 0.84     |
| GPT-4o-mini   | 0.81     | 0.79      | 0.82   | 0.80     |



> 📌 Metrics are averaged across classification reports displayed in the Streamlit dashboard.

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Streamlit** (UI Dashboard)
* **Transformers** (Hugging Face models)
* **OpenAI API** (GPT family models)
* **scikit-learn** (classification metrics)

---

## 🔮 Next Steps

* ✅ Expand evaluation to Hugging Face open-source models (Mistral, LLaMA).
* ✅ Support larger dataset sizes (31k+ samples).
* 🔐 Add adversarial robustness metrics.
* 📊 Visualize confusion matrices per model.

---

## 📎 References

* [PayloadsAllTheThings](https://github.com/swisskyrepo/PayloadsAllTheThings)
* [Streamlit](https://streamlit.io/)
* [OpenAI Models](https://platform.openai.com/docs/overview)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

💡 *Maintained by Rishav Aryan*
