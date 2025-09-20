import os
from transformers import pipeline

# Cache Hugging Face pipelines
hf_pipelines = {}

def query_hf(prompt, model_name):
    """
    Query a Hugging Face model safely.
    Supports both classification models and LLMs.
    """
    if model_name not in hf_pipelines:
        hf_token = os.getenv("HF_TOKEN")

        # Detect pipeline type automatically
        if any(x in model_name.lower() for x in ["distilbert", "bert", "roberta", "classifier"]):
            task = "text-classification"
        else:
            task = "text-generation"

        hf_pipelines[model_name] = pipeline(
            task,
            model=model_name,
            device=-1,     # Force CPU (Streamlit Cloud has no GPU)
            token=hf_token
        )

    # Run inference
    if hf_pipelines[model_name].task == "text-classification":
        outputs = hf_pipelines[model_name](prompt)
        return outputs[0]["label"]

    else:  # text-generation
        outputs = hf_pipelines[model_name](prompt, max_new_tokens=128)
        return outputs[0]["generated_text"]


def query_model(prompt, model_choice="mistral"):
    """
    High-level interface for supported models.
    """
    if model_choice == "mistral":
        return query_hf(prompt, "mistralai/Mistral-7B-Instruct-v0.3")

    elif model_choice == "llama":
        return query_hf(prompt, "meta-llama/Meta-Llama-3-8B-Instruct")

    elif model_choice == "classifier":
        return query_hf(prompt, "distilbert-base-uncased-finetuned-sst-2-english")

    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

