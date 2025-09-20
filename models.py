import os
import openai
from transformers import pipeline

# OpenAI client (needs OPENAI_API_KEY in env)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Hugging Face models (lazy init so we don’t load all at once)
hf_pipelines = {}

def query_openai(prompt, model="gpt-4o-mini"):
    """Query OpenAI chat models"""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a cybersecurity classifier. Output only 'benign' or 'malicious'."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip().lower()

def query_hf(prompt, model_name):
    """Query Hugging Face models with transformers pipeline"""
    if model_name not in hf_pipelines:
        hf_pipelines[model_name] = pipeline("text-classification", model=model_name, device=-1)  # CPU
    classifier = hf_pipelines[model_name]
    out = classifier(prompt, truncation=True)[0]
    return out["label"].lower()

def query_model(prompt, model_choice="gpt-4o-mini"):
    """
    Unified interface:
    - OpenAI: "gpt-4o-mini"
    - HF: "mistral", "llama"
    """
    if model_choice == "gpt-4o-mini":
        return query_openai(prompt, "gpt-4o-mini")
    elif model_choice == "mistral":
        return query_hf(prompt, "mistralai/Mistral-7B-Instruct-v0.3")
    elif model_choice == "llama":
        return query_hf(prompt, "meta-llama/Meta-Llama-3-8B-Instruct")
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")
