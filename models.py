# models.py
from transformers import pipeline
from openai import OpenAI

hf_pipelines = {}
client = OpenAI()

def query_hf(text, model_name):
    if model_name not in hf_pipelines:
        hf_pipelines[model_name] = pipeline(
            "text-classification",
            model=model_name,
            device=-1  # CPU
        )
    outputs = hf_pipelines[model_name](text)
    return outputs[0]["label"].lower()

def query_openai(text, model_name="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a security classifier. Reply strictly with 'benign' or 'malicious'."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()

def query_model(text, model_choice):
    if model_choice == "classifier":
        return query_hf(text, "distilbert-base-uncased-finetuned-sst-2-english")
    elif model_choice == "mistral":
        return query_openai(text, "mistral-small")  # hosted on OpenAI, not HF pipeline
    elif model_choice == "llama":
        return query_openai(text, "llama-3.1-8b-instruct")
    elif model_choice == "gpt35":
        return query_openai(text, "gpt-3.5-turbo")
    elif model_choice == "gpt40":
        return query_openai(text, "gpt-4o")
    elif model_choice == "openai":
        return query_openai(text, "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")
