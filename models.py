from transformers import pipeline
import openai
import os

hf_pipelines = {}

def query_hf(text, model_name):
    if model_name not in hf_pipelines:
        hf_pipelines[model_name] = pipeline(
            "text-classification", 
            model=model_name, 
            device=-1  # force CPU
        )
    outputs = hf_pipelines[model_name](text)  # ⚡ only pass text, not full prompt
    return outputs[0]["label"]

def query_openai(prompt, model="gpt-4o-mini"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Classify the input as MALICIOUS or BENIGN. Respond only with the label."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def query_model(text, model_choice="classifier"):
    prompt = f"Task: classify input\nInput: {text}"

    if model_choice == "classifier":
        return query_hf(text, "distilbert-base-uncased-finetuned-sst-2-english")

    elif model_choice == "mistral":
        return query_hf(text, "mistralai/Mistral-7B-Instruct-v0.3")

    elif model_choice == "llama":
        return query_hf(text, "meta-llama/Meta-Llama-3-8B-Instruct")

    elif model_choice == "openai":
        return query_openai(prompt)

    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")
