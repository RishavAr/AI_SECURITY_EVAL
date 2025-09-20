import os
import openai
from transformers import pipeline

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Hugging Face models only when needed
mistral = None
llama = None

def query_model(prompt, backend="openai", model="gpt-4o-mini"):
    global mistral, llama

    if backend == "openai":
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Classify cybersecurity inputs"},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()

    elif backend == "mistral":
        if mistral is None:
            mistral = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
        out = mistral(prompt, max_new_tokens=50)[0]["generated_text"]
        return "malicious" if "malicious" in out.lower() else "benign"

    elif backend == "llama":
        if llama is None:
            llama = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
        out = llama(prompt, max_new_tokens=50)[0]["generated_text"]
        return "malicious" if "malicious" in out.lower() else "benign"

    else:
        return "unknown"
