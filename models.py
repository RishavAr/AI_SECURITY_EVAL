import os
import openai
from transformers import pipeline

# Load API key safely (will use Streamlit secrets in cloud)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Cache Hugging Face pipelines
_mistral_pipe = None
_llama_pipe = None

def query_model(prompt, backend="openai", model="gpt-4o-mini"):
    global _mistral_pipe, _llama_pipe

    if backend == "openai":
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Classify as 'benign' or 'malicious'. Do not explain."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()

    elif backend == "mistral":
        if _mistral_pipe is None:
            _mistral_pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
        result = _mistral_pipe(prompt, max_length=200)
        return result[0]["generated_text"].strip()

    elif backend == "llama":
        if _llama_pipe is None:
            _llama_pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")
        result = _llama_pipe(prompt, max_length=200)
        return result[0]["generated_text"].strip()

    else:
        raise ValueError(f"Unsupported backend: {backend}")
