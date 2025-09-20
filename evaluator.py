import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_gpt(text, model_choice="gpt-4o-mini"):
    """
    Query an OpenAI GPT model for binary classification.
    """
    if model_choice == "gpt35":
        model = "gpt-3.5-turbo"
    elif model_choice == "gpt4o":
        model = "gpt-4o"
    elif model_choice == "gpt4omini":
        model = "gpt-4o-mini"
    else:
        raise ValueError(f"Unsupported GPT model: {model_choice}")

    system_prompt = (
        "You are a cybersecurity classifier. "
        "Classify the input strictly as 'malicious' or 'benign'. "
        "Only output one word: malicious or benign."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"OpenAI query failed: {e}")
        return "benign"  # fallback safe
