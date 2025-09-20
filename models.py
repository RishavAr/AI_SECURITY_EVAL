import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_gpt(prompt, model_choice="gpt-4o-mini"):
    """
    Query OpenAI GPT models for classification.
    Only supports: gpt35, gpt4o, gpt4omini
    """
    if model_choice == "gpt35":
        model = "gpt-3.5-turbo"
    elif model_choice == "gpt4o":
        model = "gpt-4o"
    elif model_choice == "gpt4omini":
        model = "gpt-4o-mini"
    else:
        raise ValueError("Unsupported GPT model selected.")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a security classifier. Label text as 'benign' or 'malicious'."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip().lower()
