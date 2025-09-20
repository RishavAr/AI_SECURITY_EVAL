import os
from openai import OpenAI

# Load API key from Streamlit secrets or environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def query_openai(prompt, model="gpt-4o-mini"):
    """
    Query OpenAI chat models for classification.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Classify input as 'benign' or 'malicious'. Respond with one word only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        return f"error: {str(e)}"

def query_model(text, model_choice="openai"):
    """
    Dispatch to the correct model.
    """
    if model_choice == "openai":
        return query_openai(text, "gpt-4o-mini")
    elif model_choice == "gpt4o":
        return query_openai(text, "gpt-4o")
    elif model_choice == "gpt35":
        return query_openai(text, "gpt-3.5-turbo")
    else:
        return "benign"  # fallback
