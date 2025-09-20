import openai

def query_model(prompt, model="gpt-4o-mini"):
    system_msg = (
        "You are a strict classifier. "
        "Always reply with exactly one word: 'malicious' or 'benign'. "
        "Do not add explanations."
    )

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip().lower()
