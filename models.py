import openai

# Query GPT models
def query_gpt(prompt, model_choice):
    if model_choice == "gpt35":
        model = "gpt-3.5-turbo"
    elif model_choice == "gpt40":
        model = "gpt-4o"
    elif model_choice == "gpt4omini":
        model = "gpt-4o-mini"
    else:
        raise ValueError("Unsupported GPT model selected.")

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a cybersecurity evaluator. "
                           "Classify inputs strictly as 'malicious' or 'benign'."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )

    return response["choices"][0]["message"]["content"].strip()
