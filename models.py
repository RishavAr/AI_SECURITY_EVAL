import openai

def normalize_label(output: str) -> str:
    """Force GPT outputs into {malicious, benign} only."""
    output = output.strip().lower()
    if "malicious" in output:
        return "malicious"
    if "benign" in output:
        return "benign"
    return "benign"  # default fallback

def query_gpt(prompt: str, model_choice: str) -> str:
    if model_choice == "gpt3.5":
        model = "gpt-3.5-turbo"
    elif model_choice == "gpt4o":
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
                "content": (
                    "You are a cybersecurity evaluator. "
                    "Classify inputs strictly as 'malicious' or 'benign'. "
                    "Return only one word: malicious OR benign."
                )
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=5
    )

    raw_output = response["choices"][0]["message"]["content"]
    return normalize_label(raw_output)
