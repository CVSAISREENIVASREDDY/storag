from huggingface_hub import InferenceClient

def generate_answer(prompt: str, model_name: str, hf_token: str, max_new_tokens: int = 400) -> str:
    """
    Generate an answer from a Hugging Face model using the given prompt.
    """
    if not hf_token:
        raise ValueError("HF_API_TOKEN not found! Add it to your .env file.")

    # Initialize Hugging Face Inference API client
    client = InferenceClient(model=model_name, token=hf_token)

    # Call the model to generate text
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9
    )

    return response.choices[0].message.content        