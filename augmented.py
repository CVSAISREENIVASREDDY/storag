from typing import List, Dict

def build_prompt(query: str, retrieved_docs: List[Dict]) -> str:
    context = "\n\n".join(
        [f"From {d['source']} (page {d['page']}):\n{d['text']}" for d in retrieved_docs]
    )
    prompt = f"""
You are a helpful assistant.
Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt
