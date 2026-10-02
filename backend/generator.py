import ollama

MODEL_NAME = "llama3.2:latest"


def generate_answer(query, context):
    prompt = f"""You are a document question-answering assistant.
Answer ONLY from the supplied context. Do not use outside knowledge or invent facts.
If the answer is not supported by the context, say exactly: I don't know based on the provided document.
Keep the answer concise and include the relevant source page citation(s), e.g. (Page 1).

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()
