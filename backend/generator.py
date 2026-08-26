import ollama


def generate_answer(query, context):

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided context.

IMPORTANT RULES:
1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer is not present in the context, say:
   "I don't know based on the provided document."
4. Give a concise, direct answer.
5. Include the source citation from the context in your answer.
6. Preserve the page number exactly as provided.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    query = "How many days of paid annual leave do employees get?"

    context = """
[Source: sample.pdf | Page 1]
Full-time employees are entitled to 20 days of paid annual
leave per calendar year.
"""

    answer = generate_answer(query, context)

    print("Question:")
    print(query)

    print("\n=== GENERATED ANSWER ===")
    print(answer)