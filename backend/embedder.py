from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Employees normally work from 9:00 AM to 6:00 PM."

embedding = model.encode(text)

print(embedding)
print("Vector length:", len(embedding))