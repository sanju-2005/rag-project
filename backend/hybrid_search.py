import pymupdf
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from chunker import chunk_text


# --------------------------------------------------
# 1. READ PDF
# --------------------------------------------------

pdf = pymupdf.open("data/sample.pdf")

full_text = ""

for page in pdf:
    full_text += page.get_text()


# --------------------------------------------------
# 2. CREATE CHUNKS
# --------------------------------------------------

chunks = chunk_text(full_text)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 3. LOAD EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 4. CREATE SEMANTIC VECTOR INDEX
# --------------------------------------------------

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# --------------------------------------------------
# 5. CREATE BM25 KEYWORD INDEX
# --------------------------------------------------

tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# --------------------------------------------------
# 6. ASK QUESTION
# --------------------------------------------------

query = "How many days of paid annual leave do employees get?"


# --------------------------------------------------
# 7. SEMANTIC SEARCH
# --------------------------------------------------

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")

distances, indices = index.search(
    query_embedding,
    k=3
)

semantic_results = []

for i in range(len(indices[0])):
    chunk_index = indices[0][i]

    semantic_results.append({
        "chunk": chunks[chunk_index],
        "score": distances[0][i]
    })


# --------------------------------------------------
# 8. BM25 KEYWORD SEARCH
# --------------------------------------------------

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(tokenized_query)

keyword_indices = np.argsort(bm25_scores)[::-1][:3]

keyword_results = []

for i in keyword_indices:

    keyword_results.append({
        "chunk": chunks[i],
        "score": bm25_scores[i]
    })


# --------------------------------------------------
# 9. RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=60
):

    scores = {}

    for rank, result in enumerate(semantic_results):

        chunk = result["chunk"]

        scores[chunk] = scores.get(chunk, 0) + (
            1 / (k + rank + 1)
        )

    for rank, result in enumerate(keyword_results):

        chunk = result["chunk"]

        scores[chunk] = scores.get(chunk, 0) + (
            1 / (k + rank + 1)
        )

    ranked_chunks = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_chunks


# --------------------------------------------------
# 10. COMBINE BOTH SEARCH SYSTEMS
# --------------------------------------------------

hybrid_results = reciprocal_rank_fusion(
    semantic_results,
    keyword_results
)


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print("\nQuestion:")
print(query)

print("\n=== SEMANTIC SEARCH ===")

for rank, result in enumerate(
    semantic_results,
    start=1
):

    print(f"\nRank {rank}")
    print(result["chunk"])
    print("Distance:", result["score"])


print("\n=== BM25 KEYWORD SEARCH ===")

for rank, result in enumerate(
    keyword_results,
    start=1
):

    print(f"\nRank {rank}")
    print(result["chunk"])
    print("BM25 Score:", result["score"])


print("\n=== HYBRID SEARCH ===")

for rank, (chunk, score) in enumerate(
    hybrid_results,
    start=1
):

    print(f"\nRank {rank}")
    print(chunk)
    print("RRF Score:", score)