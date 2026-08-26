from generator import generate_answer
import pymupdf
from chunker import chunk_pages
from rrf import reciprocal_rank_fusion
from reranker import rerank

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# 1. READ PDF WITH PAGE NUMBERS
# --------------------------------------------------

pdf = pymupdf.open("data/sample.pdf")

pages = []

for page_number, page in enumerate(pdf, start=1):

    pages.append({
        "page": page_number,
        "text": page.get_text()
    })


# --------------------------------------------------
# 2. CREATE CHUNKS
# --------------------------------------------------

chunks = chunk_pages(pages)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 3. QUESTION
# --------------------------------------------------

query = "How many days of paid annual leave do employees get?"


# --------------------------------------------------
# 4. TEXT FOR SEARCH
# --------------------------------------------------

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]


# --------------------------------------------------
# 5. LOAD EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 6. SEMANTIC INDEX
# --------------------------------------------------

embeddings = model.encode(chunk_texts)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# --------------------------------------------------
# 7. BM25 INDEX
# --------------------------------------------------

tokenized_chunks = [
    text.lower().split()
    for text in chunk_texts
]

bm25 = BM25Okapi(tokenized_chunks)


# --------------------------------------------------
# 8. SEMANTIC SEARCH
# --------------------------------------------------

query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding
).astype("float32")

distances, indices = index.search(
    query_embedding,
    k=3
)

semantic_results = []

for i in range(len(indices[0])):

    chunk_index = indices[0][i]

    semantic_results.append({
        "chunk": chunk_texts[chunk_index],
        "page": chunks[chunk_index]["page"],
        "score": float(distances[0][i])
    })


# --------------------------------------------------
# 9. BM25 SEARCH
# --------------------------------------------------

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(
    tokenized_query
)

keyword_indices = np.argsort(
    bm25_scores
)[::-1][:3]

keyword_results = []

for i in keyword_indices:

    keyword_results.append({
        "chunk": chunk_texts[i],
        "page": chunks[i]["page"],
        "score": float(bm25_scores[i])
    })


# --------------------------------------------------
# 10. RRF
# --------------------------------------------------

hybrid_results = reciprocal_rank_fusion(
    semantic_results,
    keyword_results
)


# --------------------------------------------------
# 11. GET HYBRID CHUNKS
# --------------------------------------------------

hybrid_chunks = [
    chunk
    for chunk, score in hybrid_results
]


# --------------------------------------------------
# 12. RERANK
# --------------------------------------------------

reranked_results = rerank(
    query,
    hybrid_chunks,
    top_k=2
)


# --------------------------------------------------
# 13. RESTORE PAGE NUMBERS
# --------------------------------------------------

final_results = []

for result in reranked_results:

    matching_chunk = next(
        chunk
        for chunk in chunks
        if chunk["text"] == result["chunk"]
    )

    final_results.append({
        "chunk": result["chunk"],
        "page": matching_chunk["page"],
        "score": result["score"]
    })


# --------------------------------------------------
# 14. BUILD CITATION-AWARE CONTEXT
# --------------------------------------------------

context_parts = []

for result in final_results:

    context_parts.append(
        f"[Source: sample.pdf | Page {result['page']}]\n"
        f"{result['chunk']}"
    )

context = "\n\n".join(context_parts)


# --------------------------------------------------
# 15. DISPLAY RESULTS
# --------------------------------------------------

print("\n=== FINAL RETRIEVED RESULTS ===")

for rank, result in enumerate(
    final_results,
    start=1
):

    print(f"\nRank {rank}")
    print(result["chunk"])
    print("Page:", result["page"])
    print("Reranker Score:", result["score"])

    print("-" * 60)


# --------------------------------------------------
# 16. DISPLAY CONTEXT
# --------------------------------------------------

print("\n=== CITATION-AWARE CONTEXT ===")

print(context)


# --------------------------------------------------
# 17. GENERATE ANSWER
# --------------------------------------------------

answer = generate_answer(
    query,
    context
)


# --------------------------------------------------
# 18. FINAL ANSWER
# --------------------------------------------------

print("\n=== FINAL ANSWER ===")

print(answer)