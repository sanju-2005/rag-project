from backend.retriever import HybridRetriever


PDF_PATH = "data/sample.pdf"


retriever = HybridRetriever()

chunk_count = retriever.build(PDF_PATH)

print(f"Indexed {chunk_count} chunks.")

results = retriever.search(
    "What are the normal working hours?",
    top_k=5,
    rerank_top_k=3,
)

print("\nRetrieved results:\n")

for i, result in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(f"Source: {result.get('source')}")
    print(f"Page: {result.get('page')}")
    print(f"Score: {result.get('score')}")
    print(f"Text: {result.get('text')}")
    print()