import os
from backend.rag_pipeline import RAGPipeline

CASES = [
    ("How many days of paid annual leave do employees get?", ["20", "annual leave"]),
    ("How many days can employees work remotely?", ["2", "remote"]),
    ("What are the normal working hours?", ["9:00", "6:00"]),
    ("Who should employees contact for technical issues?", ["IT support", "help desk"]),
]

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    pdf = os.path.join(root, "data", "sample.pdf")
    rag = RAGPipeline()
    info = rag.ingest(pdf)
    print(f"Indexed {info['chunks']} chunks from {info['source']}\n")
    correct = grounded = cited = 0
    for q, keywords in CASES:
        r = rag.ask(q)
        text = r["answer"].lower()
        ok = all(k.lower() in text for k in keywords)
        correct += int(ok)
        grounded += int(r["grounding"]["grounded"])
        cited += int(r["citation_verification"]["verified"])
        print("Q:", q)
        print("A:", r["answer"])
        print("AnswerCorrect:", ok, "CitationVerified:", r["citation_verification"]["verified"], "Grounded:", r["grounding"]["grounded"])
        print("-" * 70)
    n = len(CASES)
    print(f"Answer accuracy: {correct}/{n} = {correct/n:.1%}")
    print(f"Citation verification: {cited}/{n} = {cited/n:.1%}")
    print(f"Grounding check: {grounded}/{n} = {grounded/n:.1%}")
