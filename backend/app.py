import os
import tempfile
from flask import Flask, request, render_template_string, jsonify
from .rag_pipeline import RAGPipeline

app = Flask(__name__)
pipeline = RAGPipeline()

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Hybrid RAG Document QA</title>
<style>
body{font-family:Inter,Arial,sans-serif;background:#0b1020;color:#e8edf7;margin:0}.wrap{max-width:980px;margin:40px auto;padding:24px}.card{background:#121a2e;border:1px solid #263352;border-radius:16px;padding:22px;margin:16px 0}h1{margin-bottom:6px}p{color:#aab5cc}input,button{padding:12px;border-radius:10px;border:1px solid #34415f;background:#0e1628;color:#fff}input{width:65%}button{cursor:pointer;background:#315efb;border:0;font-weight:700}textarea{width:100%;min-height:90px;box-sizing:border-box;padding:12px;border-radius:10px;background:#0e1628;color:#fff;border:1px solid #34415f}.answer{font-size:18px;line-height:1.6}.source{border-left:3px solid #4f7cff;padding:10px 14px;margin:10px 0;background:#0e1628}.badge{display:inline-block;padding:5px 9px;border-radius:8px;background:#20304f;margin-right:8px}.err{color:#ff8e8e}</style></head>
<body><div class="wrap"><h1>Hybrid RAG Document QA</h1><p>Semantic retrieval + BM25 + RRF + cross-encoder reranking + grounded local LLM.</p>
<div class="card"><h2>1. Upload document</h2><form method="post" action="/upload" enctype="multipart/form-data"><input type="file" name="file" accept=".pdf" required><button>Process PDF</button></form>{% if status %}<p>{{status}}</p>{% endif %}</div>
<div class="card"><h2>2. Ask a question</h2><form method="post" action="/ask"><textarea name="question" placeholder="e.g. How many days of paid annual leave do employees get?" required></textarea><br><br><button>Ask Document</button></form></div>
{% if result %}<div class="card"><h2>Answer</h2><div class="answer">{{result.answer}}</div><p><span class="badge">Citation: {{'Verified' if result.citation_verification.verified else 'Check'}}</span><span class="badge">Grounding: {{result.grounding.score}}</span></p><h3>Sources</h3>{% for s in result.sources %}<div class="source"><b>{{s.source}} — Page {{s.page}}</b><br>{{s.chunk}}</div>{% endfor %}</div>{% endif %}
{% if error %}<div class="card err">{{error}}</div>{% endif %}</div></body></html>"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return render_template_string(HTML, error="Please upload a PDF file.")
    temp_dir = tempfile.mkdtemp(prefix="rag_")
    path = os.path.join(temp_dir, file.filename)
    file.save(path)
    try:
        info = pipeline.ingest(path)
        return render_template_string(HTML, status=f"Processed {info['source']} — {info['chunks']} chunks indexed.")
    except Exception as exc:
        return render_template_string(HTML, error=str(exc))

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    try:
        result = pipeline.ask(question)
        return render_template_string(HTML, result=result)
    except Exception as exc:
        return render_template_string(HTML, error=str(exc))

@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    result = pipeline.ask(data["question"])
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
