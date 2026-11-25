"""Step 1: read documents, chunk them, build the FAISS vector index."""
import glob, os, json
import numpy as np, faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# File-name prefix decides the document type (used for intent routing)
PREFIX_TO_TYPE = {"std_": "standard_lookup",
                  "scheme_": "certification_process",
                  "hallmark_": "hallmarking"}

def read_file(path):
    if path.lower().endswith(".pdf"):
        r = PdfReader(path)
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(r.pages)]
    with open(path, encoding="utf-8") as f:
        return [(1, f.read())]

def chunk(text, size=200, overlap=40):
    words = text.split()
    for i in range(0, max(len(words), 1), size - overlap):
        piece = " ".join(words[i:i + size])
        if len(piece.split()) >= 20:
            yield piece

def doc_type(name):
    for p, t in PREFIX_TO_TYPE.items():
        if name.startswith(p):
            return t
    return "other"

chunks = []
for path in glob.glob("data/raw/**/*", recursive=True):
    if not path.lower().endswith((".pdf", ".txt")):
        continue
    name = os.path.basename(path)
    category = os.path.basename(os.path.dirname(path))
    for page, text in read_file(path):
        for c in chunk(text):
            chunks.append({"text": c, "source": name, "page": page,
                           "category": category, "doc_type": doc_type(name)})

print("Total chunks:", len(chunks))
if not chunks:
    raise SystemExit("No documents found in data/raw/. Add PDFs/TXT files first.")
json.dump(chunks, open("data/chunks.json", "w"))

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                   show_progress_bar=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(np.array(emb, dtype="float32"))
faiss.write_index(index, "data/index.faiss")
print("Index built.")
