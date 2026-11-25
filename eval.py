"""Step 4: evaluation - Recall@k for RAG retrieval vs baselines."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rag import retrieve, chunks

df = pd.read_csv("eval_questions.csv")

vec = TfidfVectorizer(stop_words="english").fit([c["text"] for c in chunks])
M = vec.transform([c["text"] for c in chunks])

def baseline(q, k=4):
    s = (M @ vec.transform([q]).T).toarray().ravel()
    return [chunks[i] for i in s.argsort()[::-1][:k]]

def recall(fn, k=4):
    ok = 0
    for q, exp in zip(df.question, df.expected_source):
        ok += any(h["source"] == exp for h in fn(q)[:k])
    return ok / len(df)

print(f"Questions evaluated: {len(df)}")
print(f"Embedding + intent routing : Recall@4 = {recall(lambda q: retrieve(q, 4, True)):.2f}")
print(f"Embedding only             : Recall@4 = {recall(lambda q: retrieve(q, 4, False)):.2f}")
print(f"TF-IDF keyword baseline    : Recall@4 = {recall(baseline):.2f}")
