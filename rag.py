"""Step 3: retrieval + generation (the RAG pipeline with Google Gemini & multimodal support)."""
import os
import json
import base64
import requests
import joblib
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables (.env if present)
load_dotenv()

# ==============================================================================
# 🔑 GOOGLE GEMINI API KEY CONFIGURATION
# Paste your Gemini API key from Google AI Studio between the quotes below:
# ==============================================================================
GEMINI_API_KEY = ""  # <-- PASTE YOUR GEMINI API KEY HERE OR USE .env
# ==============================================================================

# Load models and vector indices
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/index.faiss")
chunks = json.load(open("data/chunks.json"))
intent_clf = joblib.load("data/intent.pkl")

# Human-friendly intent mapping
INTENT_LABELS = {
    "standard_lookup": "Standard Lookup (IS Specification)",
    "certification_process": "Certification Process (ISI / Scheme)",
    "hallmarking": "Hallmarking & HUID (Gold / Silver)",
    "other": "General Inquiry"
}

def get_intent(query):
    """Predict user query intent using trained TF-IDF + Logistic Regression."""
    try:
        return intent_clf.predict([query])[0]
    except Exception:
        return "other"

def retrieve(query, k=4, use_intent=True):
    """Retrieve top-k relevant document chunks with optional intent filtering."""
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, min(50, len(chunks)))
    hits = [{**chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0]) if i != -1]
    if use_intent:
        intent = get_intent(query)
        routed = [h for h in hits if h.get("doc_type") == intent]
        if routed:  # Fall back to all hits if none match predicted intent
            hits = routed
    return hits[:k]

def build_prompt(query, hits):
    """Construct a grounding prompt with retrieved BIS document context."""
    if hits:
        ctx = "\n\n".join(
            f"--- Document: {h['source']} | Page: {h['page']} | Category: {h.get('category', 'N/A')} ---\n{h['text']}"
            for h in hits
        )
    else:
        ctx = "No specific document passages retrieved."
        
    return (
        "Answer the question using the context below from official Bureau of Indian Standards (BIS) documents.\n"
        "Guidelines:\n"
        "1. Cite sources with specific document name and page number (e.g. `[std_sample.txt p.1]`).\n"
        "2. If an image or media is provided, carefully examine the ISI mark, CML number, IS standard code, or 6-digit HUID hallmark.\n"
        "3. If the retrieved context does not fully answer the question, state what is known from the context and offer accurate general BIS guidance.\n"
        "4. Format your answer cleanly with bullet points, bold key terms, and markdown tables where suitable.\n\n"
        f"Retrieved BIS Context:\n{ctx}\n\n"
        f"User Question: {query}\n"
    )

def generate_gemini(prompt, api_key=None, gemini_model="gemini-flash-lite-latest",
                    image_b64=None, mime_type="image/jpeg", chat_history=None):
    """Generate response using Google Gemini REST API with automated resilient fallback."""
    key = api_key or (GEMINI_API_KEY.strip() if GEMINI_API_KEY else None) or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None, "Gemini API key not configured. Paste your API key in rag.py (GEMINI_API_KEY) or set GEMINI_API_KEY in .env.", None

    system_instruction = {
        "parts": [
            {
                "text": (
                    "You are the official Bureau of Indian Standards (BIS) AI Assistant. "
                    "You assist consumers, manufacturers, and jewellers with Indian Standards (IS), "
                    "ISI certification marks, Compulsory Registration Scheme (CRS), and gold/silver hallmarking (HUID). "
                    "Provide authoritative, polite, structured, and helpful responses. "
                    "Always mention standard numbers, required documents, and verification steps clearly."
                )
            }
        ]
    }

    contents = []

    # Include recent chat history for context awareness
    if chat_history:
        for msg in chat_history[-6:]:
            role = "user" if msg.get("role") == "user" else "model"
            msg_parts = [{"text": msg.get("content", "")}]
            contents.append({"role": role, "parts": msg_parts})

    # Add current query
    current_parts = [{"text": prompt}]
    if image_b64:
        current_parts.append({
            "inline_data": {
                "mime_type": mime_type or "image/jpeg",
                "data": image_b64
            }
        })

    contents.append({"role": "user", "parts": current_parts})

    payload = {
        "system_instruction": system_instruction,
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }
    }

    # Model candidates list starting with requested model
    candidates_to_try = [gemini_model]
    for alt in ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-flash-latest", "gemini-3.5-flash"]:
        if alt not in candidates_to_try:
            candidates_to_try.append(alt)

    last_err = None
    for m in candidates_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=45
            )
            data = response.json()

            if response.status_code == 200:
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                    generated_text = "".join(part.get("text", "") for part in candidates[0]["content"]["parts"])
                    return generated_text, None, m
            else:
                last_err = data.get("error", {}).get("message", f"HTTP {response.status_code}")
                # Retry on 404 (model retired) or 503 (high demand)
                if response.status_code in [404, 503]:
                    continue
                return None, f"Google Gemini Error: {last_err}", None
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            last_err = str(e)
            continue

    return None, f"Google Gemini Error: {last_err or 'Service busy, please retry.'}", None

def generate_ollama(prompt, model_name="llama3.2"):
    """Optional local fallback if Ollama is running."""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=15
        )
        if r.status_code == 200:
            return r.json().get("response")
    except Exception:
        pass
    return None

def answer(query, api_key=None, gemini_model="gemini-flash-lite-latest",
           image_bytes=None, mime_type="image/jpeg", chat_history=None, use_intent=True):
    """Full RAG pipeline answering with Gemini multimodal, intent routing, and fallbacks."""
    hits = retrieve(query, use_intent=use_intent)
    predicted_intent = get_intent(query)
    prompt = build_prompt(query, hits)

    image_b64 = None
    if image_bytes:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # 1. Attempt generation via Google Gemini
    gemini_resp, gemini_err, actual_model = generate_gemini(
        prompt=prompt,
        api_key=api_key,
        gemini_model=gemini_model,
        image_b64=image_b64,
        mime_type=mime_type,
        chat_history=chat_history
    )

    model_used = f"Google Gemini ({actual_model or gemini_model})" if gemini_resp else None
    final_answer = gemini_resp

    # 2. If no Gemini key was provided, try local Ollama if active
    has_gemini_key = bool(api_key or (GEMINI_API_KEY.strip() if GEMINI_API_KEY else None) or os.getenv("GEMINI_API_KEY"))
    if final_answer is None and not has_gemini_key:
        ollama_ans = generate_ollama(prompt)
        if ollama_ans:
            final_answer = ollama_ans
            model_used = "Ollama (llama3.2)"
            gemini_err = None

    # 3. Fallback when no LLM is connected
    if final_answer is None:
        if hits:
            lead_passage = hits[0]["text"]
            final_answer = (
                f"> ℹ️ *To enable AI generation & image analysis, paste your Gemini API key in `rag.py` (`GEMINI_API_KEY = \"...\"`) or set it in `.env`.*\n\n"
                f"**Top Relevant BIS Reference [{hits[0]['source']} p.{hits[0]['page']}]:**\n\n"
                f"{lead_passage}"
            )
        else:
            final_answer = "No matching BIS documents found in the database. Please check your query or add documentation to `data/raw/`."

    return {
        "intent": predicted_intent,
        "intent_label": INTENT_LABELS.get(predicted_intent, predicted_intent),
        "answer": final_answer,
        "sources": hits,
        "model_used": model_used,
        "error": gemini_err
    }

if __name__ == "__main__":
    print("BIS Assistant RAG Module loaded.")
    while True:
        q = input("\nEnter question (blank to exit): ").strip()
        if not q:
            break
        res = answer(q)
        print(f"\n[Intent]: {res['intent']} ({res['intent_label']})")
        print(f"[Model]: {res.get('model_used', 'Fallback')}")
        print(f"\n[Answer]:\n{res['answer']}")
        print("\n[Sources]:")
        for h in res["sources"]:
            print(f"  - {h['source']} (Page {h['page']}, Score {h['score']:.2f})")
