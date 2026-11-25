# 🏛️ BIS Standards Assistant — Gemini RAG

> An AI-powered assistant for Indian Standards, BIS certification, and hallmarking, built with **Google Gemini**, **FAISS vector search**, **RAG**, and a beautiful **Streamlit** chat UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-4285F4?logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Google Gemini Integration** | Uses Gemini models (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-2.5-pro`) via Google AI Studio API |
| 🖼️ **Multimodal Image Inspection** | Upload photos of ISI marks, 6-digit HUID hallmarks, or product labels for visual compliance checks |
| 💬 **Gemini-Themed Chat UI** | Authentic dark-mode interface with multi-turn conversation memory, intent pill badges & interactive question starters |
| 🧭 **Intent Routing** | Classifies queries into `standard_lookup`, `certification_process`, or `hallmarking` using **TF-IDF + Logistic Regression** |
| 🔍 **Vector Search Grounding** | FAISS vector search with `all-MiniLM-L6-v2` dense embeddings, returning exact **source document & page citations** |
| 📋 **Chat Management** | Clear history, export transcripts as JSON, adjust retrieval parameters on the fly |

---

## 🗂️ Project Structure

```
bis-assistant/
├── app.py              # Gemini-style Streamlit UI with chat history & media upload
├── rag.py              # RAG pipeline, FAISS search, Gemini REST client & fallback logic
├── build_index.py      # Document chunking, SentenceTransformer embeddings & FAISS indexing
├── intent.py           # Intent classifier training (TF-IDF + Logistic Regression)
├── eval.py             # Recall@k evaluation against keyword baseline
├── intent_data.csv     # Intent classification training dataset
├── eval_questions.csv  # Benchmark evaluation questions with ground-truth sources
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment config for GEMINI_API_KEY
└── data/               # Place your BIS PDFs/TXT documents here
    └── raw/
        ├── electrical/
        ├── food/
        └── toys/
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Streamlit** | Web UI framework |
| **Google Gemini API** | LLM for answer generation & multimodal reasoning |
| **FAISS** (`faiss-cpu`) | Fast vector similarity search |
| **SentenceTransformers** (`all-MiniLM-L6-v2`) | Dense text embeddings |
| **scikit-learn** | TF-IDF vectorizer + Logistic Regression intent classifier |
| **PyPDF** (`pypdf`) | PDF parsing and text extraction |
| **pandas / numpy** | Data handling |
| **python-dotenv** | Environment variable management |
| **joblib** | Model serialization |
| **requests** | Gemini REST API calls |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/SakshamKumar4/bis-assistant.git
cd bis-assistant
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API Key
Copy `.env.example` to `.env` and add your key:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
> 💡 You can also paste the API key directly into the Streamlit sidebar at runtime — no `.env` file needed.

---

## 🔑 Getting a Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Copy the key and paste it into the app sidebar or into your `.env` file

---

## 📂 Adding Your Own BIS Documents

Place real BIS PDFs or TXT files into `data/raw/<category>/` (`electrical`, `food`, `toys`).  
Use these **filename prefixes** for automatic intent routing:

| Prefix | Document Type |
|---|---|
| `std_*` | Standards & product-to-standard specs (e.g. `std_electric_iron_is302.pdf`) |
| `scheme_*` | Certification scheme guidelines, fee schedules, licensing procedures |
| `hallmark_*` | Hallmarking guidelines, HUID verification, gold/silver purity rules |

---

## 🚀 Run the Pipeline

```bash
# Step 1 — Build the FAISS vector index from your documents
python build_index.py

# Step 2 — Train the intent classifier
python intent.py

# Step 3 — (Optional) Evaluate retrieval accuracy
python eval.py

# Step 4 — Launch the Streamlit app
streamlit run app.py
```

Open **http://localhost:8501** in your browser. 🎉

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Author

**Saksham Kumar**  
GitHub: [@SakshamKumar4](https://github.com/SakshamKumar4)
