# 🏛️ BIS Standards Assistant — Gemini RAG

> An AI-powered assistant for Indian Standards, BIS certification, and hallmarking, built with **Google Gemini**, **FAISS vector search**, **RAG**, and a beautiful **Streamlit** chat UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-4285F4?logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![License](https://img.shields.io/badge/License-MIT-green)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://bis-assist.streamlit.app/)

> 🌐 **Live Demo:** The project is live and deployed on **[https://bis-assist.streamlit.app/](https://bis-assist.streamlit.app/)**

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
Bis-Assist/
├── app.py              # Gemini-style Streamlit UI with chat history & media upload
├── rag.py              # RAG pipeline, FAISS search, Gemini REST client & fallback logic
├── build_index.py      # Document chunking, SentenceTransformer embeddings & FAISS indexing
├── intent.py           # Intent classifier training (TF-IDF + Logistic Regression)
├── eval.py             # Recall@k evaluation against keyword baseline
├── intent_data.csv     # Intent classification training dataset
├── eval_questions.csv  # Benchmark evaluation questions with ground-truth sources
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment config for GEMINI_API_KEY
└── data/
    ├── chunks.json     # 346 indexed chunks from 7 official BIS PDFs
    ├── index.faiss     # Pre-built FAISS vector index (ready for deployment)
    ├── intent.pkl      # Trained TF-IDF + Logistic Regression intent classifier
    └── raw/            # 7 official BIS PDFs (grouped by category)
        ├── hallmarking/
        ├── certification/
        ├── electrical/
        └── standards/
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
git clone https://github.com/SakshamKumar4/Bis-Assist.git
cd Bis-Assist
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
> 💡 For Streamlit Cloud deployments, configure `GEMINI_API_KEY` in **App Settings → Secrets**.

---

## 🔑 Getting a Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Copy the key and paste it into your `.env` file (or `rag.py`)

---

## 📂 Indexed Official BIS Documents & Adding Your Own

The repository ships with a pre-built FAISS index (`346` chunks in `data/index.faiss` & `data/chunks.json`) built from **7 official BIS documents** (`bis.gov.in` / `crsbis.in`):
- **Hallmarking (`hallmark_*`):** Brief on Hallmarking Scheme (Oct 2026 stats), Revised Jeweller Registration Guidelines (Jul 2026), Mandatory Hallmarking Order — 385 Districts (Apr 2026)
- **Certification Schemes (`scheme_*`):** Guidelines for Grant of Licence under Scheme-I (Feb 2026), CRS Scheme-II Revised Fee Structure
- **Standards & QCOs (`std_*`):** Guidance Document on Quality Control Orders (QCOs), Simplified Procedure Product Listing (1,000+ IS standards)

To add more documents, place BIS PDFs or TXT files into `data/raw/<category>/` using these **filename prefixes** for automatic intent routing:

| Prefix | Document Type |
|---|---|
| `std_*` | Standards & product-to-standard specs (e.g. `std_qco_guidance_document.pdf`) |
| `scheme_*` | Certification scheme guidelines, fee schedules, licensing procedures |
| `hallmark_*` | Hallmarking guidelines, HUID verification, gold/silver purity rules |

---

## 🧪 Demo Showcase Questions

Try these queries in the app to verify grounded PDF citations & intent routing:
- *"What are the 3 marks in a gold hallmark under HUID and what is the hallmarking fee?"*
- *"What is the CRS Scheme-II fee for a new application, annual licence fee, and application processing fee?"*
- *"Under the 2026 Grant of Licence Guidelines, what are Option 1 and Option 2 for Scheme-I certification?"*
- *"What is the penalty for contravention of a Quality Control Order (QCO) under the BIS Act 2016?"*
- *"Which IS standard applies to Electric Instantaneous Water Heaters and Stationary Storage Type Electric Water Heaters?"*
- *"Which IS standard covers Safety Of Toys and Safety of Electric Toys?"*

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

---

## 🤝 Contributors

- **Udit Taneja**
- **Ved Singh**
