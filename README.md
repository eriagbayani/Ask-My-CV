# Ask My CV 🎯

A RAG-powered career advisor that compares your CV against any job description and answers questions about your fit — instantly.

## What It Does

Upload your CV and a Job Description as text files. The app uses Retrieval Augmented Generation (RAG) to answer questions based only on your actual documents — not generic advice.

Ask it:
- **"Am I a good fit for this role?"**
- **"What should I highlight in my application?"**
- **"What skills am I missing?"**
- **"What are my key strengths?"**
- Or any custom question about your fit

## Demo

![Ask My CV Demo](demo.png)

🔗 **[Live Demo](#)** ← replace with your Streamlit URL

## How It Works

```
cv.txt + job.txt
      ↓
   CHUNK
Split documents into 500-character chunks
      ↓
   EMBED
Convert chunks to vectors using HuggingFace
sentence-transformers (all-MiniLM-L6-v2)
      ↓
   STORE
Save vectors in ChromaDB (in-memory)
      ↓
   RETRIEVE
Fetch relevant chunks from both CV and JD
      ↓
   ANSWER
Groq LLM answers using only retrieved context
```

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core application logic |
| LangChain | Document loading, chunking, retrieval pipeline |
| ChromaDB | In-memory vector database |
| HuggingFace | Embedding model (all-MiniLM-L6-v2) |
| Groq API | LLM inference (LLaMA 3.3 70b) |
| Streamlit | Web UI and deployment |
| python-dotenv | Environment variable management |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/ask-my-cv
cd ask-my-cv
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**
```bash
cp .env.example .env
```
Fill in your Groq API key in `.env`

**4. Add your documents**

Create two text files in the project root:
- `cv.txt` — paste your CV content
- `job.txt` — paste the job description

**5. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Switching CVs or Job Descriptions

Just replace the content in `cv.txt` and `job.txt` and restart the app. The vectorstore rebuilds automatically on every run — no manual re-ingestion needed.

## Key Design Decisions

- **In-memory ChromaDB** — no file persistence means no permission conflicts and instant rebuild on every run. Appropriate for a single-user career tool.
- **Separate CV and JD retrieval** — fetches chunks from both documents independently before passing to the LLM, ensuring the CV is always represented in the context regardless of query phrasing.
- **Groq over OpenAI** — faster inference, free tier, no cost during development and demos.
- **Streamlit** — fastest path to a deployed, shareable UI for a Python-based AI app.

## What is RAG?

RAG (Retrieval Augmented Generation) gives the LLM access to your documents at query time without retraining it. Think of it like an open-book exam — the AI doesn't memorize your CV, it gets handed the relevant pages right before answering. This means answers are grounded in your actual content, not generic advice.

## Skills Demonstrated

`Python` `LangChain` `RAG` `ChromaDB` `Vector Embeddings` `Groq API` `Prompt Engineering` `Streamlit` `HuggingFace`