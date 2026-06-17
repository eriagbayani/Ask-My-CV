# Ask My CV 🎯

A RAG-powered career advisor that compares your CV against any job description and answers questions about your fit — instantly.

🔗 **[Live Demo](https://ask-my-cv-rainerioagbayani.streamlit.app/)**

## What It Does

Upload your CV and a Job Description (PDF or TXT). The app uses Retrieval Augmented Generation (RAG) to answer questions based only on your actual documents — not generic advice.

Ask it:
- **"Am I a good fit for this role?"**
- **"What should I highlight in my application?"**
- **"What skills am I missing?"**
- **"What are my key strengths?"**
- Or any custom question about your fit

## How It Works

![](/ask_my_cv_architecture.png)

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core application logic |
| LangChain | Document loading, chunking, retrieval pipeline |
| ChromaDB | In-memory vector database |
| HuggingFace | Embedding model (all-MiniLM-L6-v2) |
| Groq API | LLM inference (LLaMA 3.3 70b) |
| Streamlit | Web UI and deployment |
| pypdf | PDF text extraction |
| python-dotenv | Environment variable management |

## Setup — Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/ask-my-cv
cd ask-my-cv
```

**2. Install dependencies**

With uv (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key_here
```

**4. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Setup — Deploy on Streamlit Cloud

1. Fork this repo
2. Go to **share.streamlit.io**
3. Click **Create app** → select this repo
4. Set main file to `app.py`
5. Under **Advanced settings → Secrets** add:
```
GROQ_API_KEY="your_groq_key_here"
```
6. Click **Deploy**

## Usage

1. Upload your CV — PDF or TXT
2. Upload the Job Description — PDF or TXT
3. Click **Start Analysis**
4. Use suggested questions or type your own in the chat

Click **Upload new documents** anytime to analyze a different role.

## Key Design Decisions

- **In-memory ChromaDB** — no file persistence means no permission conflicts and instant rebuild on every upload. Appropriate for a single-user career tool.
- **Separate CV and JD retrieval** — fetches chunks from both documents independently before passing to the LLM, ensuring the CV is always represented in the context regardless of query phrasing.
- **PDF + TXT support** — covers both common CV formats without requiring conversion.
- **Groq over OpenAI** — faster inference, free tier, no cost during development and demos.
- **Streamlit** — fastest path to a deployed, shareable UI for a Python-based AI app.

## What is RAG?

RAG (Retrieval Augmented Generation) gives the LLM access to your documents at query time without retraining it. Think of it like an open-book exam — the AI doesn't memorize your CV, it gets handed the relevant pages right before answering. This means answers are grounded in your actual content, not generic advice.

## Skills Demonstrated

`Python` `LangChain` `RAG` `ChromaDB` `Vector Embeddings` `Groq API` `Prompt Engineering` `Streamlit` `HuggingFace` `pypdf`