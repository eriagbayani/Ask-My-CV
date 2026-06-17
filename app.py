import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import pypdf
import io

load_dotenv()

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="Ask My CV",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Ask My CV")
st.caption("Upload any two documents and ask questions about them")

# ── SESSION STATE ──────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "ready" not in st.session_state:
    st.session_state.ready = False

# ── FILE READER ────────────────────────────────────────
def read_file(uploaded_file) -> str:
    if uploaded_file.type == "application/pdf":
        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        return uploaded_file.read().decode("utf-8")

# ── BUILD VECTORSTORE ──────────────────────────────────
def build_vectorstore(cv_text: str, jd_text: str):
    cv_doc = Document(page_content=cv_text, metadata={"source": "CV"})
    jd_doc = Document(page_content=jd_text, metadata={"source": "Job Description"})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents([cv_doc, jd_doc])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="ask_my_cv"
    )

    return vectorstore

# ── ASK QUESTION ───────────────────────────────────────
def ask_question(question: str):
    vectorstore = st.session_state.vectorstore
    all_docs = vectorstore.get()

    cv_chunks = [doc for doc, meta in zip(
        all_docs['documents'], all_docs['metadatas']
    ) if meta.get('source') == 'CV']

    jd_chunks = [doc for doc, meta in zip(
        all_docs['documents'], all_docs['metadatas']
    ) if meta.get('source') == 'Job Description']

    cv_context = "\n\n".join(cv_chunks[:4])
    jd_context = "\n\n".join(jd_chunks[:4])

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""You are a career advisor evaluating job fit.
You have the candidate's CV and a Job Description below.
Answer the question using ONLY these documents.
Be specific — reference the candidate's actual skills and experience from the CV.

CV:
{cv_context}

JOB DESCRIPTION:
{jd_context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content

# ── UPLOAD UI (shows when not ready) ──────────────────
if not st.session_state.ready:
    st.markdown("### 📁 Upload your documents")

    col1, col2 = st.columns(2)
    with col1:
        cv_file = st.file_uploader(
            "Upload your CV",
            type=["txt", "pdf"],
            key="cv_upload"
        )
    with col2:
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=["txt", "pdf"],
            key="jd_upload"
        )

    if cv_file and jd_file:
        if st.button("🚀 Start Analysis"):
            with st.spinner("📄 Reading documents..."):
                cv_text = read_file(cv_file)
                jd_text = read_file(jd_file)

            with st.spinner("🧮 Building knowledge base..."):
                st.session_state.vectorstore = build_vectorstore(
                    cv_text, jd_text
                )
                st.session_state.ready = True

            st.rerun()
    else:
        st.info("👆 Upload both files to get started")

# ── CHAT UI (shows when ready) ─────────────────────────
if st.session_state.ready:
    st.success("✅ Documents analyzed — ask me anything!")

    if st.button("🔄 Upload new documents"):
        st.session_state.ready = False
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.rerun()

    # Suggested questions
    st.markdown("### 💡 Suggested questions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Am I a good fit?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Am I a good fit for this role based on my CV?"
            })
    with col2:
        if st.button("What should I highlight?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What should I highlight in my application?"
            })

    col3, col4 = st.columns(2)
    with col3:
        if st.button("What skills am I missing?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What skills am I missing for this role?"
            })
    with col4:
        if st.button("What are my strengths?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What are my key strengths for this role?"
            })

    # Chat input
    st.markdown("### 💬 Chat")
    user_input = st.chat_input("Ask anything about your fit for this role...")
    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

    # Process last question
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        if last["role"] == "user":
            with st.spinner("🤔 Analyzing..."):
                answer = ask_question(last["content"])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# ── FOOTER ─────────────────────────────────────────────
st.divider()
st.caption("Built with LangChain · ChromaDB · Groq · Streamlit")