import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="Ask My CV",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Ask My CV")
st.caption("Compare your CV against a Job Description")

# ── SESSION STATE ──────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ── BUILD VECTORSTORE ONCE ─────────────────────────────
def build_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    cv_docs = TextLoader("cv.txt").load()
    jd_docs = TextLoader("job.txt").load()

    for doc in cv_docs:
        doc.metadata["source"] = "CV"
    for doc in jd_docs:
        doc.metadata["source"] = "Job Description"

    all_docs = cv_docs + jd_docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="ask_my_cv"
        # No persist_directory — runs in memory only, no file conflicts
    )

    return vectorstore

# ── LOAD ON STARTUP ────────────────────────────────────
if st.session_state.vectorstore is None:
    with st.spinner("📄 Loading CV and Job Description..."):
        st.session_state.vectorstore = build_vectorstore()

st.success("✅ Documents loaded — ask me anything!")

# ── ASK QUESTION ───────────────────────────────────────
def ask_question(question: str):
    vectorstore = st.session_state.vectorstore
    
    # Retrieve from BOTH sources separately
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


# ── SUGGESTED QUESTIONS ────────────────────────────────
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

# ── CHAT INPUT ─────────────────────────────────────────
st.markdown("### 💬 Chat")
user_input = st.chat_input("Ask anything about your fit for this role...")
if user_input:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

# ── PROCESS LAST QUESTION ──────────────────────────────
if st.session_state.chat_history:
    last = st.session_state.chat_history[-1]
    if last["role"] == "user":
        with st.spinner("🤔 Analyzing..."):
            answer = ask_question(last["content"])
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

# ── DISPLAY CHAT ───────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

st.divider()
st.caption("Built with LangChain · ChromaDB · Groq · Streamlit")