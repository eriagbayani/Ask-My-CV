import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

def load_chatbot(collection_name: str = "documents"):
    print("🔄 Loading vector database...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    return retriever, llm

def ask_question(question: str, retriever, llm):
    # 1. RETRIEVE relevant chunks
    docs = retriever.invoke(question)

    # 2. Build context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Build prompt with context
    prompt = f"""Answer the question based only on the following context.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

    # 4. ASK LLM
    response = llm.invoke(prompt)

    return response.content, docs

if __name__ == "__main__":
    print("🤖 RAG Chatbot ready! Ask questions about the document.")
    print("Type 'exit' to quit.\n")

    retriever, llm = load_chatbot()

    while True:
        question = input("\n❓ Your question: ")
        if question.lower() == "exit":
            break

        answer, sources = ask_question(question, retriever, llm)

        print(f"\n💬 Answer: {answer}")
        print(f"\n📚 Sources used: {len(sources)} chunks")