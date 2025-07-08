# app/chatbot_logic.py

import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. SETUP MODELS ---
llm = ChatGroq(
    model='gemma2-9b-it',
    api_key=os.environ.get("GROQ_API_KEY") # Use environment variables for security
)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# --- 2. DEFINE PROMPT TEMPLATES ---
# Prompt for RAG: Instructs the model to use the context
RAG_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an expert assistant on power grids. Use the following context to answer the user's question.
    If the context does not contain the answer, say "Based on the provided documents, I cannot answer this question."
    Your answer should be clear and based only on the provided text.

    Context:
    {context}

    Question: {input}
    """
)

# Prompt for Fallback: A general conversational prompt
GENERAL_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the following question to the best of your ability.

    Question: {input}
    """
)

# --- 3. SETUP THE RAG PIPELINE ---
def setup_rag_pipeline():
    docs_path = os.path.join(os.path.dirname(__file__), 'docs')
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        print("⚠️  'docs' folder is empty or does not exist. RAG will be disabled.")
        return None

    loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    if not documents:
        print("⚠️  No PDF documents found. RAG will be disabled.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain=create_stuff_documents_chain(llm, RAG_PROMPT)
    )
    print("✅ RAG pipeline setup complete.")
    return retrieval_chain

# --- 4. ENHANCED ANSWER GENERATION LOGIC ---
async def generate_answer(rag_chain, question: str):
    # If RAG is not enabled, go straight to general knowledge
    if not rag_chain:
        print("-> RAG disabled. Using general knowledge.")
        general_chain = GENERAL_PROMPT | llm
        response = await general_chain.ainvoke({"input": question})
        return response.content

    # First, try to answer using the RAG chain
    print("-> Attempting to answer with RAG...")
    rag_response = await rag_chain.ainvoke({"input": question})
    rag_answer = rag_response.get("answer", "")

    # Check if the RAG answer was conclusive or a "I can't answer" fallback
    if "based on the provided documents" not in rag_answer.lower() and "cannot answer" not in rag_answer.lower():
        print("-> RAG provided a conclusive answer.")
        return rag_answer
    else:
        # If RAG failed, use the general knowledge chain
        print("-> RAG inconclusive. Falling back to general knowledge.")
        general_chain = GENERAL_PROMPT | llm
        response = await general_chain.ainvoke({"input": question})
        # Prepend a small note to the user
        clarification = "Based on my general knowledge (as this was not in the documents): "
        return clarification + response.content