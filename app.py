import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, UnstructuredExcelLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# --- Sidebar ---
st.sidebar.title("⚙️ Configuration")

# Groq API Key Input
groq_api_key = st.sidebar.text_input("🔑 Enter your Groq API Key", type="password")

# Model selection dropdown
model_name = st.sidebar.selectbox("🤖 Choose a Groq Model", [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-qwen-32b",
    "gemma2-9b-it"
])

# Upload file
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload a File (PDF, CSV, Excel, JSON)",
    type=["pdf", "csv", "xlsx", "json"]
)

# Buttons for embedding and vector store
embed_button = st.sidebar.button("🔍 Generate Embeddings")

# Info section
st.sidebar.markdown("""
### ℹ️ About
This is a Retrieval-Augmented Generation (RAG) powered chatbot using **Groq LLMs** and **FAISS Vector Store**.

**Features:**
- Multi-format file support: PDF, CSV, Excel, JSON
- HuggingFace Embeddings
- Fast and accurate answers with Groq LLM
- Custom model & key selection
- Retrieved document chunks shown
""")

# --- Main Panel ---
#st.title("🧠 AI-Powered Document QA ChatBot")
st.markdown("<h1 style='text-align: center; font-size: 40px; white-space: nowrap;'>🧠 AI-Powered Document QA ChatBot</h1>", unsafe_allow_html=True)

#st.subheader("Ask Questions from Your Uploaded Document")

if uploaded_file and embed_button:
    with st.spinner("Processing your file..."):
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext == "csv":
            loader = CSVLoader(file_path)
        elif ext == "xlsx":
            loader = UnstructuredExcelLoader(file_path)
        elif ext == "json":
            loader = JSONLoader(file_path)
        else:
            st.error("Unsupported file format")
            st.stop()

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        st.session_state.vectors = vectorstore
        st.success("✅ Embeddings generated and stored in FAISS vector DB")

# Prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the question based on the context below. Be as accurate and specific as possible.
<context>
{context}
</context>
Question: {input}
""")

# Initialize LLM only if key is present
if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

    if "vectors" in st.session_state:
        query = st.text_input("🔍 Ask a question from the document")
        if query:
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})  # You can change top k here
            doc_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({"input": query})
            elapsed = time.process_time() - start

            st.write("🧠 **Answer:**", response["answer"])
            st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

            with st.expander("📄 Retrieved Chunks"):
                for doc in response.get("context", []):
                    st.write(doc.page_content)
                    st.write("---")
else:
    st.warning("Please enter your Groq API key to start querying.")
