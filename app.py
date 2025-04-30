import os
import streamlit as st
from pinecone import Pinecone as PineconeSDK, ServerlessSpec
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import uuid

# Authentication
def check_password():
    def login_form():
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)
    def password_entered():
        # implement real auth logic here
        st.session_state['authenticated'] = True
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        login_form()

# Initialize Pinecone SDK client
api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
environment = st.secrets.get("PINECONE_ENV") or os.getenv("PINECONE_ENV")
if not api_key or not environment:
    st.error("Pinecone API key and environment must be set in Streamlit secrets or env variables.")
    st.stop()
pc = PineconeSDK(api_key=api_key, environment=environment)

# PDF processing
def get_pdf_text(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Main UI
check_password()
st.title("PDF Uploader & Vector Indexer")

# Sidebar controls
st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("Chunk size", min_value=500, max_value=5000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap", min_value=0, max_value=500, value=100, step=50)

namespace = st.text_input("Namespace", value="default")
pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if 'vector_ids' not in st.session_state:
    st.session_state['vector_ids'] = {}

col1, col2 = st.columns(2)
with col1:
    process = st.button("Process")
with col2:
    clear = st.button("Clear history")

if clear:
    st.session_state['vector_ids'].clear()

if process and pdf_docs:
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # Ensure index exists
    index_name = namespace
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
    index = pc.Index(index_name)

    with st.spinner("Indexing documents..."):
        for pdf in pdf_docs:
            text = get_pdf_text(pdf)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(text)
            # embed and upsert in batch
            vectors = embeddings.embed_documents(chunks)
            ids = []
            for chunk, vector in zip(chunks, vectors):
                vid = str(uuid.uuid4())
                index.upsert(vectors=[(vid, vector, {'text': chunk})], namespace=namespace)
                ids.append(vid)
            st.session_state['vector_ids'][pdf.name] = ids
    st.success("All documents indexed.")

# Display indexed filenames and counts
if st.session_state['vector_ids']:
    st.subheader("Indexed Documents")
    for fname, ids in st.session_state['vector_ids'].items():
        st.write(f"- {fname} ({len(ids)} chunks indexed)")

# Example query section
if st.session_state['vector_ids'] and st.button("Test Query"):
    # show how to query using LangChain wrapper
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vstore = LangChainPinecone.from_existing_index(
        embedding=embeddings,
        index_name=namespace,
        namespace=namespace
    )
    query = st.text_input("Enter a query to test similarity search:")
    if query:
        results = vstore.similarity_search(query)
        for res in results:
            st.write(res.page_content)

