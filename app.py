import os
import streamlit as st
from pinecone import Pinecone as PineconeSDK, ServerlessSpec
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import uuid
from itertools import islice

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

# Utility to batch an iterable into chunks of size n
def batch_iterable(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

# Main UI
check_password()
st.title("PDF Uploader & Vector Indexer")

# Load credentials from secrets
try:
    oai_key = st.secrets["OPENAI_API_KEY"]
    pc_api_key = st.secrets["PINECONE_API_KEY"]
    pc_env = st.secrets["ENVIRONMENT"]
    def_index = st.secrets.get("INDEX_NAME", "default")
except KeyError as e:
    st.error(f"Missing required secret: {e.args[0]}")
    st.stop()

# Initialize Pinecone SDK client
pc = PineconeSDK(api_key=pc_api_key, environment=pc_env)

# Sidebar settings
st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("Chunk size", 500, 5000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, 100, 50)
batch_size = st.sidebar.number_input("Upsert batch size", min_value=10, max_value=500, value=100, step=10)

# Upload and index controls
namespace = st.text_input("Namespace / Index Name", value=def_index)
pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if 'vector_ids' not in st.session_state:
    st.session_state['vector_ids'] = {}

col1, col2 = st.columns(2)
with col1:
    process = st.button("Process and Index")
with col2:
    clear = st.button("Clear History")

if clear:
    st.session_state['vector_ids'].clear()

# Process PDFs and index
if process and pdf_docs:
    # sanitize index/namespace name
    raw_ns = namespace or def_index
    # enforce lowercase alphanumeric or hyphens
    import re
    idx = raw_ns.lower()
    idx = re.sub(r"[^a-z0-9-]", "-", idx)
    
    # set OpenAI API key
    os.environ["OPENAI_API_KEY"] = oai_key
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # ensure index exists
    existing = pc.list_indexes().names()
    if idx not in existing:
        pc.create_index(
            name=idx,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region=pc_env)
        )
    index = pc.Index(idx)

    with st.spinner("Indexing documents in batches..."):
        for pdf in pdf_docs:
            # extract full text
            text = "".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
            # chunk text
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_text(text)
            # embed chunks
            vectors = embeddings.embed_documents(chunks)
            # prepare entries with filename metadata
            entries = [
                (str(uuid.uuid4()), vec, {'text': chunk, 'document': pdf.name})
                for chunk, vec in zip(chunks, vectors)
            ]
            # batch upsert
            for batch in batch_iterable(entries, batch_size):
                index.upsert(vectors=batch, namespace=idx)
            # record chunk IDs per file
            st.session_state['vector_ids'][pdf.name] = [vid for vid, _, _ in entries]
    st.success("All documents indexed.")
if process and pdf_docs:
    # set OpenAI API key
    os.environ["OPENAI_API_KEY"] = oai_key
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # ensure index exists
    idx = namespace
    existing = pc.list_indexes().names()
    if idx not in existing:
        pc.create_index(
            name=idx,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region=pc_env)
        )
    index = pc.Index(idx)

    with st.spinner("Indexing documents in batches..."):
        for pdf in pdf_docs:
            # extract full text
            text = "".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
            # chunk text
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_text(text)
            # embed chunks
            vectors = embeddings.embed_documents(chunks)
            # prepare entries with filename metadata
            entries = [
                (str(uuid.uuid4()), vec, {'text': chunk, 'document': pdf.name})
                for chunk, vec in zip(chunks, vectors)
            ]
            # batch upsert
            for batch in batch_iterable(entries, batch_size):
                index.upsert(vectors=batch, namespace=namespace)
            # record chunk IDs per file
            st.session_state['vector_ids'][pdf.name] = [vid for vid, _, _ in entries]
    st.success("All documents indexed.")

# Display indexed documents
if st.session_state['vector_ids']:
    st.subheader("Indexed Documents & Chunk Counts")
    for fname, ids in st.session_state['vector_ids'].items():
        st.write(f"- {fname}: {len(ids)} chunks (batched upsert)")

# Similarity search example with document metadata
if st.session_state['vector_ids']:
    st.subheader("Test Similarity Search")
    query = st.text_input("Enter a query to test similarity search:")
    if query:
        os.environ["OPENAI_API_KEY"] = oai_key
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vstore = LangChainPinecone.from_existing_index(
            embedding=embeddings,
            index_name=namespace,
            namespace=namespace
        )
        # include metadata in results
        results = vstore.similarity_search_with_score(query)
        for doc, score in results:
            st.write(f"**Document:** {doc.metadata.get('document')}  ")
            st.write(f"**Score:** {score}  ")
            st.write(doc.page_content)
