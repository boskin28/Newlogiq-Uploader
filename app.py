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

# Main UI
check_password()
st.title("PDF Uploader & Vector Indexer")

# Load Pinecone and OpenAI credentials from Streamlit secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_env = st.secrets["ENVIRONMENT"]
    default_index = st.secrets.get("INDEX_NAME", "default")
except KeyError as e:
    st.error(f"Missing required secret: {e.args[0]}")
    st.stop()

# Initialize Pinecone SDK client (control-plane only)
pc = PineconeSDK(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

# Sidebar: chunking settings
st.sidebar.header("Text Chunking Settings")
chunk_size = st.sidebar.slider("Chunk size", min_value=500, max_value=5000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap", min_value=0, max_value=500, value=100, step=50)

# Upload and index configurations
namespace = st.text_input("Namespace / Index Name", value=default_index)
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
    # Set OpenAI API key for embeddings
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Ensure index exists
    index_name = namespace
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region=pinecone_env)
        )
    # Connect to index (will fetch data-plane host automatically)
    index = pc.Index(index_name)

    with st.spinner("Indexing documents..."):
        for pdf in pdf_docs:
            text = "".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_text(text)
            vectors = embeddings.embed_documents(chunks)
            ids = []
            for chunk, vector in zip(chunks, vectors):
                vid = str(uuid.uuid4())
                index.upsert(vectors=[(vid, vector, {'text': chunk})], namespace=namespace)
                ids.append(vid)
            st.session_state['vector_ids'][pdf.name] = ids
    st.success("All documents indexed.")

# Display indexed documents
if st.session_state['vector_ids']:
    st.subheader("Indexed Documents & Chunk Counts")
    for fname, ids in st.session_state['vector_ids'].items():
        st.write(f"- {fname}: {len(ids)} chunks")

# Similarity search example
if st.session_state['vector_ids']:
    st.subheader("Test Similarity Search")
    query = st.text_input("Enter a query to test similarity search:")
    if query:
        # ensure embedder is initialized
        os.environ["OPENAI_API_KEY"] = openai_api_key
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vstore = LangChainPinecone.from_existing_index(
            embedding=embeddings,
            index_name=namespace,
            namespace=namespace
        )
        results = vstore.similarity_search(query)
        for res in results:
            st.write(res.page_content)
