import streamlit as st
from langchain.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import uuid
import hmac

# Authentication (unchanged)
def check_password():
    def login_form():
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)
    def password_entered():
        # ...
        pass
    if 'authenticated' not in st.session_state:
        login_form()
    elif not st.session_state['authenticated']:
        login_form()

# Vector store helper
def get_vectorstore(chunks, index_name, namespace):
    embeddings = OpenAIEmbeddings()
    all_ids = []
    for chunk in chunks:
        id = str(uuid.uuid4())
        vector = embeddings.embed(chunk)
        # upsert vectors into Pinecone index
        Pinecone.upsert(index_name=index_name, namespace=namespace, vectors=[(id, vector, {'text': chunk})])
        all_ids.append(id)
    st.session_state['vector_ids'][index_name] = all_ids
    # return a Pinecone-based LangChain VectorStore for querying
    return Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )

# PDF processing
def get_pdf_text(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Main UI
st.title("PDF Uploader & Vector Indexer")
namespace = st.text_input("Namespace", value="default")
pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if 'vector_ids' not in st.session_state:
    st.session_state['vector_ids'] = {}

if st.button("Process") and pdf_docs:
    with st.spinner("Indexing documents..."):
        for pdf in pdf_docs:
            text = get_pdf_text(pdf)
            chunks = get_text_chunks(text)
            get_vectorstore(chunks, pdf.name, namespace)
    st.success("All documents indexed.")

# Display indexed filenames and counts
if st.session_state['vector_ids']:
    st.subheader("Indexed Documents")
    for fname, ids in st.session_state['vector_ids'].items():
        st.write(f"- {fname} ({len(ids)} chunks indexed)")
