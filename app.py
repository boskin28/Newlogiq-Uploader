import streamlit as st
from pinecone import Pinecone
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
        if (
            st.session_state.get("username") in st.secrets.get("passwords", {})
            and hmac.compare_digest(
                st.session_state.get("password", ""),
                st.secrets.passwords.get(st.session_state.get("username", ""), ""),
            )
        ):
            st.session_state["password_correct"] = True
            del st.session_state["username"]
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True
    login_form()
    if "password_correct" in st.session_state:
        st.error("Username or password incorrect")
    return False

if not check_password():
    st.stop()

# --- Load secrets & init clients ---
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV = st.secrets['ENVIRONMENT']
index_name = st.secrets['INDEX_NAME']
host = st.secrets['HOST']
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=host)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize storage for vector IDs
if 'vector_ids' not in st.session_state:
    st.session_state['vector_ids'] = {}

# Helpers

def get_pdf_text(pdf):
    reader = PdfReader(pdf)
    return "".join(page.extract_text() or "" for page in reader.pages)


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vectorstore(text_chunks, pdf_name, namespace):
    """
    Batch upsert chunks and record their vector IDs under session_state.vector_ids[pdf_name]
    """
    batch_size = 200
    all_ids = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i : i + batch_size]
        texts = [f"{pdf_name}: {c}" for c in batch]
        metadatas = [{"filename": pdf_name} for _ in texts]
        embs = embeddings.embed_documents(texts)
        vectors = []
        for emb, meta in zip(embs, metadatas):
            vid = str(uuid.uuid4())
            vectors.append((vid, emb, meta))
            all_ids.append(vid)
        index.upsert(vectors=vectors, namespace=namespace)
    # store ids list
    st.session_state['vector_ids'][pdf_name] = all_ids
    # return a PineconeVectorStore for querying
    return PineconeVectorStore.from_existing_index(
        embedding=embeddings, index_name=index_name, namespace=namespace
    )

# Main UI

st.set_page_config(page_title="Upload & Index PDFs", page_icon=":outbox_tray:")
st.title("Upload & Index PDFs 📤")

namespace = st.text_input("Vector DB Namespace", value="default")
pdf_docs = st.file_uploader(
    "Upload your PDFs and click 'Process'", accept_multiple_files=True, type='pdf'
)

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
