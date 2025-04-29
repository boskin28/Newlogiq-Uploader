import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_pinecone import PineconeVectorStore
import hmac

# Authentication

def check_password():
    def login_form():
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and hmac.compare_digest(
                st.session_state["password"],
                st.secrets.passwords[st.session_state["username"]],
            )
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
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

# Secrets
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV = st.secrets['ENVIRONMENT']
index_name = st.secrets['INDEX_NAME']
index_host = st.secrets['HOST']

# Initialize Pinecone client and embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=index_host)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Helpers

def get_pdf_text(pdf):
    text = ""
    reader = PdfReader(pdf)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vectorstore(text_chunks, pdf_name, namespace):
    texts = [f"{pdf_name}: {chunk}" for chunk in text_chunks]
    metadatas = [{"filename": pdf_name} for _ in texts]

    # final: use from_texts to avoid init-arg mismatch
    vectorstore = PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        index_name=index_name,
        namespace=namespace,
    )
    return vectorstore


def main():
    st.set_page_config(page_title="Upload Files", page_icon=":outbox_tray:")
    st.header("Upload Files :outbox_tray:")

    namespace = st.text_input("Enter the Vector Database Namespace", value="ME")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type='pdf'
    )

    if st.button("Process"):
        with st.spinner("Processing"):
            for pdf in pdf_docs:
                raw_text = get_pdf_text(pdf)
                chunks = get_text_chunks(raw_text)
                get_vectorstore(chunks, pdf.name, namespace)
        st.success('Upload complete.')

if __name__ == '__main__':
    main()
