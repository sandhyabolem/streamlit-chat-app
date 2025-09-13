import streamlit as st
import fitz  # PyMuPDF
import tempfile
import chromadb
from chromadb.config import Settings
import openai

# ✅ Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ Initialize Chroma DB
chroma_client = chromadb.Client(Settings(persist_directory="chroma_store"))
collection = chroma_client.get_or_create_collection(name="pdf_docs")

# --------- Helper Functions ---------
def extract_text_from_pdf(file):
    text = ""
    pdf_doc = fitz.open(file)
    for page in pdf_doc:
        text += page.get_text()
    return text

def add_to_chroma(text, filename):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

def query_chroma(query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    if results and "documents" in results and results["documents"]:
        return results["documents"][0]
    return []

def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""
    You are a helpful assistant. Answer the question using ONLY the following context:

    {context}

    Question: {query}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --------- Streamlit UI ---------
st.set_page_config(page_title="📘 PDF Q&A Bot", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stTextInput input {
        background-color: #2d2d2d;
        color: white;
    }
    .stButton button {
        background-color: #3c3c3c;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📘 PDF Q&A Chatbot")

# Layout with 2 columns
left_col, right_col = st.columns([1, 2])

# --------- LEFT PANEL ---------
with left_col:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("📥 Processing PDF..."):
            pdf_text = extract_text_from_pdf(tmp_path)
            add_to_chroma(pdf_text, uploaded_file.name)

        st.success(f"✅ {uploaded_file.name} processed!")

# --------- RIGHT PANEL ---------
with right_col:
    st.header("💬 Chat with PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Type your question here...")

    if st.button("Ask"):
        if query:
            with st.spinner("🤔 Thinking..."):
                context = query_chroma(query)
                answer = generate_answer(query, context)

                # Save chat history
                st.session_state.chat_history.append({"user": query, "bot": answer})

    # Display chat history in nice format
    for chat in st.session_state.chat_history:
        st.chat_message("user").write(chat["user"])
        st.chat_message("assistant").write(chat["bot"])
