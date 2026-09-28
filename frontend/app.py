"""
Streamlit frontend for the RAG chatbot.

WHY A SEPARATE PROCESS FROM THE FASTAPI BACKEND (rather than importing
app/services/ directly)?
Keeping the frontend as a pure HTTP client of the API is what makes the
backend genuinely reusable - the exact same FastAPI app could later serve
a mobile app, a Slack bot, or a different frontend framework, with zero
changes. Streamlit only ever talks over the network, exactly like any
other API consumer would.

Run with (backend must already be running separately, e.g. `python main.py`):
    streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st

# The backend URL is configurable via an env var so this can point at a
# deployed API later without editing code - defaults to the local dev server.
API_BASE_URL = os.environ.get("RAG_API_BASE_URL", "http://localhost:8000/api")

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")


# --- Helpers ---------------------------------------------------------

def get_health() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def ask_question(query: str, top_k: int, source_filter: str | None) -> dict:
    payload = {"query": query, "top_k": top_k}
    if source_filter and source_filter != "All documents":
        payload["source_filter"] = source_filter
    resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def upload_file(uploaded_file) -> dict:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    resp = requests.post(f"{API_BASE_URL}/ingest/upload", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


def ingest_sample_docs() -> dict:
    resp = requests.post(f"{API_BASE_URL}/ingest/sample-docs", timeout=180)
    resp.raise_for_status()
    return resp.json()


# --- Sidebar: connection status + document management ---------------------------------------------------------

with st.sidebar:
    st.header("📚 Knowledge Base")

    health = get_health()
    if health is None:
        st.error(
            f"Can't reach the API at {API_BASE_URL}. "
            "Make sure the backend is running (`python main.py`)."
        )
    else:
        st.success("Backend connected")
        st.metric("Chunks stored", health["chunk_count"])
        st.caption(f"Collection: `{health['collection_name']}`")
        st.caption(f"Embedding model: `{health['embedding_model']}`")
        st.caption(f"LLM: `{health['llm_model']}`")

    st.divider()

    st.subheader("Seed with sample docs")
    st.caption("Ingest every file already in data/sample_docs.")
    if st.button("Ingest sample docs", use_container_width=True):
        with st.spinner("Ingesting sample documents..."):
            try:
                result = ingest_sample_docs()
                st.success(f"Ingested {result['total_files']} file(s), {result['total_chunks']} chunk(s) total.")
                for r in result["results"]:
                    icon = "✅" if r["status"] == "success" else "⚠️"
                    st.caption(f"{icon} {r['filename']}: {r['chunks_created']} chunks")
            except requests.RequestException as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()

    st.subheader("Upload a document")
    uploaded_file = st.file_uploader("Choose a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None and st.button("Upload & ingest", use_container_width=True):
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                result = upload_file(uploaded_file)
                st.success(f"Ingested {result['filename']}: {result['chunks_created']} chunks created.")
            except requests.RequestException as e:
                st.error(f"Upload failed: {e}")


# --- Main panel: chat interface ---------------------------------------------------------

st.title("💬 RAG Knowledge Assistant")
st.caption("Ask a question about the ingested documents. Answers are grounded only in retrieved context.")

# Advanced options, tucked away so the main flow stays simple by default
with st.expander("Advanced options"):
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=4)
    source_filter = st.text_input(
        "Restrict to a specific source filename (optional)",
        placeholder="e.g. employee_handbook.txt",
    )

# Chat history lives in Streamlit's session state, so it persists across
# reruns within one browser session (Streamlit reruns the whole script on
# every interaction) but resets if the page is reloaded.
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} source(s)"):
                for src in message["sources"]:
                    page_info = f", page {src['page']}" if src.get("page") else ""
                    st.markdown(f"**{src['source']}**{page_info} (distance: {src['distance']:.3f})")
                    st.text(src["text"][:500] + ("..." if len(src["text"]) > 500 else ""))

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_question(prompt, top_k=top_k, source_filter=source_filter or None)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander(f"📎 {len(result['sources'])} source(s)"):
                        for src in result["sources"]:
                            page_info = f", page {src['page']}" if src.get("page") else ""
                            st.markdown(f"**{src['source']}**{page_info} (distance: {src['distance']:.3f})")
                            st.text(src["text"][:500] + ("..." if len(src["text"]) > 500 else ""))
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                )
            except requests.RequestException as e:
                error_msg = f"Something went wrong talking to the backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
