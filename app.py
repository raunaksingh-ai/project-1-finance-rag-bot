import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="Finance Document Q&A", page_icon="📊", layout="wide")

st.title("Finance Document Q&A Bot")
st.caption("Ask questions about your financial documents — powered by RAG")


@st.cache_resource
def get_engine() -> RAGEngine:
    engine = RAGEngine()
    engine.load_existing()
    return engine


engine = get_engine()

with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload financial documents",
        type=["pdf", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Documents"):
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for uf in uploaded_files:
                path = os.path.join(tmpdir, uf.name)
                with open(path, "wb") as f:
                    f.write(uf.getbuffer())
            count = engine.ingest(tmpdir)
            st.success(f"Ingested {count} chunks from {len(uploaded_files)} file(s)")
            st.cache_resource.clear()

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload PDF/CSV financial docs")
    st.markdown("2. Click 'Ingest Documents'")
    st.markdown("3. Ask questions in the chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your financial documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = engine.query(prompt)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("Sources"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {i}:** {src['source']} (Page {src['page']})")
                    st.markdown(f"> {src['content']}...")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
