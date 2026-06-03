import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="SEBI/RBI Regulation Assistant",
    page_icon="📋",
    layout="wide"
)

st.title("📋 SEBI/RBI Regulation Assistant")
st.caption("Ask questions over SEBI and RBI regulatory documents. All answers are cited from source documents.")

with st.sidebar:
    st.header("About")
    st.write("This assistant answers questions using only official SEBI and RBI regulatory documents.")
    st.divider()
    st.write("**How it works:**")
    st.write("1. Hybrid search (BM25 + vector)")
    st.write("2. Cross-encoder reranking")
    st.write("3. LLM answering with citation enforcement")
    st.divider()
    top_k = st.slider("Number of source chunks", min_value=3, max_value=8, value=5)
    st.divider()
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.success(f"API online — {health['chunks_loaded']} chunks loaded")
    except Exception:
        st.error("API offline — start uvicorn first")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**[{i+1}] {source['source'] or source['chunk_id']}**")
                    st.caption(f"Rerank score: {source['rerank_score']} | URL: {source['url'] or 'N/A'}")
                    st.text(source['text_preview'])
                    st.divider()

if prompt := st.chat_input("Ask about SEBI or RBI regulations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "top_k": top_k}
                ).json()

                answer = response["answer"]
                sources = response["sources"]

                st.markdown(answer)
                with st.expander("View sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**[{i+1}] {source['source'] or source['chunk_id']}**")
                        st.caption(f"Rerank score: {source['rerank_score']} | URL: {source['url'] or 'N/A'}")
                        st.text(source['text_preview'])
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}. Make sure the API is running.")