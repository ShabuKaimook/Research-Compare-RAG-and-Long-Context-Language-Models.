import streamlit as st
import requests
import os
import time
import json

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://backend:8000")

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="RAG vs Long Context",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 RAG vs Long Context Comparison")

st.caption(
    "Ask one question. Compare **Retrieval-Augmented Generation** "
    "vs **Long-Context LLM** in real time."
)

if st.button("🔄 New question"):
    st.session_state.last_question = None
    st.rerun()

# -------------------------
# Session state (PREVENT DOUBLE RUN)
# -------------------------
if "last_question" not in st.session_state:
    st.session_state.last_question = None


# -------------------------
# Helper: stream reader
# -------------------------
def stream_answer(endpoint: str, question: str, answer_box, meta_box, label: str):
    buffer = ""
    full_answer = ""
    meta_data = None

    status = meta_box.empty()
    status.markdown("🧠 Thinking…")

    start = time.time()
    first_token_time = None

    with requests.post(
        f"{FASTAPI_URL}{endpoint}",
        json={"question": question},
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()

        for chunk in r.iter_content(chunk_size=1024):
            if not chunk:
                continue

            text = chunk.decode("utf-8", errors="ignore")
            buffer += text

            # ---- META DETECTION ----
            if "[[META]]" in buffer:
                content, meta_part = buffer.split("[[META]]", 1)
                full_answer += content
                answer_box.markdown(full_answer)
                meta_data = json.loads(meta_part)
                break

            if first_token_time is None:
                first_token_time = time.time()
                status.empty()

            full_answer += text
            answer_box.markdown(full_answer + "▌")

    answer_box.markdown(full_answer)
    status.empty()

    # ---- META DISPLAY (quiet) ----
    if meta_data:
        latency_first = meta_data.get("latency_first_token")
        latency_total = meta_data.get("latency_total")
        tokens = meta_data.get("tokens", {})
        sources = meta_data.get("sources", [])

        with meta_box:
            st.caption(
                f"⏱️ First token: **{latency_first:.2f}s** · "
                f"Total: **{latency_total:.2f}s**"
            )

            if tokens:
                with st.expander("🧮 Tokens", expanded=False):
                    st.caption(
                        f"Prompt: {tokens['prompt_tokens']} | "
                        f"Completion: {tokens['completion_tokens']} | "
                        f"Total: {tokens['total_tokens']} | "
                        f"Cost: ${tokens['cost_usd']:.6f}"
                    )

            if sources:
                with st.expander("📚 Sources", expanded=False):
                    for s in sources:
                        st.write("-", s)


# -------------------------
# Question input
# -------------------------
question = st.text_input(
    "💬 Ask a question",
    placeholder="e.g. Tell me about Google in Thai",
)

if question and question != st.session_state.last_question:
    st.session_state.last_question = question
    st.divider()

    col_rag, col_long = st.columns(2)

    # =========================
    # RAG COLUMN
    # =========================
    with col_rag:
        st.subheader("🔎 RAG")
        rag_answer = st.empty()
        rag_meta = st.container()

        stream_answer(
            endpoint="/chat/rag/stream",
            question=question,
            answer_box=rag_answer,
            meta_box=rag_meta,
            label="RAG",
        )

    # =========================
    # LONG CONTEXT COLUMN
    # =========================
    with col_long:
        st.subheader("🧠 Long Context")
        long_answer = st.empty()
        long_meta = st.container()

        stream_answer(
            endpoint="/chat/long/stream",
            question=question,
            answer_box=long_answer,
            meta_box=long_meta,
            label="LONG",
        )
