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
    page_title="RAG Chat",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Chat with Your Documents")

# -------------------------
# Session state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Render history
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat input
# -------------------------
question = st.chat_input("Ask a question about your documents...")

if question:
    # ---- User message ----
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # ---- Assistant ----
    with st.chat_message("assistant"):
        status = st.empty()
        answer_box = st.empty()
        meta_box = st.empty()

        full_answer = ""
        meta_data = None
        buffer = ""              # 🔴 CHANGED: buffer สำหรับรวม stream

        start_time = time.time()
        first_token_time = None

        status.markdown("🔍 **Searching documents…**")

        try:
            with requests.post(
                f"{FASTAPI_URL}/chat/stream",
                json={"question": question},
                stream=True,
                timeout=300,
            ) as r:

                r.raise_for_status()
                status.empty()
                status.markdown("🧠 **Thinking…**")

                for chunk in r.iter_content(chunk_size=1024):
                    if not chunk:
                        continue

                    text = chunk.decode("utf-8", errors="ignore")
                    buffer += text   # 🔴 CHANGED

                    # st.write("RAW CHUNK >>>", repr(text))

                    # ---- META BLOCK (ROBUST) ----
                    if "[[META]]" in buffer:
                        content, meta_part = buffer.split("[[META]]", 1)

                        # render content ก่อน META
                        if content:
                            full_answer += content
                            answer_box.markdown(full_answer)

                        # parse META
                        meta_data = json.loads(meta_part)
                        # print(meta_data)
                        break  # 🔴 IMPORTANT: stop stream here

                    # ---- First token ----
                    if first_token_time is None:
                        first_token_time = time.time()
                        status.empty()

                    full_answer += text
                    answer_box.markdown(full_answer + "▌")

            # ---- Done ----
            status.empty()
            answer_box.markdown(full_answer)

            # -------------------------
            # Metadata
            # -------------------------
            st.write("Lateny, Tokens, Sources:", meta_data)

            # ---- Save history ----
            st.session_state.messages.append(
                {"role": "assistant", "content": full_answer}
            )

        except Exception as e:
            status.empty()
            st.error(f"❌ Error: {e}")