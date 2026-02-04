import streamlit as st
import requests
import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://0.0.0.0:8000")

st.set_page_config(
    page_title="RAG Chat App",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.set_page_config(page_title="Chat", page_icon="💬")

st.title("💬 Chat with Your Documents")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/chat", json={"question": prompt}
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "I don't know")
                    sources = data.get("sources", [])

                    # Display response
                    st.markdown(answer)

                    # Display sources if available
                    if sources:
                        st.divider()
                        st.subheader("📚 Sources")
                        for i, source in enumerate(sources, 1):
                            st.write(f"{i}. {source}")

                    # Add assistant message to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    error_msg = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error_msg}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
