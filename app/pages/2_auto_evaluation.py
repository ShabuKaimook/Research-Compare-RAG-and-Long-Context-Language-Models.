import streamlit as st
import requests
import pandas as pd
import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://backend:8000")

st.set_page_config(
    page_title="Automated Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Automated Evaluation: RAG vs Long Context")

st.caption(
    "Run automatic evaluation and compare quality, hallucination, and latency."
)

# ----------------------------
# Run evaluation
# ----------------------------
if st.button("🚀 Run Automated Evaluation"):
    with st.spinner("Running evaluation..."):
        res = requests.post(f"{FASTAPI_URL}/evaluate/run")
        if res.ok:
            st.success("Evaluation completed")
        else:
            st.error("Evaluation failed")

st.divider()

# ----------------------------
# Load results
# ----------------------------
if st.button("📥 Load Results"):
    res = requests.get(f"{FASTAPI_URL}/evaluate/results")

    if not res.ok:
        st.error("No results found")
    else:
        data = res.json()
        df = pd.DataFrame(data)

        st.subheader("📋 Average Scores")
        st.dataframe(df, use_container_width=True)

        st.divider()

        st.subheader("📈 Visual Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                f"{FASTAPI_URL}/evaluate/plot/latency",
                caption="Latency (seconds)",
                use_column_width=True,
            )

        with col2:
            st.image(
                f"{FASTAPI_URL}/evaluate/plot/quality",
                caption="Answer Quality",
                use_column_width=True,
            )

        with col3:
            st.image(
                f"{FASTAPI_URL}/evaluate/plot/hallucination",
                caption="Hallucination Rate",
                use_column_width=True,
            )
