import asyncio
from pathlib import Path
import time
import os
import requests
import re
import json

import streamlit as st
import inngest
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG Ingest PDF",
    page_icon="📄",
    layout="centered",
)

# ------------------------------------------------------------------
# Inngest client (USED ONLY FOR INGESTION)
# ------------------------------------------------------------------


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/inngest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


# ------------------------------------------------------------------
# UI: PDF INGESTION (ASYNC / BACKGROUND)
# ------------------------------------------------------------------

st.title("📄 Upload a PDF to Ingest")

uploaded = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        asyncio.run(send_rag_ingest_event(path))
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

# ------------------------------------------------------------------
# UI: QUERY (SYNC – FAST)
# ------------------------------------------------------------------

st.divider()
st.title("💬 Ask a question about your PDFs")


def normalize_latex(text: str) -> str:
    if not text:
        return text

    # แปลง \[ ... \] → $$ ... $$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # แปลง \( ... \) → $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

    return text


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1,
        max_value=20,
        value=3,  # ลด default จาก 5 → 3
        step=1,
    )
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        st.subheader("Answer")

        placeholder = st.empty()
        full_text = ""

        ui_start = time.perf_counter()
        ui_first_token = None

        with st.spinner("Streaming answer..."):
            with requests.post(
                f"{API_BASE}/rag/query/stream",
                json={"question": question, "top_k": int(top_k)},
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()

                # อ่าน sources + backend timing จาก header
                sources = json.loads(resp.headers.get("X-Sources", "[]"))

                # Server ใช้เวลากี่วินาทีก่อนส่ง token แรก
                backend_ttfb = None
                backend_total = None

                for chunk in resp.iter_content(chunk_size=None):
                    if not chunk:
                        continue

                    text = chunk.decode("utf-8")

                    if text.startswith("<<TTFB:"):
                        backend_ttfb = float(
                            text.replace("<<TTFB:", "").replace(">>", "")
                        )
                        continue

                    if text.startswith("<<TOTAL:"):
                        backend_total = float(
                            text.replace("<<TOTAL:", "").replace(">>", "")
                        )
                        continue

                    # token ปกติ
                    if ui_first_token is None:
                        ui_first_token = time.perf_counter()

                    full_text += text
                    placeholder.markdown(normalize_latex(full_text))

        ui_end = time.perf_counter()

        # Show timing
        st.caption("⏱ Timing")

		# TTFB = Time To First Byte (ผู้ใช้รอกี่วินาทีก่อน “เห็นตัวอักษรแรก”)
        st.write(f"- Frontend TTFB: {(ui_first_token - ui_start):.3f} sec")
        st.write(f"- Frontend Total: {(ui_end - ui_start):.3f} sec")

        if backend_ttfb:
            st.write(f"- Backend TTFB: {backend_ttfb} sec")

        if backend_total:
            st.write(f"- Backend Total: {backend_total} sec")

        # Sources
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
