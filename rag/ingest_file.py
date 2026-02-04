from .load_and_chunk import load_and_chunk
from .embedding import embed_docs
from .store_embedding import store_embeddings
from .pdf_to_html import pdf_to_html
from pathlib import Path
from langchain_community.document_loaders import TextLoader


def ingest_file(file_path, db):
    if file_path.lower().endswith(".pdf"):
        html_path = pdf_to_html(file_path)
        print(f"Converted PDF to HTML: {html_path}")
        file_path = html_path
    elif file_path.lower().endswith(".txt"):
        file = TextLoader(file_path).load()
        file_path = "./upload/" + file_path.split("/")[-1]
        path = Path(file_path).with_suffix(".txt")
        path.write_text(file[0].page_content, encoding="utf-8")

    chunks = load_and_chunk(file_path)

    print(f"Loaded and chunked {len(chunks)} chunks from {file_path}")

    embeddings = embed_docs(chunks)
    print(f"Generated embeddings for {len(embeddings)} chunks")

    store_embeddings(chunks, embeddings, db)
    print(f"Stored embeddings in vector DB for {file_path}")
