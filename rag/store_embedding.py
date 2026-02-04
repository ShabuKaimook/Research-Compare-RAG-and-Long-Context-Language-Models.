import uuid

def store_embeddings(chunks: list, embeddings: list, vector_db):
    ids = [str(uuid.uuid4()) for _ in chunks]

    payloads = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", None),
        }
        for doc in chunks
    ]

    vector_db.upsert(ids, embeddings, payloads)