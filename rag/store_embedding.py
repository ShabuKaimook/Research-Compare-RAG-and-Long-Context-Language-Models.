import uuid

# def store_embeddings(chunks: list, embeddings: list, vector_db):
#     ids = [str(uuid.uuid4()) for _ in chunks]

#     payloads = [
#         {
#             "text": doc.page_content,
#             "source": doc.metadata.get("source", ""),
#             "page": doc.metadata.get("page", None),
#         }
#         for doc in chunks
#     ]

#     vector_db.upsert(ids, embeddings, payloads)

def store_embeddings(chunks: list, embeddings: list, vector_db, batch_size: int = 100):

    total = len(chunks)

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        batch_ids = [str(uuid.uuid4()) for _ in batch_chunks]

        batch_payloads = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", None),
            }
            for doc in batch_chunks
        ]

        vector_db.upsert(batch_ids, batch_embeddings, batch_payloads)

    print(f"Inserted {total} vectors in batches of {batch_size}")