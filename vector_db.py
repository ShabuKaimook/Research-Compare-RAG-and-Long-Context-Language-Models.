from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os


class QdrantStorage:
    # collection_name is the name of the collection
    # dim is the dimension of the vector
    def __init__(
        self,
        url="http://localhost:6333",
        collection_name="docs",
        dim=int(os.getenv("EMBED_DIM", "1536")),
    ):
        self.client = QdrantClient(url=url, timeout=30)

        # 🔍 DEBUG: print client info
        print("QdrantClient class:", type(self.client))
        print(
            "Methods:",
            [m for m in dir(self.client) if "search" in m],
        )

        self.collection = collection_name
        self.dim = dim
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    # ids are unique for each document
    # vectors are the embeddings of the documents
    # payloads are the metadata of the documents
    def upsert(self, ids, vectors, payloads):
        # PointStruct is the document
        point = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=point)

    # top_k is the number of results to return
    # query_vector is the embedding of the query
    def search(self, query_vector, top_k: int = 5):
        # search for the most similar documents
        # results = self.client.search(
        #     collection_name=self.collection,
        #     vector=query_vector,
        #     with_payload=True,
        #     limit=top_k,
        # )
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        response = response.points

        # extract the context and sources from the results
        contexts = []
        sources = set()
        for res in response:
            # get payload from res or None if it doesn't exist
            payload = getattr(res, "payload", None) or {}

            # get text from payload or empty string if it doesn't exist
            # get source from payload or empty string if it doesn't exist
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}