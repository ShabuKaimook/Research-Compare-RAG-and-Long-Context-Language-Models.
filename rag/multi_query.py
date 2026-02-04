from .embedding import embed_query


def multi_query_search(query: str, db, top_k: int = 5):
    vector = embed_query(query)
    result = db.search(vector, top_k=top_k)
    return {
        "contexts": result["contexts"],
        "sources": result["sources"],
    }
