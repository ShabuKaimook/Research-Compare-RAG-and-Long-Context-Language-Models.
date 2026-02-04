from .embedding import embed_query
from .query_rewriter import rewrite_query
from .multi_query import multi_query_search
from .dedup_chunks import deduplicate_chunks
from .reranker import rerank_chunks
from .merge_chunks import merge_heading_chunks


def retrieve_context(question: str, db, top_k: int = 4):
    query_vector = embed_query(question)
    result = db.search(query_vector, top_k=top_k)

    contexts = result["contexts"]
    sources = result["sources"]

    context_text = "\n\n".join(contexts)

    return context_text, sources


# OPTIMIZED RETRIEVE CONTEXT WITH MULTI-QUERY, DEDUP, RERANK
def advanced_retrieve_context(
    question: str,
    db,
    top_k: int = 4,
):
    print("Starting advanced_retrieve_context")

    # 1. rewrite question -> fix grammar, expand query
    queries = rewrite_query(question)
    print("Rewritten queries:", queries)

    # 2. retrieve chunks from vector DB (multi-query)
    all_results = []
    all_sources = set()

    for q in queries:
        result = multi_query_search(q, db, top_k=top_k)
        print(f"Retrieved {result['sources']} chunks for query: {q}")
        all_results.extend(result["contexts"])
        all_sources.update(result["sources"])

    print("Multi-query retrieved chunks:", len(all_results))
    print("Multi-query")
    print("Multi-query retrieved sources:", all_sources)

    # 3. deduplicate chunks
    unique_chunks = deduplicate_chunks(all_results)
    print("Deduplicated chunks:", len(unique_chunks))

    if not unique_chunks:
        return "", []

    # merge heading chunks
    unique_chunks = merge_heading_chunks(unique_chunks)
    print("After merge heading:", len(unique_chunks))

    # 4. rerank chunks
    if len(unique_chunks) <= top_k:
        ranked_chunks = unique_chunks
    else:
        ranked_chunks = rerank_chunks(question, unique_chunks, top_k=top_k)

    print("Reranked chunks:", len(ranked_chunks))
    print("Top ranked chunk preview:", ranked_chunks[0])

    # 5. join context
    context_text = "\n\n".join(ranked_chunks)
    print("Final context length:", len(context_text))

    return context_text, list(all_sources)
