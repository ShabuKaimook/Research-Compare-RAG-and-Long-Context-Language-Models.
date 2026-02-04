def deduplicate_chunks(chunks: list[str]) -> list[str]:
    seen = set()
    unique = []
    for c in chunks:
        key = c.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    return unique
