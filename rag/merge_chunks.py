import re

def is_heading(chunk: str) -> bool:
    """
    ตรวจว่าเป็น heading เช่น:
    1.2 Something
    2. Title
    """
    chunk = chunk.strip()
    return (
        len(chunk.split()) < 15
        and re.match(r"^\d+(\.\d+)*\s+", chunk) is not None
    )


def merge_heading_chunks(chunks: list[str]) -> list[str]:
    merged = []
    i = 0

    while i < len(chunks):
        current = chunks[i].strip()

        # ถ้าเป็น heading และมี chunk ถัดไป
        if is_heading(current) and i + 1 < len(chunks):
            next_chunk = chunks[i + 1].strip()
            combined = current + "\n" + next_chunk
            merged.append(combined)
            i += 2  # ข้าม chunk ถัดไป
        else:
            merged.append(current)
            i += 1

    return merged
