from pathlib import Path
from rag.ingest_file import ingest_file

UPLOAD_DIR = Path("./assets")


def auto_ingest(db):
    if not UPLOAD_DIR.exists():
        return

    for file in UPLOAD_DIR.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() not in {".pdf", ".txt", ".html"}:
            continue

        marker = file.with_suffix(file.suffix)
        if marker.exists():
            continue  # already ingested

        print(f"[AUTO INGEST] Ingesting {file.name}")
        ingest_file(str(file), db)

        marker.touch()  # mark as ingested
