from pathlib import Path
from rag.ingest_file import ingest_file

# UPLOAD_DIR = Path("./assets")
# UPLOAD_DIR = Path("./assets/test02/document.txt")


def auto_ingest(db, path):
    path = Path(path)
    if not path.exists():
        return

    for file in path.iterdir():
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