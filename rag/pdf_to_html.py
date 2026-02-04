import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATALAB_MARKER_URL = "https://www.datalab.to/api/v1/marker"

def pdf_to_html(file_path: str) -> str:
    api_key = os.getenv("DATALAB_API_KEY")
    if not api_key:
        raise RuntimeError("DATALAB_API_KEY not set")

    headers = {"X-API-Key": api_key}

    # submit
    with open(file_path, "rb") as f:
      r = requests.post(
          DATALAB_MARKER_URL,
          headers=headers,
          files={
              "file": (
                  os.path.basename(file_path), # file name
                  f,
                  "application/pdf", # content type
              )
          },
          data={"output_format": "html", "mode": "balanced"},
      )

    if not r.ok:
        raise RuntimeError(r.text)

    check_url = r.json()["request_check_url"]

    # poll
    while True:
        result = requests.get(check_url, headers=headers).json()
        if result["status"] == "complete":
            html = result["html"]
            
            file_path = "./upload/" + file_path.split("/")[-1]
            html_path = Path(file_path).with_suffix(".html")
            html_path.write_text(html, encoding="utf-8")

            return str(html_path)
          
        if result["status"] == "failed":
            raise RuntimeError(result.get("error"))
        time.sleep(2)
        
# print(pdf_to_html("./upload/trigon.pdf"))