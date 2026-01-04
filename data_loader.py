from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))


# chunk size is the number of tokens in each chunk (the value is characters)
# chunk overlap is how much of the end of a chunk is shared with the next chunk (the value is characters)
# for example if chunkoverlap = 1
# the chuck content is hello world my name is guy
# the first chunck is hello world my
# and the second chuck is my name is guy
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(pdf_path: str):
    # get the text from the pdf
    docs = PDFReader().load_data(file=pdf_path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    # split the text into chunks
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks


# return vector
def embed_texts(texts: list[str]) -> list[list[float]]:
    # send the request to openai by pass all the text that chunked
    # then the response is the vector that will be stored in the vector database
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)

    return [item.embedding for item in response.data]
