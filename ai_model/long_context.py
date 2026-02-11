from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
import time
import os

load_dotenv()

llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL"),
    temperature=0,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY"),
)

llm_non_stream = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL"),
    temperature=0,
    streaming=False,   # 🔴 IMPORTANT
    api_key=os.getenv("OPENAI_API_KEY"),
)

prompt = ChatPromptTemplate.from_template(
    """You are a careful and truthful assistant.

You are given multiple documents as CONTEXT.
Each document starts with a source marker in the following format:

[SOURCE: filename]

The context may contain multiple sources.
You MUST rely ONLY on the information provided in the context.
Do NOT use prior knowledge.
Do NOT invent facts.
Do NOT invent sources.

--------------------------------
TASK
--------------------------------
Answer the question using ONLY the provided context.

If the context does not contain enough information to answer the question,
respond exactly with:
"I don't know"

--------------------------------
LANGUAGE RULES
--------------------------------
- If the question is in a different language from the context,
  translate the answer into the language of the question.
- If the question asks for translation, translate using the context only.

--------------------------------
SOURCE RULES (VERY IMPORTANT)
--------------------------------
- Every answer MUST include a "Sources used" section.
- Only list sources that appear in the context.
- Only list sources that directly support the answer.
- Do NOT list sources that were not used.
- Do NOT invent filenames.
- If no source was used, write: Sources used: NONE

--------------------------------
OUTPUT FORMAT (MUST FOLLOW EXACTLY)
--------------------------------
<your answer here>

Sources used:
- <filename>
- <filename>

--------------------------------
CONTEXT
--------------------------------
{context}

--------------------------------
QUESTION
--------------------------------
{question}
"""
)

chain = prompt | llm | StrOutputParser()
chain_non_stream = prompt | llm_non_stream | StrOutputParser()

UPLOAD_DIR = "./upload"


# use to build full context from all files in upload directory
def build_full_context(upload_dir=UPLOAD_DIR) -> str:
    contents = []

    print("Adding files to context...")
    for file in os.listdir(upload_dir):
        path = os.path.join(upload_dir, file)
        if not os.path.isfile(path):
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                contents.append(f"[SOURCE: {file}]\n{f.read().strip()}")
        except Exception:
            continue

    # print("context:", contents)
    return "\n\n".join(contents)


def ask_long_context(question: str):
    start_time = time.time()

    context = build_full_context()

    if not context.strip():
        return {
            "answer": "I don't know",
            "sources": [],
            "tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            },
            "latency_seconds": 0.0,
        }

    with get_openai_callback() as cb:
        answer = chain_non_stream.invoke(
            {
                "context": context,
                "question": question,
            }
        )

    latency = time.time() - start_time

    return {
        "answer": answer,
        "sources": ["FULL_CONTEXT"],  # long context always uses everything
        "tokens": {
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens": cb.total_tokens,
            "cost_usd": cb.total_cost,
        },
        "latency_seconds": round(latency, 3),
    }

def stream_long_context(context: str, question: str):
    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
