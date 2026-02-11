from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ai_model.rag import ask_rag
from ai_model.long_context import ask_long_context
from rag.ingest_file import ingest_file
from vector_db import QdrantStorage

import csv
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import os
import json


load_dotenv()

TEST_DIR = Path("./assets/test02")


llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL"),
    temperature=0,
    streaming=False,
    api_key=os.getenv("OPENAI_API_KEY"),
)

evaluation_prompt = """
You are an expert evaluator for Retrieval-Augmented Generation systems.

Your task is to evaluate an AI-generated answer using the following information:

- Question
- Ground Truth Answer (if available)
- Retrieved Context
- Model Answer

You must score the answer on the following 5 dimensions from 1 to 5:

-------------------------------------------------------
1. Correctness
-------------------------------------------------------
How factually correct is the answer compared to the Ground Truth?
5 = Fully correct
3 = Partially correct
1 = Mostly incorrect

-------------------------------------------------------
2. Completeness
-------------------------------------------------------
Does the answer cover all important parts of the Ground Truth?
5 = Covers all key points
3 = Missing some important details
1 = Very incomplete

-------------------------------------------------------
3. Faithfulness
-------------------------------------------------------
Is the answer fully supported by the provided Context?
5 = Fully grounded in context
3 = Mostly grounded but slight extrapolation
1 = Contains unsupported claims

-------------------------------------------------------
4. Hallucination
-------------------------------------------------------
Does the answer contain fabricated or unsupported information?
5 = No hallucination
3 = Minor unsupported claims
1 = Major hallucinations

-------------------------------------------------------
5. Readability
-------------------------------------------------------
Is the answer clear, coherent, and well structured?
5 = Very clear and professional
3 = Understandable but somewhat unclear
1 = Hard to read or poorly structured

-------------------------------------------------------
IMPORTANT RULES
-------------------------------------------------------

- If Ground Truth is empty (no-answer question):
    - If the model correctly says "I don't know", give:
        Correctness = 5
        Completeness = 5
        Faithfulness = 5
        Hallucination = 5
    - If the model attempts to answer anyway:
        Correctness = 1
        Faithfulness = 1
        Hallucination = 1

- Do NOT be overly generous.
- Penalize hallucinations strictly.
- Be objective.

-------------------------------------------------------
INPUT
-------------------------------------------------------

Context:
{context}

Question:
{question}

Ground Truth:
{ground_truth}

Model Answer:
{model_answer}

-------------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
-------------------------------------------------------

Return ONLY valid JSON in this exact format:

{{
  "correctness": <1-5>,
  "completeness": <1-5>,
  "faithfulness": <1-5>,
  "hallucination": <1-5>,
  "readability": <1-5>,
  "explanation": "<short explanation of scores>"
}}
"""

prompt = ChatPromptTemplate.from_template(evaluation_prompt)

chain = prompt | llm | StrOutputParser()


def ingest(path: Path, db: QdrantStorage):
    print("Auto ingest on startup in evaluation")
    ingest_file(path, db)
    print("Auto ingest completed")


def load_documents(base_path: Path):
    """
    Load all documents in assets/test02/docs
    Returns: dict {index: document_text}
    """
    docs_path = base_path / "docs"
    documents = {}

    files = sorted(os.listdir(docs_path))

    for idx, file_name in enumerate(files):
        file_path = docs_path / file_name
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                documents[str(idx)] = f.read()

    return documents


def load_questions(base_path: Path):
    evaluation_questions = []

    files = [
        "single_passage_answer_question.txt",
        "multi_passage_answer_question.txt",
        "no_answer_question.txt",
    ]

    question_id = 0

    for file_name in files:
        file_path = base_path / file_name  # 🔴 FIXED
        if not file_path.exists():
            continue

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                gold_answer = row.get("answer")
                gold_answer = gold_answer.strip() if gold_answer else None

                evaluation_questions.append(
                    {
                        "question_id": question_id,
                        "document_index": row.get("document_index"),
                        "question": row.get("question", "").strip(),
                        "gold_answer": gold_answer,
                        "has_answer": gold_answer is not None,
                        "category": file_name,
                    }
                )

                question_id += 1

    return evaluation_questions


def safe_parse_evaluation(result: str):
    is_parse_failed = False

    try:
        parsed = json.loads(result)

        # Validate required fields
        required_fields = [
            "correctness",
            "completeness",
            "faithfulness",
            "hallucination",
            "readability",
        ]

        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing field: {field}")

        # Clamp scores to 1–5
        for field in required_fields:
            value = int(parsed[field])
            parsed[field] = max(1, min(5, value))

        return parsed, is_parse_failed

    except Exception as e:
        print("⚠️ Evaluation JSON parsing failed:", e)
        is_parse_failed = True

        return {
            "correctness": 1,
            "completeness": 1,
            "faithfulness": 1,
            "hallucination": 1,
            "readability": 1,
            "explanation": "Invalid evaluation output",
        }, is_parse_failed


def evaluate(answers: List[str], evaluation_questions: List[Dict], documents):
    evaluation_results = []
    parse_failures = 0

    for model_answer, question in zip(answers, evaluation_questions):
        doc_index = question["document_index"]
        context = documents.get(str(doc_index), "")

        result = chain.invoke(
            {
                "context": context,
                "question": question["question"],
                "ground_truth": question["gold_answer"] or "NONE",
                "model_answer": model_answer,
            }
        )

        parsed, is_parse_failed = safe_parse_evaluation(result)
        evaluation_results.append(parsed)

        if is_parse_failed:
            parse_failures += 1

    return evaluation_results, parse_failures


def run_evaluation(path: Path, db: QdrantStorage):
    docs_path = str(path / "docs")
    file_in_docs = os.listdir(docs_path)

    for file_name in file_in_docs:
        print("Ingesting file:", file_name)
        ingest(docs_path + "/" + file_name, db)

    evaluation_questions = load_questions(path)

    # build context from all files in upload directory
    documents = load_documents(path)

    rag_answers = []
    long_answers = []

    print("asking rag and long context...")
    for q in evaluation_questions:
        question_text = q["question"]

        # ---- RAG ----
        rag_result = ask_rag(question_text, db)
        rag_answers.append(rag_result)

        # ---- LONG CONTEXT ----
        long_output = ask_long_context(question_text)
        long_answers.append(long_output)

    print("asking rag and long context completed")

    # ---- Evaluate ----
    print("evaluating...")
    rag_scores, rag_parse_failures = evaluate(
        rag_answers, evaluation_questions, documents
    )
    long_scores, long_parse_failures = evaluate(
        long_answers, evaluation_questions, documents
    )

    # ---- Calculate Parse Failures ----
    parse_failures = rag_parse_failures + long_parse_failures

    print("evaluating completed")
    for i, eq in enumerate(evaluation_questions):
        eq["rag"] = {
            "answer": rag_answers[i],
            "scores": rag_scores[i],
        }

        eq["long"] = {
            "answer": long_answers[i],
            "scores": long_scores[i],
        }

    return evaluation_questions, parse_failures


# db = QdrantStorage(collection_name="test")
# evaluation_questions, parse_failures = run_evaluation(TEST_DIR, db)

# print(
#     "evalation success",
#     evaluation_questions[:2],
#     evaluation_questions[:2],
# )

# print(
#     "evaluation_questions",
#     json.dumps(evaluation_questions, ensure_ascii=False, indent=4),
# )
# print("Parse Failures:", parse_failures)
