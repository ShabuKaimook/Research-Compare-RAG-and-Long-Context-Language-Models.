#!/usr/bin/env bash

set -e

echo "🚀 Starting RAG Dev Environment..."

# 1. Start FastAPI (uvicorn)
echo "▶ Starting FastAPI..."
uv run uvicorn main:app --reload &

# รอให้ FastAPI ขึ้นก่อน
sleep 2

# 2. Start Inngest dev server
echo "▶ Starting Inngest Dev Server..."
npx inngest-cli@latest dev -u http://127.0.0.1:8000 &

# รอ Inngest
sleep 2

# 3. Start Streamlit
echo "▶ Starting Streamlit UI..."
uv run streamlit run ./streamlit_app.py

echo "🛑 Shutting down..."
