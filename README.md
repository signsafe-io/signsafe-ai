# signsafe-ai

SignSafe.io AI worker written in Python.

Consumes messages from RabbitMQ, processes documents via a RAG pipeline, and stores results in PostgreSQL.

## Getting Started

```bash
cp .env.example .env
uv pip install -r pyproject.toml
python -m app.main
```

## Project Structure

```
app/
  main.py           - asyncio entrypoint
  config.py         - pydantic-settings configuration
  db.py             - asyncpg database connection
  queue.py          - aio-pika RabbitMQ connection
  workers/
    ingestion.py    - document ingestion pipeline
    analysis.py     - contract analysis pipeline
  services/
    parser.py       - PDF/DOCX text extraction
    embeddings.py   - vector embedding generation
    rag.py          - Qdrant retrieval
    llm.py          - LLM completion (Anthropic / OpenAI)
```
