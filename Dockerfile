# ── Python API ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for pdfminer, sentence-transformers, httpx
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install-deps chromium \
    && playwright install chromium

COPY . .
# Frontend pre-built locally and committed — copy to the path the API expects
COPY frontend/dist ./frontend_dist

# Pre-warm SBERT model so first analysis request doesn't block 30s
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "linkedin_api:app", "--host", "0.0.0.0", "--port", "8000"]
