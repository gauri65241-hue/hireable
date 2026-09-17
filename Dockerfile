# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build
# Output lands in /frontend/../frontend_dist (vite.config outDir: '../frontend_dist')


# ── Stage 2: Python API ────────────────────────────────────────────────────────
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

# Copy built frontend from stage 1
COPY --from=frontend-builder /frontend_dist ./frontend_dist

# Pre-warm SBERT model so first analysis request doesn't block 30s
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "linkedin_api:app", "--host", "0.0.0.0", "--port", "8000"]
