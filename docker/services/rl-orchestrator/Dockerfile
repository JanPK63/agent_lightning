FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-rl.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-rl.txt

COPY . .

RUN groupadd -r rluser && useradd -r -g rluser rluser
RUN chown -R rluser:rluser /app
USER rluser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health')" || exit 1

CMD ["python", "-m", "rl_orch.core.orchestrator"]