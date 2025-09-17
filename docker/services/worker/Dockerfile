FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt celery

COPY . .

RUN groupadd -r worker && useradd -r -g worker worker
RUN chown -R worker:worker /app
USER worker

CMD ["celery", "worker", "-A", "agent_message_queue", "--loglevel=info"]