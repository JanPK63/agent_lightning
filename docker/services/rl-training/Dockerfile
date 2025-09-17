FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-rl.txt .
RUN pip install --no-cache-dir -r requirements-rl.txt

COPY lightning_server_rl.py .

EXPOSE 8010

CMD ["python", "lightning_server_rl.py"]