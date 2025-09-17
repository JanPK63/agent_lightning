FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit

COPY agent_dashboard.py .

EXPOSE 8051

CMD ["streamlit", "run", "agent_dashboard.py", "--server.port=8051", "--server.address=0.0.0.0"]