FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit

COPY visual_code_builder.py .
COPY app.py .

EXPOSE 8052

CMD ["streamlit", "run", "visual_code_builder.py", "--server.port=8052", "--server.address=0.0.0.0"]