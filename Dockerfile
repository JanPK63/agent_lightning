FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose the dashboard port
EXPOSE 8501

# Run the dashboard
CMD ["streamlit", "run", "monitoring_dashboard_integrated.py", "--server.port=8501", "--server.address=0.0.0.0"]