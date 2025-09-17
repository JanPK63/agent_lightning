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

# Copy the monitoring dashboard and dependencies
COPY monitoring_dashboard_integrated.py .
COPY observability_setup.py .
COPY project_config.py .

# Expose port
EXPOSE 8053

# Run the dashboard
CMD ["streamlit", "run", "monitoring_dashboard_integrated.py", "--server.port=8053", "--server.address=0.0.0.0", "--server.headless=true"]