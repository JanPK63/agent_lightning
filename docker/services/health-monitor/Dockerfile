FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir requests psutil

COPY docker/services/health-monitor/monitor.py .

RUN groupadd -r monitor && useradd -r -g monitor monitor
RUN chown -R monitor:monitor /app
USER monitor

CMD ["python", "monitor.py"]