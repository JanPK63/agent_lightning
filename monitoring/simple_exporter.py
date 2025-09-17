#!/usr/bin/env python3
from prometheus_client import start_http_server, Gauge
import psutil
import time

# Create metrics
cpu_gauge = Gauge('cpu_usage_percent', 'CPU usage')
memory_gauge = Gauge('memory_usage_percent', 'Memory usage')

def collect_metrics():
    while True:
        cpu_gauge.set(psutil.cpu_percent(interval=1))
        memory_gauge.set(psutil.virtual_memory().percent)
        time.sleep(5)

if __name__ == '__main__':
    start_http_server(8000)
    print("✅ Metrics server started on http://localhost:8000")
    collect_metrics()