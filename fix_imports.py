#!/usr/bin/env python3
"""Fix OpenTelemetry imports for monitoring dashboard"""

import subprocess
import sys

packages = [
    "opentelemetry-api",
    "opentelemetry-sdk", 
    "opentelemetry-exporter-otlp",
    "opentelemetry-exporter-prometheus",
    "opentelemetry-instrumentation-requests",
    "opentelemetry-instrumentation-logging",
    "prometheus-client"
]

print("Installing OpenTelemetry packages...")
for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("\nAll packages installed successfully!")
print("Now you can run: streamlit run monitoring_dashboard.py")