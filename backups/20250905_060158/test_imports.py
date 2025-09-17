#!/usr/bin/env python3
"""Test imports for monitoring dashboard"""

import sys
import subprocess

# Install required packages
packages = [
    "opentelemetry-api",
    "opentelemetry-sdk", 
    "opentelemetry-exporter-otlp",
    "opentelemetry-instrumentation-requests",
    "opentelemetry-instrumentation-logging",
    "prometheus-client",
    "opentelemetry-propagator-b3",
    "opentelemetry-exporter-prometheus"
]

print("Installing OpenTelemetry packages...")
for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("\nTesting imports...")
try:
    from observability_setup import AgentLightningObservability, MetricsAggregator
    print("✅ Successfully imported AgentLightningObservability and MetricsAggregator")
except ImportError as e:
    print(f"❌ Import error: {e}")
    
print("\nDone!")