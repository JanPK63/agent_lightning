#!/bin/bash
# Install dependencies for Agent Lightning monitoring

echo "Installing OpenTelemetry dependencies..."

/Users/jankootstra/miniforge3/bin/pip install \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    opentelemetry-instrumentation-requests \
    opentelemetry-instrumentation-logging \
    prometheus-client \
    opentelemetry-propagator-b3 \
    opentelemetry-exporter-prometheus

echo "Dependencies installed!"
echo "You can now run:"
echo "  cd ~/agent-lightning-main"
echo "  streamlit run monitoring_dashboard.py"