#!/usr/bin/env python3
"""
Test script for HTTP metrics collection
Demonstrates that the HTTP metrics middleware is working correctly
"""

import asyncio
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient

from monitoring.http_metrics_middleware import add_http_metrics_middleware
from monitoring.metrics import get_metrics

# Create a test FastAPI app
app = FastAPI(title="Test Metrics API")
app = add_http_metrics_middleware(app, service_name="test_api")

# Get metrics instance
metrics = get_metrics("test_api")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/slow")
async def slow_endpoint():
    """Slow endpoint for testing response times"""
    await asyncio.sleep(0.1)  # Simulate slow operation
    return {"message": "Slow response"}

@app.get("/error")
async def error_endpoint():
    """Error endpoint for testing error rates"""
    raise Exception("Test error")

@app.get("/metrics")
async def get_prometheus_metrics():
    """Get current metrics in Prometheus format"""
    return metrics.get_all_metrics_output()

def test_http_metrics():
    """Test HTTP metrics collection"""
    print("🚀 Testing HTTP Metrics Collection")
    print("=" * 50)

    client = TestClient(app)

    # Test successful requests
    print("\n📊 Testing successful requests...")
    for i in range(5):
        response = client.get("/")
        print(f"  Request {i+1}: {response.status_code}")

    # Test health endpoint
    print("\n🏥 Testing health endpoint...")
    for i in range(3):
        response = client.get("/health")
        print(f"  Health check {i+1}: {response.status_code}")

    # Test slow endpoint
    print("\n⏱️  Testing slow endpoint...")
    for i in range(2):
        start_time = time.time()
        response = client.get("/slow")
        duration = time.time() - start_time
        print(".3f")

    # Test error endpoint
    print("\n❌ Testing error endpoint...")
    for i in range(2):
        try:
            response = client.get("/error")
        except:
            print(f"  Error request {i+1}: Exception raised (expected)")

    # Get metrics
    print("\n📈 Current Metrics:")
    print("-" * 30)
    response = client.get("/metrics")
    print(response.text)

    print("\n✅ HTTP Metrics Test Complete!")
    print("Metrics should show:")
    print("  - HTTP request counts by endpoint and method")
    print("  - Response time histograms")
    print("  - Error counts")

if __name__ == "__main__":
    test_http_metrics()