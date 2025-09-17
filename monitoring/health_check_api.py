#!/usr/bin/env python3
"""
Health Check API for Agent Lightning System
Provides health endpoints for all services
"""

from flask import Flask, jsonify
from metrics_collector import MetricsCollector
import threading
import time

app = Flask(__name__)
collector = MetricsCollector()

@app.route('/health', methods=['GET'])
def health():
    """Basic health check"""
    return jsonify({'status': 'healthy', 'service': 'monitoring'})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get current system metrics"""
    return jsonify(collector.get_current_metrics())

@app.route('/services/status', methods=['GET'])
def services_status():
    """Get status of all services"""
    metrics = collector.get_current_metrics()
    services = metrics.get('services', [])
    
    summary = {
        'total_services': len(services),
        'healthy': len([s for s in services if s['status'] == 'healthy']),
        'unhealthy': len([s for s in services if s['status'] == 'unhealthy']),
        'down': len([s for s in services if s['status'] == 'down']),
        'services': services
    }
    
    return jsonify(summary)

@app.route('/system/status', methods=['GET'])
def system_status():
    """Get system resource status"""
    metrics = collector.get_current_metrics()
    system = metrics.get('system', {})
    
    status = 'healthy'
    if system.get('cpu_percent', 0) > 80 or system.get('memory_percent', 0) > 80:
        status = 'warning'
    if system.get('cpu_percent', 0) > 95 or system.get('memory_percent', 0) > 95:
        status = 'critical'
    
    return jsonify({
        'status': status,
        'system': system
    })

def start_monitoring():
    """Start background monitoring"""
    collector.start_monitoring(interval=15)

if __name__ == '__main__':
    # Start monitoring in background
    monitor_thread = threading.Thread(target=start_monitoring, daemon=True)
    monitor_thread.start()
    
    print("Health Check API starting on port 8899...")
    app.run(host='0.0.0.0', port=8899, debug=False)