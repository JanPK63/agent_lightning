#!/usr/bin/env python3
"""
Start Complete Monitoring Stack for Agent Lightning
Prometheus + Grafana + Custom Exporter
"""

import subprocess
import sys
import time
import os
import requests

def start_monitoring_stack():
    """Start the complete monitoring stack"""
    print("🚀 Starting Agent Lightning Monitoring Stack...")
    
    monitoring_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(monitoring_dir)
    
    try:
        # 1. Start Prometheus exporter
        print("📊 Starting Prometheus exporter...")
        exporter_process = subprocess.Popen([
            sys.executable, 'prometheus_exporter.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(3)
        
        # 2. Start Docker stack
        print("🐳 Starting Prometheus + Grafana with Docker...")
        docker_process = subprocess.run([
            'docker-compose', 'up', '-d'
        ], capture_output=True, text=True)
        
        if docker_process.returncode != 0:
            print(f"❌ Docker error: {docker_process.stderr}")
            return 1
        
        print("✅ Monitoring stack started successfully!")
        print("\n📊 Access Points:")
        print("🔍 Prometheus: http://localhost:9091")
        print("📈 Grafana: http://localhost:3000 (admin/admin123)")
        print("📡 Metrics Exporter: http://localhost:9090/metrics")
        print("🏥 Health API: http://localhost:8899")
        
        print("\n🎯 Grafana Setup:")
        print("1. Login to Grafana at http://localhost:3000")
        print("2. Add Prometheus data source: http://prometheus:9090")
        print("3. Import dashboard from grafana_dashboard.json")
        
        # Wait for services to be ready
        print("\n⏳ Waiting for services to be ready...")
        time.sleep(10)
        
        # Check if services are up
        services = {
            "Prometheus": "http://localhost:9091",
            "Grafana": "http://localhost:3000",
            "Metrics Exporter": "http://localhost:9090/metrics"
        }
        
        for name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {name} is ready")
                else:
                    print(f"⚠️ {name} returned status {response.status_code}")
            except:
                print(f"❌ {name} is not responding")
        
        print("\n🎉 Monitoring stack is ready!")
        print("Press Ctrl+C to stop all services...")
        
        # Keep running
        try:
            exporter_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring stack...")
            exporter_process.terminate()
            subprocess.run(['docker-compose', 'down'], cwd=monitoring_dir)
            print("✅ Monitoring stack stopped")
        
    except Exception as e:
        print(f"❌ Error starting monitoring stack: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(start_monitoring_stack())