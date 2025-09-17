#!/usr/bin/env python3
"""
Test and fix the monitoring setup
"""

import requests
import time
import subprocess
import sys

def test_monitoring():
    print("🔍 Testing monitoring setup...")
    
    # 1. Test if exporter is running and producing metrics
    try:
        response = requests.get("http://localhost:9090/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Metrics exporter is working")
            metrics = response.text
            if "system_cpu_percent" in metrics:
                print("✅ CPU metrics found")
            if "system_memory_percent" in metrics:
                print("✅ Memory metrics found")
            if "service_up" in metrics:
                print("✅ Service metrics found")
        else:
            print("❌ Metrics exporter not responding")
            return False
    except Exception as e:
        print(f"❌ Cannot reach metrics exporter: {e}")
        return False
    
    # 2. Test Prometheus
    try:
        response = requests.get("http://localhost:9091", timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus is running")
        else:
            print("❌ Prometheus not responding")
    except Exception as e:
        print(f"❌ Cannot reach Prometheus: {e}")
    
    # 3. Test Grafana
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Grafana is running")
        else:
            print("❌ Grafana not responding")
    except Exception as e:
        print(f"❌ Cannot reach Grafana: {e}")
    
    return True

def start_simple_working_setup():
    """Start a simple setup that actually works"""
    print("🚀 Starting simple working monitoring...")
    
    # Start just the exporter first
    print("1. Starting metrics exporter...")
    exporter = subprocess.Popen([sys.executable, "prometheus_exporter.py"])
    
    time.sleep(3)
    
    # Test it works
    if test_monitoring():
        print("✅ Basic monitoring is working!")
        print("📊 View metrics at: http://localhost:9090/metrics")
        
        try:
            input("Press Enter to stop...")
        except KeyboardInterrupt:
            pass
        
        exporter.terminate()
        print("✅ Stopped")
    else:
        exporter.terminate()
        print("❌ Setup failed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_monitoring()
    else:
        start_simple_working_setup()