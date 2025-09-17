#!/usr/bin/env python3
"""
Start Monitoring System for Agent Lightning
"""

import subprocess
import sys
import time
import os

def start_monitoring():
    """Start the monitoring system"""
    print("🔍 Starting Agent Lightning Monitoring System...")
    
    # Change to monitoring directory
    monitoring_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(monitoring_dir)
    
    try:
        # Start health check API
        print("Starting Health Check API on port 8899...")
        process = subprocess.Popen([
            sys.executable, 'health_check_api.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Monitoring system started successfully!")
        print("📊 Health Check API: http://localhost:8899")
        print("🔍 Metrics endpoint: http://localhost:8899/metrics")
        print("🏥 Services status: http://localhost:8899/services/status")
        print("💻 System status: http://localhost:8899/system/status")
        print("\nPress Ctrl+C to stop monitoring...")
        
        # Keep running
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring system...")
        process.terminate()
        process.wait()
        print("✅ Monitoring stopped")
    except Exception as e:
        print(f"❌ Error starting monitoring: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(start_monitoring())