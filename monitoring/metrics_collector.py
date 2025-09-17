#!/usr/bin/env python3
"""
Metrics Collector for Agent Lightning System
"""

import time
import psutil
import requests
import json
from datetime import datetime
from typing import Dict, List, Any
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.services = {
            'dashboard': 'http://localhost:8051',
            'agent_api': 'http://localhost:8052',
            'task_api': 'http://localhost:8053',
            'knowledge_api': 'http://localhost:8054',
            'auth_api': 'http://localhost:8055',
            'internet_agent': 'http://localhost:8892'
        }
        self.metrics = {}
        self.running = False
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    
    def check_service_health(self, service_name: str, url: str) -> Dict[str, Any]:
        """Check health of individual service"""
        try:
            start_time = time.time()
            response = requests.get(f"{url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            return {
                'service': service_name,
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time_ms': round(response_time, 2),
                'status_code': response.status_code,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'service': service_name,
                'status': 'down',
                'response_time_ms': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        service_metrics = []
        for service_name, url in self.services.items():
            metrics = self.check_service_health(service_name, url)
            service_metrics.append(metrics)
            
        return {
            'system': self.collect_system_metrics(),
            'services': service_metrics,
            'collection_time': datetime.now().isoformat()
        }
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    metrics = self.collect_all_metrics()
                    self.metrics = metrics
                    
                    # Log critical issues
                    system = metrics['system']
                    if system['cpu_percent'] > 80:
                        logger.warning(f"High CPU usage: {system['cpu_percent']}%")
                    if system['memory_percent'] > 80:
                        logger.warning(f"High memory usage: {system['memory_percent']}%")
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"Monitoring started with {interval}s interval")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics or self.collect_all_metrics()

if __name__ == "__main__":
    collector = MetricsCollector()
    collector.start_monitoring(interval=10)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Monitoring stopped")