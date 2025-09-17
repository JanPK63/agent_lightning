#!/usr/bin/env python3
import requests
import time
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVICES = {
    'api': 'http://api:8001/health',
    'dashboard': 'http://dashboard:8051/_stcore/health',
    'agents': 'http://agents:8002/health',
    'rl-orchestrator': 'http://rl-orchestrator:8003/health'
}

def check_service_health(name, url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ {name} is healthy")
            return True
        else:
            logger.warning(f"⚠️ {name} returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ {name} health check failed: {e}")
        return False

def main():
    logger.info("🔍 Starting health monitor...")
    
    while True:
        healthy_services = 0
        total_services = len(SERVICES)
        
        for service_name, health_url in SERVICES.items():
            if check_service_health(service_name, health_url):
                healthy_services += 1
        
        logger.info(f"📊 Health Status: {healthy_services}/{total_services} services healthy")
        
        time.sleep(30)

if __name__ == "__main__":
    main()