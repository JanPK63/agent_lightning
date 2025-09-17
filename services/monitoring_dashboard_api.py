#!/usr/bin/env python3
"""
Monitoring Dashboard API Wrapper
Provides JSON health endpoint for Streamlit dashboard
"""

import os
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringDashboardAPI:
    """API wrapper for Streamlit Monitoring Dashboard"""
    
    def __init__(self):
        self.app = FastAPI(title="Monitoring Dashboard API", version="1.0.0")
        self.streamlit_url = "http://localhost:8051"
        
        logger.info("✅ Monitoring Dashboard API initialized")
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint that returns JSON"""
            # Check if Streamlit is running
            streamlit_healthy = False
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.streamlit_url, timeout=2) as response:
                        streamlit_healthy = response.status == 200
            except:
                pass
            
            return {
                "service": "monitoring_dashboard",
                "status": "healthy" if streamlit_healthy else "degraded",
                "streamlit_running": streamlit_healthy,
                "streamlit_url": self.streamlit_url,
                "api_wrapper": "active",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/")
        async def root():
            """Redirect to Streamlit dashboard"""
            return RedirectResponse(url=self.streamlit_url)
        
        @self.app.get("/dashboard")
        async def dashboard():
            """Redirect to Streamlit dashboard"""
            return RedirectResponse(url=self.streamlit_url)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get dashboard metrics"""
            # Check various services and return aggregated metrics
            metrics = {
                "services_monitored": 16,
                "services_healthy": 0,
                "services_unhealthy": 0,
                "last_update": datetime.utcnow().isoformat()
            }
            
            # Check each service
            services = [
                ("API Gateway", 8000),
                ("Auth Service", 8001),
                ("Agent Designer", 8002),
                ("Workflow Engine", 8003),
                ("AI Model Service", 8004),
                ("Service Discovery", 8005),
                ("Integration Hub", 8006),
                ("Monitoring Service", 8007),
                ("WebSocket Service", 8008),
                ("RL Server", 8010),
                ("RL Orchestrator", 8011),
                ("Memory Service", 8012),
                ("Checkpoint Service", 8013),
                ("Batch Accumulator", 8014),
                ("AutoGen Integration", 8015),
            ]
            
            async with aiohttp.ClientSession() as session:
                for name, port in services:
                    try:
                        async with session.get(f"http://localhost:{port}/health", timeout=1) as response:
                            if response.status == 200:
                                metrics["services_healthy"] += 1
                            else:
                                metrics["services_unhealthy"] += 1
                    except:
                        metrics["services_unhealthy"] += 1
            
            return metrics
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Monitoring Dashboard API starting up...")
        
        # Ensure Streamlit is running
        streamlit_running = False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.streamlit_url, timeout=2) as response:
                    streamlit_running = response.status == 200
        except:
            pass
        
        if streamlit_running:
            logger.info(f"Streamlit dashboard accessible at {self.streamlit_url}")
        else:
            logger.warning("Streamlit dashboard not responding")
        
        logger.info("Monitoring Dashboard API ready")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Monitoring Dashboard API shutting down...")


def main():
    """Main entry point"""
    import uvicorn
    
    service = MonitoringDashboardAPI()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service on port 8052 (API wrapper for 8051 Streamlit)
    port = int(os.getenv("MONITORING_API_PORT", 8052))
    logger.info(f"Starting Monitoring Dashboard API on port {port}")
    logger.info(f"Wrapping Streamlit dashboard at http://localhost:8051")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()