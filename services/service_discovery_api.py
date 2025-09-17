#!/usr/bin/env python3
"""
Service Discovery API Endpoint
Provides HTTP API for service registration and discovery
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the existing service discovery
from service_discovery import ServiceRegistry, ServiceInstance, ServiceStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceRegisterRequest(BaseModel):
    """Service registration request"""
    name: str = Field(description="Service name")
    host: str = Field(description="Service host")
    port: int = Field(description="Service port")
    version: str = Field(default="1.0.0", description="Service version")
    protocol: str = Field(default="http", description="Protocol")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Service metadata")
    tags: List[str] = Field(default_factory=list, description="Service tags")
    health_check_url: Optional[str] = Field(default=None, description="Health check URL")
    weight: int = Field(default=1, description="Load balancing weight")
    zone: Optional[str] = Field(default=None, description="Availability zone")


class ServiceDiscoveryAPI:
    """Service Discovery API Service"""
    
    def __init__(self):
        self.app = FastAPI(title="Service Discovery API", version="1.0.0")
        self.registry = ServiceRegistry()
        
        logger.info("✅ Service Discovery API initialized")
        
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
            """Health check endpoint"""
            return {
                "service": "service_discovery",
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "registered_services": len(self.registry.services),
                "total_instances": sum(len(instances) for instances in self.registry.services.values())
            }
        
        @self.app.post("/register")
        async def register_service(request: ServiceRegisterRequest):
            """Register a new service instance"""
            try:
                instance = await self.registry.register_service(
                    name=request.name,
                    host=request.host,
                    port=request.port,
                    version=request.version,
                    protocol=request.protocol,
                    metadata=request.metadata,
                    tags=request.tags,
                    health_check_url=request.health_check_url,
                    weight=request.weight,
                    zone=request.zone
                )
                
                return {
                    "status": "registered",
                    "instance": instance.to_dict()
                }
            except Exception as e:
                logger.error(f"Failed to register service: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/deregister/{service_id}")
        async def deregister_service(service_id: str):
            """Deregister a service instance"""
            success = await self.registry.deregister_service(service_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Service {service_id} not found")
            
            return {"status": "deregistered", "service_id": service_id}
        
        @self.app.get("/discover/{service_name}")
        async def discover_service(service_name: str, tags: Optional[str] = None):
            """Discover available service instances"""
            tag_list = tags.split(",") if tags else None
            instances = self.registry.discover_service(service_name, tag_list)
            
            return {
                "service": service_name,
                "instances": [instance.to_dict() for instance in instances],
                "count": len(instances)
            }
        
        @self.app.get("/services")
        async def list_services():
            """List all registered services"""
            services = {}
            for name, instances in self.registry.services.items():
                services[name] = {
                    "instances": [i.to_dict() for i in instances],
                    "healthy": sum(1 for i in instances if i.status == ServiceStatus.HEALTHY),
                    "total": len(instances)
                }
            
            return services
        
        @self.app.get("/topology")
        async def get_topology():
            """Get complete service topology"""
            return self.registry.get_service_topology()
        
        @self.app.get("/metrics/{service_name}")
        async def get_service_metrics(service_name: str):
            """Get metrics for a specific service"""
            metrics = self.registry.get_service_metrics(service_name)
            if not metrics:
                raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
            
            return metrics
        
        @self.app.get("/load-balance/{service_name}")
        async def get_service_with_load_balancing(
            service_name: str,
            strategy: str = "round-robin",
            tags: Optional[str] = None
        ):
            """Get a service instance with load balancing"""
            tag_list = tags.split(",") if tags else None
            instance = self.registry.get_service_instance(service_name, strategy, tag_list)
            
            if not instance:
                raise HTTPException(
                    status_code=404,
                    detail=f"No available instances for service {service_name}"
                )
            
            return instance.to_dict()
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Service Discovery API starting up...")
        
        # Auto-register existing services
        from service_discovery import auto_register_services
        await auto_register_services(self.registry)
        
        logger.info(f"Service Discovery ready with {len(self.registry.services)} services")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Service Discovery API shutting down...")
        
        # Cancel all health check tasks
        for task in self.registry.health_check_tasks.values():
            task.cancel()


def main():
    """Main entry point"""
    import uvicorn
    
    service = ServiceDiscoveryAPI()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("SERVICE_DISCOVERY_PORT", 8005))
    logger.info(f"Starting Service Discovery API on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()