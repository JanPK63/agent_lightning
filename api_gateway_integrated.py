#!/usr/bin/env python3
"""
API Gateway Implementation for Agent Lightning - Integrated Services Version
Routes to the new integrated services with shared database
"""

import os
import asyncio
import aiohttp
import json
import time
import hashlib
import jwt
import redis
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from fastapi import FastAPI, Request, Response, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
from urllib.parse import urlparse
import yaml
from collections import defaultdict, deque
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Types of backend services"""
    AGENT_DESIGNER = "agent_designer"
    WORKFLOW_ENGINE = "workflow_engine"
    INTEGRATION_HUB = "integration_hub"
    AI_MODEL = "ai_model"
    AUTH = "auth"
    WEBSOCKET = "websocket"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"
    RANDOM = "random"


@dataclass
class ServiceEndpoint:
    """Backend service endpoint"""
    service_id: str
    service_type: ServiceType
    url: str
    health_check_path: str = "/health"
    weight: int = 1
    active: bool = True
    last_health_check: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    current_connections: int = 0
    
    def get_avg_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0
        return sum(self.response_times) / len(self.response_times)
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        total = self.success_count + self.error_count
        if total == 0:
            return 1.0
        return self.success_count / total


@dataclass
class RouteConfig:
    """API route configuration"""
    path_pattern: str
    service_type: ServiceType
    methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    auth_required: bool = True
    rate_limit: int = 100  # requests per minute
    cache_ttl: int = 0  # seconds, 0 = no cache
    transform_request: bool = False
    transform_response: bool = False
    timeout: int = 30  # seconds
    retry_count: int = 3


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_limit = 100  # requests per minute
        
    async def check_rate_limit(self, client_id: str, limit: int = None) -> bool:
        """Check if request is within rate limit"""
        limit = limit or self.default_limit
        key = f"rate_limit:{client_id}"
        
        try:
            current = self.redis.incr(key)
            if current == 1:
                self.redis.expire(key, 60)  # Reset after 1 minute
            
            return current <= limit
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error


class CircuitBreaker:
    """Circuit breaker for service protection"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures: Dict[str, int] = defaultdict(int)
        self.last_failure_time: Dict[str, datetime] = {}
        self.state: Dict[str, str] = defaultdict(lambda: "closed")  # closed, open, half-open
        
    def record_success(self, service_id: str):
        """Record successful request"""
        self.failures[service_id] = 0
        if self.state[service_id] == "half-open":
            self.state[service_id] = "closed"
            
    def record_failure(self, service_id: str):
        """Record failed request"""
        self.failures[service_id] += 1
        self.last_failure_time[service_id] = datetime.now()
        
        if self.failures[service_id] >= self.failure_threshold:
            self.state[service_id] = "open"
            
    def is_open(self, service_id: str) -> bool:
        """Check if circuit is open"""
        if self.state[service_id] == "open":
            # Check if recovery timeout has passed
            if service_id in self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time[service_id]).seconds
                if time_since_failure >= self.recovery_timeout:
                    self.state[service_id] = "half-open"
                    return False
            return True
        return False


class APIGateway:
    """Main API Gateway class"""
    
    def __init__(self):
        self.app = FastAPI(title="Agent Lightning API Gateway (Integrated)", version="2.0.0")
        
        # Service registry - UPDATED FOR INTEGRATED SERVICES
        self.services: Dict[ServiceType, List[ServiceEndpoint]] = {
            ServiceType.AGENT_DESIGNER: [
                ServiceEndpoint(
                    service_id="agent-designer-1",
                    service_type=ServiceType.AGENT_DESIGNER,
                    url="http://localhost:8102"  # Updated port
                )
            ],
            ServiceType.WORKFLOW_ENGINE: [
                ServiceEndpoint(
                    service_id="workflow-engine-1",
                    service_type=ServiceType.WORKFLOW_ENGINE,
                    url="http://localhost:8103"  # Updated port
                )
            ],
            ServiceType.INTEGRATION_HUB: [
                ServiceEndpoint(
                    service_id="integration-hub-1",
                    service_type=ServiceType.INTEGRATION_HUB,
                    url="http://localhost:8104"  # Updated port
                )
            ],
            ServiceType.AI_MODEL: [
                ServiceEndpoint(
                    service_id="ai-model-1",
                    service_type=ServiceType.AI_MODEL,
                    url="http://localhost:8105"  # Updated port
                )
            ],
            ServiceType.AUTH: [
                ServiceEndpoint(
                    service_id="auth-1",
                    service_type=ServiceType.AUTH,
                    url="http://localhost:8106"  # Updated port
                )
            ],
            ServiceType.WEBSOCKET: [
                ServiceEndpoint(
                    service_id="websocket-1",
                    service_type=ServiceType.WEBSOCKET,
                    url="http://localhost:8107"  # Updated port
                )
            ]
        }
        
        # Route configuration - Updated for dashboard compatibility
        self.routes = [
            # Agent Designer routes
            RouteConfig("/api/v1/agents", ServiceType.AGENT_DESIGNER),
            RouteConfig("/api/v2/agents", ServiceType.AGENT_DESIGNER),
            RouteConfig("/agents", ServiceType.AGENT_DESIGNER),
            RouteConfig("/tasks", ServiceType.AGENT_DESIGNER),  # Task status endpoint
            
            # Workflow Engine routes
            RouteConfig("/api/v1/workflows", ServiceType.WORKFLOW_ENGINE),
            RouteConfig("/workflows", ServiceType.WORKFLOW_ENGINE),
            RouteConfig("/api/v1/executions", ServiceType.WORKFLOW_ENGINE),
            RouteConfig("/executions", ServiceType.WORKFLOW_ENGINE),
            
            # Integration Hub routes
            RouteConfig("/api/v1/integrations", ServiceType.INTEGRATION_HUB),
            RouteConfig("/integrations", ServiceType.INTEGRATION_HUB),
            RouteConfig("/api/v1/webhooks", ServiceType.INTEGRATION_HUB),
            RouteConfig("/webhooks", ServiceType.INTEGRATION_HUB),
            
            # AI Model routes
            RouteConfig("/api/v1/models", ServiceType.AI_MODEL),
            RouteConfig("/api/v1/inference", ServiceType.AI_MODEL),
            RouteConfig("/models", ServiceType.AI_MODEL),
            RouteConfig("/inference", ServiceType.AI_MODEL),
            
            # Auth routes
            RouteConfig("/api/v1/auth", ServiceType.AUTH, auth_required=False),
            RouteConfig("/auth", ServiceType.AUTH, auth_required=False),
            
            # WebSocket info route
            RouteConfig("/api/v1/ws", ServiceType.WEBSOCKET),
            RouteConfig("/ws", ServiceType.WEBSOCKET),
        ]
        
        # Initialize components
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                password=os.getenv('REDIS_PASSWORD', 'redis_secure_password_123'),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Rate limiting disabled.")
            self.redis_client = None
            
        self.rate_limiter = RateLimiter(self.redis_client) if self.redis_client else None
        self.circuit_breaker = CircuitBreaker()
        self.load_balancer_index: Dict[ServiceType, int] = defaultdict(int)
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0,
            "requests_by_service": defaultdict(int),
            "errors_by_service": defaultdict(int)
        }
        
        self._setup_middleware()
        self._setup_routes()
        
    def _setup_middleware(self):
        """Setup middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            """Add process time header"""
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Gateway health check"""
            service_health = {}
            
            # Check all services
            for service_type, endpoints in self.services.items():
                for endpoint in endpoints:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"{endpoint.url}{endpoint.health_check_path}",
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as response:
                                endpoint.active = response.status == 200
                                endpoint.last_health_check = datetime.now()
                    except:
                        endpoint.active = False
                        
                    service_health[endpoint.service_id] = {
                        "active": endpoint.active,
                        "url": endpoint.url,
                        "last_check": endpoint.last_health_check.isoformat() if endpoint.last_health_check else None
                    }
                    
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": service_health,
                "metrics": self.metrics
            }
            
        @self.app.get("/metrics")
        async def get_metrics():
            """Get gateway metrics"""
            return self.metrics
            
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
        async def proxy_request(request: Request, path: str):
            """Proxy requests to backend services"""
            
            # Update metrics
            self.metrics["total_requests"] += 1
            
            # Find matching route
            service_type = None
            route_config = None
            
            # Normalize path
            full_path = f"/{path}" if not path.startswith("/") else path
            
            # Try exact match first
            for route in self.routes:
                if full_path.startswith(route.path_pattern):
                    service_type = route.service_type
                    route_config = route
                    break
            
            # If no match, try without leading slash
            if not service_type:
                for route in self.routes:
                    if path.startswith(route.path_pattern.lstrip("/")):
                        service_type = route.service_type
                        route_config = route
                        break
                        
            if not service_type:
                self.metrics["failed_requests"] += 1
                raise HTTPException(status_code=404, detail="Route not found")
                
            # Check rate limit
            if self.rate_limiter and route_config.auth_required:
                client_id = request.client.host
                if not await self.rate_limiter.check_rate_limit(client_id, route_config.rate_limit):
                    self.metrics["failed_requests"] += 1
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                    
            # Get backend service
            service_endpoint = self._get_service_endpoint(service_type)
            
            if not service_endpoint:
                self.metrics["failed_requests"] += 1
                self.metrics["errors_by_service"][service_type.value] += 1
                raise HTTPException(status_code=503, detail="Service unavailable")
                
            # Check circuit breaker
            if self.circuit_breaker.is_open(service_endpoint.service_id):
                self.metrics["failed_requests"] += 1
                self.metrics["errors_by_service"][service_type.value] += 1
                raise HTTPException(status_code=503, detail="Service circuit breaker open")
                
            # Proxy the request
            try:
                # Prepare request
                headers = dict(request.headers)
                headers.pop("host", None)
                
                # Get request body
                body = await request.body()
                
                # Make request to backend service
                service_endpoint.current_connections += 1
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=request.method,
                        url=f"{service_endpoint.url}/{path}",
                        headers=headers,
                        data=body,
                        timeout=aiohttp.ClientTimeout(total=route_config.timeout)
                    ) as response:
                        # Record response time
                        response_time = time.time() - start_time
                        service_endpoint.response_times.append(response_time)
                        
                        # Get response
                        response_body = await response.read()
                        
                        # Update metrics
                        if response.status < 400:
                            service_endpoint.success_count += 1
                            self.circuit_breaker.record_success(service_endpoint.service_id)
                            self.metrics["successful_requests"] += 1
                        else:
                            service_endpoint.error_count += 1
                            self.circuit_breaker.record_failure(service_endpoint.service_id)
                            self.metrics["failed_requests"] += 1
                            self.metrics["errors_by_service"][service_type.value] += 1
                            
                        service_endpoint.current_connections -= 1
                        self.metrics["requests_by_service"][service_type.value] += 1
                        
                        # Return response
                        return Response(
                            content=response_body,
                            status_code=response.status,
                            headers=dict(response.headers),
                            media_type=response.headers.get("content-type", "application/json")
                        )
                        
            except asyncio.TimeoutError:
                service_endpoint.current_connections -= 1
                service_endpoint.error_count += 1
                self.circuit_breaker.record_failure(service_endpoint.service_id)
                self.metrics["failed_requests"] += 1
                self.metrics["errors_by_service"][service_type.value] += 1
                raise HTTPException(status_code=504, detail="Gateway timeout")
                
            except Exception as e:
                service_endpoint.current_connections -= 1
                service_endpoint.error_count += 1
                self.circuit_breaker.record_failure(service_endpoint.service_id)
                self.metrics["failed_requests"] += 1
                self.metrics["errors_by_service"][service_type.value] += 1
                logger.error(f"Proxy error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
    def _get_service_endpoint(self, service_type: ServiceType) -> Optional[ServiceEndpoint]:
        """Get service endpoint using load balancing"""
        endpoints = self.services.get(service_type, [])
        active_endpoints = [e for e in endpoints if e.active]
        
        if not active_endpoints:
            return None
            
        # Round-robin load balancing
        index = self.load_balancer_index[service_type] % len(active_endpoints)
        self.load_balancer_index[service_type] += 1
        
        return active_endpoints[index]
        
    async def startup(self):
        """Startup tasks"""
        logger.info("API Gateway starting up...")
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
    async def _health_check_loop(self):
        """Periodic health check for services"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            for service_type, endpoints in self.services.items():
                for endpoint in endpoints:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"{endpoint.url}{endpoint.health_check_path}",
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as response:
                                endpoint.active = response.status == 200
                                endpoint.last_health_check = datetime.now()
                    except:
                        endpoint.active = False
                        
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("API Gateway shutting down...")
        if self.redis_client:
            self.redis_client.close()


def main():
    """Main entry point"""
    import uvicorn
    
    gateway = APIGateway()
    
    # Add lifecycle events
    gateway.app.add_event_handler("startup", gateway.startup)
    gateway.app.add_event_handler("shutdown", gateway.shutdown)
    
    print("\n" + "="*60)
    print("⚡ Agent Lightning API Gateway (Integrated)")
    print("="*60)
    print("\n🌐 Starting API Gateway on port 8000")
    print("\n📍 Endpoints:")
    print("  • http://localhost:8000/health - Health check")
    print("  • http://localhost:8000/metrics - Gateway metrics")
    print("  • http://localhost:8000/api/v1/* - API routes")
    print("\n🔄 Routing to Integrated Services:")
    print("  • Agent Designer:  http://localhost:8102")
    print("  • Workflow Engine: http://localhost:8103")
    print("  • Integration Hub: http://localhost:8104")
    print("  • AI Model:        http://localhost:8105")
    print("  • Auth Service:    http://localhost:8106")
    print("  • WebSocket:       http://localhost:8107")
    print("\n✨ Features:")
    print("  • Load balancing")
    print("  • Rate limiting")
    print("  • Circuit breaker")
    print("  • Request routing")
    print("  • Health monitoring")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(gateway.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()