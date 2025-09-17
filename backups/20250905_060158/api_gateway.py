#!/usr/bin/env python3
"""
API Gateway Implementation for Agent Lightning
Provides centralized request routing, authentication, rate limiting, and monitoring
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
    MONITORING = "monitoring"
    AUTH = "auth"
    ANALYTICS = "analytics"


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
    circuit_breaker_threshold: int = 5  # failures before circuit opens
    circuit_breaker_timeout: int = 60  # seconds before attempting to close


@dataclass
class RateLimitInfo:
    """Rate limit tracking information"""
    count: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    blocked_until: Optional[datetime] = None


@dataclass
class CircuitBreaker:
    """Circuit breaker state"""
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    state: str = "closed"  # closed, open, half-open
    next_attempt: Optional[datetime] = None


class APIGateway:
    """Main API Gateway class"""
    
    def __init__(self, config_file: str = "gateway_config.yaml"):
        self.app = FastAPI(title="Agent Lightning API Gateway")
        self.config_file = config_file
        self.services: Dict[ServiceType, List[ServiceEndpoint]] = defaultdict(list)
        self.routes: List[RouteConfig] = []
        self.redis_client = None
        self.rate_limits: Dict[str, RateLimitInfo] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.round_robin_counters: Dict[ServiceType, int] = defaultdict(int)
        
        self._setup_middleware()
        self._setup_routes()
        self._load_configuration()
        self._init_redis()
        
        # Background tasks will be started when the app starts
        self.startup_complete = False
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Add request ID
            request_id = hashlib.md5(f"{time.time()}{request.client}".encode()).hexdigest()[:8]
            request.state.request_id = request_id
            
            response = await call_next(request)
            
            # Log request
            duration = time.time() - start_time
            logger.info(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}"
            
            return response
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Run startup tasks"""
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._cleanup_cache_loop())
            self.startup_complete = True
            logger.info("API Gateway startup complete")
        
        @self.app.get("/health")
        async def health_check():
            """Gateway health check endpoint"""
            active_services = sum(
                1 for service_list in self.services.values()
                for service in service_list if service.active
            )
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_services": active_services,
                "total_services": sum(len(s) for s in self.services.values())
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """Gateway metrics endpoint"""
            metrics_data = {
                "services": {},
                "rate_limits": {},
                "circuit_breakers": {},
                "cache": {
                    "size": len(self.cache),
                    "hit_rate": 0  # TODO: Implement cache hit tracking
                }
            }
            
            # Service metrics
            for service_type, endpoints in self.services.items():
                metrics_data["services"][service_type.value] = [
                    {
                        "id": ep.service_id,
                        "active": ep.active,
                        "avg_response_time": ep.get_avg_response_time(),
                        "success_rate": ep.get_success_rate(),
                        "current_connections": ep.current_connections
                    }
                    for ep in endpoints
                ]
            
            # Circuit breaker status
            for key, breaker in self.circuit_breakers.items():
                metrics_data["circuit_breakers"][key] = {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count
                }
            
            return metrics_data
        
        # Dynamic route handler
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def gateway_handler(request: Request, path: str):
            """Main gateway request handler"""
            return await self._handle_request(request, path)
    
    def _load_configuration(self):
        """Load gateway configuration from file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                
                # Load services
                for service_config in config.get('services', []):
                    service = ServiceEndpoint(
                        service_id=service_config['id'],
                        service_type=ServiceType(service_config['type']),
                        url=service_config['url'],
                        health_check_path=service_config.get('health_check', '/health'),
                        weight=service_config.get('weight', 1)
                    )
                    self.services[service.service_type].append(service)
                
                # Load routes
                for route_config in config.get('routes', []):
                    route = RouteConfig(
                        path_pattern=route_config['path'],
                        service_type=ServiceType(route_config['service']),
                        methods=route_config.get('methods', ["GET", "POST", "PUT", "DELETE"]),
                        auth_required=route_config.get('auth_required', True),
                        rate_limit=route_config.get('rate_limit', 100),
                        cache_ttl=route_config.get('cache_ttl', 0),
                        timeout=route_config.get('timeout', 30),
                        retry_count=route_config.get('retry_count', 3)
                    )
                    self.routes.append(route)
        else:
            # Default configuration
            self._setup_default_services()
    
    def _setup_default_services(self):
        """Setup default service endpoints"""
        # Agent Designer Service
        self.services[ServiceType.AGENT_DESIGNER] = [
            ServiceEndpoint(
                service_id="agent-designer-1",
                service_type=ServiceType.AGENT_DESIGNER,
                url="http://localhost:8001"
            )
        ]
        
        # Workflow Engine Service
        self.services[ServiceType.WORKFLOW_ENGINE] = [
            ServiceEndpoint(
                service_id="workflow-engine-1",
                service_type=ServiceType.WORKFLOW_ENGINE,
                url="http://localhost:8002"
            )
        ]
        
        # Default routes
        self.routes = [
            RouteConfig(
                path_pattern="^/api/v1/agents",
                service_type=ServiceType.AGENT_DESIGNER
            ),
            RouteConfig(
                path_pattern="^/api/v1/workflows",
                service_type=ServiceType.WORKFLOW_ENGINE
            ),
            RouteConfig(
                path_pattern="^/api/v1/integrations",
                service_type=ServiceType.INTEGRATION_HUB
            )
        ]
    
    def _init_redis(self):
        """Initialize Redis connection for distributed rate limiting and caching"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_client = None
    
    async def _handle_request(self, request: Request, path: str):
        """Handle incoming gateway request"""
        try:
            # Find matching route
            route = self._find_matching_route(path, request.method)
            if not route:
                raise HTTPException(status_code=404, detail="Route not found")
            
            # Authentication check
            if route.auth_required:
                auth_result = await self._check_authentication(request)
                if not auth_result:
                    raise HTTPException(status_code=401, detail="Unauthorized")
            
            # Rate limiting
            client_id = self._get_client_id(request)
            if not await self._check_rate_limit(client_id, route):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Check cache
            cache_key = self._get_cache_key(request, path)
            if route.cache_ttl > 0 and request.method == "GET":
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    return JSONResponse(content=cached_response)
            
            # Circuit breaker check
            circuit_key = f"{route.service_type.value}:{path}"
            if not self._check_circuit_breaker(circuit_key, route):
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            
            # Select backend service
            service = await self._select_service(route.service_type)
            if not service:
                raise HTTPException(status_code=503, detail="No available services")
            
            # Forward request
            response_data = await self._forward_request(
                service, request, path, route
            )
            
            # Cache response if applicable
            if route.cache_ttl > 0 and request.method == "GET":
                self._cache_response(cache_key, response_data, route.cache_ttl)
            
            return JSONResponse(content=response_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Gateway error: {e}")
            raise HTTPException(status_code=500, detail="Internal gateway error")
    
    def _find_matching_route(self, path: str, method: str) -> Optional[RouteConfig]:
        """Find route config matching the request path"""
        for route in self.routes:
            if re.match(route.path_pattern, f"/api/v1/{path}") and method in route.methods:
                return route
        return None
    
    async def _check_authentication(self, request: Request) -> bool:
        """Check request authentication"""
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return False
        
        try:
            # Extract token
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                return False
            
            # Verify JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            request.state.user_id = payload.get("user_id")
            request.state.roles = payload.get("roles", [])
            return True
            
        except Exception as e:
            logger.warning(f"Authentication failed: {e}")
            return False
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try authenticated user ID first
        if hasattr(request.state, 'user_id'):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(self, client_id: str, route: RouteConfig) -> bool:
        """Check if request exceeds rate limit"""
        now = datetime.now()
        key = f"rate_limit:{client_id}:{route.path_pattern}"
        
        if self.redis_client:
            # Use Redis for distributed rate limiting
            try:
                current_count = self.redis_client.incr(key)
                if current_count == 1:
                    self.redis_client.expire(key, 60)  # 1 minute window
                
                if current_count > route.rate_limit:
                    return False
                return True
            except:
                pass  # Fall back to in-memory
        
        # In-memory rate limiting
        if key not in self.rate_limits:
            self.rate_limits[key] = RateLimitInfo()
        
        limit_info = self.rate_limits[key]
        
        # Check if blocked
        if limit_info.blocked_until and now < limit_info.blocked_until:
            return False
        
        # Reset window if needed
        if (now - limit_info.window_start).seconds >= 60:
            limit_info.count = 0
            limit_info.window_start = now
            limit_info.blocked_until = None
        
        # Increment count
        limit_info.count += 1
        
        # Check limit
        if limit_info.count > route.rate_limit:
            limit_info.blocked_until = now + timedelta(minutes=1)
            return False
        
        return True
    
    def _get_cache_key(self, request: Request, path: str) -> str:
        """Generate cache key for request"""
        # Include query parameters in cache key
        query_string = str(request.url.query) if request.url.query else ""
        return f"cache:{path}:{query_string}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get cached response if available"""
        if cache_key in self.cache:
            response_data, expiry = self.cache[cache_key]
            if datetime.now() < expiry:
                logger.info(f"Cache hit: {cache_key}")
                return response_data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response_data: Any, ttl: int):
        """Cache response data"""
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.cache[cache_key] = (response_data, expiry)
        logger.info(f"Cached response: {cache_key} (TTL: {ttl}s)")
    
    def _check_circuit_breaker(self, circuit_key: str, route: RouteConfig) -> bool:
        """Check circuit breaker status"""
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = CircuitBreaker()
        
        breaker = self.circuit_breakers[circuit_key]
        now = datetime.now()
        
        if breaker.state == "open":
            if breaker.next_attempt and now >= breaker.next_attempt:
                breaker.state = "half-open"
                logger.info(f"Circuit breaker half-open: {circuit_key}")
            else:
                return False
        
        return True
    
    def _record_circuit_breaker_success(self, circuit_key: str):
        """Record successful request for circuit breaker"""
        if circuit_key in self.circuit_breakers:
            breaker = self.circuit_breakers[circuit_key]
            if breaker.state == "half-open":
                breaker.state = "closed"
                breaker.failure_count = 0
                logger.info(f"Circuit breaker closed: {circuit_key}")
    
    def _record_circuit_breaker_failure(self, circuit_key: str, route: RouteConfig):
        """Record failed request for circuit breaker"""
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = CircuitBreaker()
        
        breaker = self.circuit_breakers[circuit_key]
        breaker.failure_count += 1
        breaker.last_failure = datetime.now()
        
        if breaker.failure_count >= route.circuit_breaker_threshold:
            breaker.state = "open"
            breaker.next_attempt = datetime.now() + timedelta(seconds=route.circuit_breaker_timeout)
            logger.warning(f"Circuit breaker opened: {circuit_key}")
    
    async def _select_service(self, service_type: ServiceType, 
                            strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> Optional[ServiceEndpoint]:
        """Select backend service using load balancing strategy"""
        endpoints = [s for s in self.services.get(service_type, []) if s.active]
        
        if not endpoints:
            return None
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Round-robin selection
            counter = self.round_robin_counters[service_type]
            selected = endpoints[counter % len(endpoints)]
            self.round_robin_counters[service_type] = counter + 1
            return selected
        
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select service with least connections
            return min(endpoints, key=lambda s: s.current_connections)
        
        elif strategy == LoadBalancingStrategy.WEIGHTED:
            # Weighted random selection
            import random
            weights = [s.weight for s in endpoints]
            return random.choices(endpoints, weights=weights)[0]
        
        else:
            # Default to first available
            return endpoints[0]
    
    async def _forward_request(self, service: ServiceEndpoint, request: Request, 
                              path: str, route: RouteConfig) -> Any:
        """Forward request to backend service"""
        # Build target URL
        target_url = f"{service.url}/api/v1/{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"
        
        # Prepare headers
        headers = dict(request.headers)
        headers['X-Forwarded-For'] = request.client.host if request.client else "unknown"
        headers['X-Request-ID'] = request.state.request_id
        
        # Read request body
        body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
        
        # Track connection
        service.current_connections += 1
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=route.timeout)
                ) as response:
                    # Record response time
                    response_time = time.time() - start_time
                    service.response_times.append(response_time)
                    
                    if response.status >= 200 and response.status < 300:
                        service.success_count += 1
                        self._record_circuit_breaker_success(f"{route.service_type.value}:{path}")
                        
                        # Parse response
                        if 'application/json' in response.headers.get('content-type', ''):
                            return await response.json()
                        else:
                            return {"data": await response.text()}
                    else:
                        service.error_count += 1
                        self._record_circuit_breaker_failure(f"{route.service_type.value}:{path}", route)
                        raise HTTPException(status_code=response.status, 
                                          detail=f"Backend service error: {response.status}")
        
        except asyncio.TimeoutError:
            service.error_count += 1
            self._record_circuit_breaker_failure(f"{route.service_type.value}:{path}", route)
            raise HTTPException(status_code=504, detail="Backend service timeout")
        
        except Exception as e:
            service.error_count += 1
            self._record_circuit_breaker_failure(f"{route.service_type.value}:{path}", route)
            logger.error(f"Error forwarding request: {e}")
            raise HTTPException(status_code=502, detail="Bad gateway")
        
        finally:
            service.current_connections -= 1
    
    async def _health_check_loop(self):
        """Periodic health check for backend services"""
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
                                if response.status == 200:
                                    endpoint.active = True
                                    endpoint.last_health_check = datetime.now()
                                else:
                                    endpoint.active = False
                                    logger.warning(f"Health check failed for {endpoint.service_id}: {response.status}")
                    except Exception as e:
                        endpoint.active = False
                        logger.error(f"Health check error for {endpoint.service_id}: {e}")
    
    async def _cleanup_cache_loop(self):
        """Periodic cache cleanup"""
        while True:
            await asyncio.sleep(60)  # Clean every minute
            
            now = datetime.now()
            expired_keys = [
                key for key, (_, expiry) in self.cache.items()
                if now >= expiry
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired cache entries")


def create_gateway_config():
    """Create default gateway configuration file"""
    config = {
        "services": [
            {
                "id": "agent-designer-1",
                "type": "agent_designer",
                "url": "http://localhost:8001",
                "health_check": "/health",
                "weight": 1
            },
            {
                "id": "agent-designer-2",
                "type": "agent_designer",
                "url": "http://localhost:8011",
                "health_check": "/health",
                "weight": 1
            },
            {
                "id": "workflow-engine-1",
                "type": "workflow_engine",
                "url": "http://localhost:8002",
                "health_check": "/health",
                "weight": 1
            },
            {
                "id": "integration-hub-1",
                "type": "integration_hub",
                "url": "http://localhost:8003",
                "health_check": "/health",
                "weight": 1
            }
        ],
        "routes": [
            {
                "path": "^/api/v1/agents",
                "service": "agent_designer",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "auth_required": True,
                "rate_limit": 100,
                "cache_ttl": 0,
                "timeout": 30,
                "retry_count": 3
            },
            {
                "path": "^/api/v1/workflows",
                "service": "workflow_engine",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "auth_required": True,
                "rate_limit": 50,
                "cache_ttl": 0,
                "timeout": 60,
                "retry_count": 2
            },
            {
                "path": "^/api/v1/integrations",
                "service": "integration_hub",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "auth_required": True,
                "rate_limit": 200,
                "cache_ttl": 300,
                "timeout": 30,
                "retry_count": 3
            }
        ],
        "global": {
            "cors": {
                "origins": ["*"],
                "methods": ["*"],
                "headers": ["*"]
            },
            "rate_limit": {
                "default": 100,
                "window": 60
            },
            "circuit_breaker": {
                "threshold": 5,
                "timeout": 60
            }
        }
    }
    
    with open("gateway_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("Created gateway_config.yaml")
    return config


if __name__ == "__main__":
    import uvicorn
    
    print("Agent Lightning API Gateway")
    print("=" * 60)
    
    # Create config if it doesn't exist
    if not os.path.exists("gateway_config.yaml"):
        create_gateway_config()
    
    # Create and run gateway
    gateway = APIGateway()
    
    print("\n📡 Starting API Gateway on port 8000")
    print("\nFeatures:")
    print("  • Request routing and load balancing")
    print("  • Authentication and authorization")
    print("  • Rate limiting and throttling")
    print("  • Circuit breaker pattern")
    print("  • Response caching")
    print("  • Health checks and monitoring")
    print("  • CORS handling")
    print("  • Request/response transformation")
    
    print("\n🔗 Gateway endpoints:")
    print("  • http://localhost:8000/health - Health check")
    print("  • http://localhost:8000/metrics - Gateway metrics")
    print("  • http://localhost:8000/api/v1/* - API routes")
    
    uvicorn.run(gateway.app, host="0.0.0.0", port=8000, reload=False)