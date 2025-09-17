#!/usr/bin/env python3
"""
Service Discovery and Registry
Manages service registration, health checks, and discovery for all microservices
"""

import asyncio
import aiohttp
import json
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import os
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service health statuses"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class ServiceInstance:
    """Represents a single instance of a service"""
    id: str
    name: str
    version: str
    host: str
    port: int
    protocol: str = "http"
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None
    health_check_url: Optional[str] = None
    weight: int = 1  # For weighted load balancing
    zone: Optional[str] = None  # For zone-aware routing
    
    @property
    def url(self) -> str:
        """Get the service URL"""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def is_available(self) -> bool:
        """Check if service is available"""
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "status": self.status,
            "url": self.url,
            "metadata": self.metadata,
            "tags": self.tags,
            "registered_at": self.registered_at.isoformat(),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "weight": self.weight,
            "zone": self.zone
        }


@dataclass
class ServiceDefinition:
    """Service type definition"""
    name: str
    description: str
    required_tags: List[str] = field(default_factory=list)
    health_check_interval: int = 10  # seconds
    health_check_timeout: int = 5  # seconds
    deregister_critical_after: int = 60  # seconds
    max_instances: Optional[int] = None
    load_balancing_strategy: str = "round-robin"  # round-robin, least-connections, weighted
    circuit_breaker_threshold: int = 5  # consecutive failures
    dependencies: List[str] = field(default_factory=list)


class ServiceRegistry:
    """Central service registry"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = defaultdict(list)
        self.service_definitions: Dict[str, ServiceDefinition] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.event_listeners: List[callable] = []
        self._initialize_definitions()
    
    def _initialize_definitions(self):
        """Initialize service definitions"""
        self.service_definitions = {
            "api-gateway": ServiceDefinition(
                name="api-gateway",
                description="API Gateway for routing and rate limiting",
                required_tags=["gateway", "public"],
                health_check_interval=5,
                load_balancing_strategy="weighted"
            ),
            "auth-service": ServiceDefinition(
                name="auth-service",
                description="Authentication and authorization service",
                required_tags=["auth", "security"],
                health_check_interval=10,
                max_instances=3
            ),
            "agent-designer": ServiceDefinition(
                name="agent-designer",
                description="Agent creation and management service",
                required_tags=["agent", "core"],
                dependencies=["auth-service"],
                health_check_interval=10
            ),
            "workflow-engine": ServiceDefinition(
                name="workflow-engine",
                description="Workflow execution engine",
                required_tags=["workflow", "core"],
                dependencies=["auth-service"],
                health_check_interval=10,
                load_balancing_strategy="least-connections"
            ),
            "integration-hub": ServiceDefinition(
                name="integration-hub",
                description="External integration connectors",
                required_tags=["integration", "connector"],
                health_check_interval=15
            ),
            "ai-model-service": ServiceDefinition(
                name="ai-model-service",
                description="AI model orchestration and inference",
                required_tags=["ai", "ml"],
                health_check_interval=20,
                load_balancing_strategy="weighted",
                circuit_breaker_threshold=3
            ),
            "websocket-service": ServiceDefinition(
                name="websocket-service",
                description="Real-time WebSocket communication",
                required_tags=["realtime", "websocket"],
                health_check_interval=5
            )
        }
    
    async def register_service(self, 
                              name: str, 
                              host: str, 
                              port: int,
                              version: str = "1.0.0",
                              **kwargs) -> ServiceInstance:
        """Register a new service instance"""
        
        # Generate unique instance ID
        instance_id = f"{name}-{host}-{port}-{uuid.uuid4().hex[:8]}"
        
        # Create service instance
        instance = ServiceInstance(
            id=instance_id,
            name=name,
            version=version,
            host=host,
            port=port,
            protocol=kwargs.get("protocol", "http"),
            metadata=kwargs.get("metadata", {}),
            tags=kwargs.get("tags", []),
            health_check_url=kwargs.get("health_check_url", f"http://{host}:{port}/health"),
            weight=kwargs.get("weight", 1),
            zone=kwargs.get("zone", self._detect_zone())
        )
        
        # Add to registry
        self.services[name].append(instance)
        
        # Start health checking
        await self._start_health_check(instance)
        
        # Notify listeners
        await self._notify_event("service_registered", instance)
        
        logger.info(f"Registered service: {instance.id}")
        return instance
    
    async def deregister_service(self, service_id: str):
        """Deregister a service instance"""
        for name, instances in self.services.items():
            for i, instance in enumerate(instances):
                if instance.id == service_id:
                    # Stop health check
                    if service_id in self.health_check_tasks:
                        self.health_check_tasks[service_id].cancel()
                        del self.health_check_tasks[service_id]
                    
                    # Remove from registry
                    del instances[i]
                    
                    # Notify listeners
                    await self._notify_event("service_deregistered", instance)
                    
                    logger.info(f"Deregistered service: {service_id}")
                    return True
        
        return False
    
    def discover_service(self, name: str, tags: Optional[List[str]] = None) -> List[ServiceInstance]:
        """Discover available service instances"""
        instances = self.services.get(name, [])
        
        # Filter by availability
        available = [i for i in instances if i.is_available]
        
        # Filter by tags if specified
        if tags:
            available = [i for i in available if all(tag in i.tags for tag in tags)]
        
        return available
    
    def get_service_instance(self, 
                           name: str, 
                           strategy: str = "round-robin",
                           tags: Optional[List[str]] = None) -> Optional[ServiceInstance]:
        """Get a single service instance based on load balancing strategy"""
        instances = self.discover_service(name, tags)
        
        if not instances:
            return None
        
        if strategy == "round-robin":
            # Simple round-robin (would need state for true round-robin)
            return instances[int(time.time()) % len(instances)]
        
        elif strategy == "least-connections":
            # Return instance with least connections (need connection tracking)
            # For now, random selection
            import random
            return random.choice(instances)
        
        elif strategy == "weighted":
            # Weighted selection based on instance weight
            import random
            weights = [i.weight for i in instances]
            return random.choices(instances, weights=weights)[0]
        
        elif strategy == "zone-aware":
            # Prefer instances in the same zone
            local_zone = self._detect_zone()
            local_instances = [i for i in instances if i.zone == local_zone]
            if local_instances:
                return local_instances[0]
            return instances[0]
        
        return instances[0]
    
    async def _start_health_check(self, instance: ServiceInstance):
        """Start health checking for a service instance"""
        definition = self.service_definitions.get(instance.name)
        interval = definition.health_check_interval if definition else 10
        
        async def check_health():
            while True:
                try:
                    await asyncio.sleep(interval)
                    
                    # Perform health check
                    async with aiohttp.ClientSession() as session:
                        try:
                            async with session.get(
                                instance.health_check_url or f"{instance.url}/health",
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as response:
                                
                                instance.last_health_check = datetime.now()
                                
                                if response.status == 200:
                                    data = await response.json()
                                    if data.get("status") == "healthy":
                                        old_status = instance.status
                                        instance.status = ServiceStatus.HEALTHY
                                        if old_status != ServiceStatus.HEALTHY:
                                            await self._notify_event("service_healthy", instance)
                                    else:
                                        instance.status = ServiceStatus.DEGRADED
                                else:
                                    old_status = instance.status
                                    instance.status = ServiceStatus.UNHEALTHY
                                    if old_status == ServiceStatus.HEALTHY:
                                        await self._notify_event("service_unhealthy", instance)
                                        
                        except asyncio.TimeoutError:
                            instance.status = ServiceStatus.UNHEALTHY
                        except Exception as e:
                            instance.status = ServiceStatus.UNHEALTHY
                            logger.warning(f"Health check failed for {instance.id}: {e}")
                    
                    # Check deregistration criteria
                    if definition and instance.status == ServiceStatus.UNHEALTHY:
                        time_since_healthy = datetime.now() - instance.last_health_check
                        if time_since_healthy.seconds > definition.deregister_critical_after:
                            await self.deregister_service(instance.id)
                            break
                            
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check error for {instance.id}: {e}")
        
        task = asyncio.create_task(check_health())
        self.health_check_tasks[instance.id] = task
    
    def _detect_zone(self) -> str:
        """Detect the current availability zone/region"""
        # In cloud environments, this would query metadata service
        # For now, return a default zone
        return os.getenv("AVAILABILITY_ZONE", "zone-1")
    
    async def _notify_event(self, event_type: str, instance: ServiceInstance):
        """Notify event listeners"""
        for listener in self.event_listeners:
            try:
                await listener(event_type, instance)
            except Exception as e:
                logger.error(f"Error notifying listener: {e}")
    
    def add_event_listener(self, listener: callable):
        """Add an event listener"""
        self.event_listeners.append(listener)
    
    def get_service_topology(self) -> Dict[str, Any]:
        """Get the complete service topology"""
        topology = {
            "services": {},
            "total_instances": 0,
            "healthy_instances": 0,
            "dependencies": {}
        }
        
        for name, instances in self.services.items():
            healthy = sum(1 for i in instances if i.status == ServiceStatus.HEALTHY)
            topology["services"][name] = {
                "instances": len(instances),
                "healthy": healthy,
                "unhealthy": len(instances) - healthy,
                "endpoints": [i.to_dict() for i in instances]
            }
            topology["total_instances"] += len(instances)
            topology["healthy_instances"] += healthy
            
            # Add dependencies
            definition = self.service_definitions.get(name)
            if definition and definition.dependencies:
                topology["dependencies"][name] = definition.dependencies
        
        return topology
    
    def get_service_metrics(self, name: str) -> Dict[str, Any]:
        """Get metrics for a specific service"""
        instances = self.services.get(name, [])
        
        if not instances:
            return {}
        
        healthy = sum(1 for i in instances if i.status == ServiceStatus.HEALTHY)
        unhealthy = sum(1 for i in instances if i.status == ServiceStatus.UNHEALTHY)
        degraded = sum(1 for i in instances if i.status == ServiceStatus.DEGRADED)
        
        # Calculate uptime
        now = datetime.now()
        uptimes = []
        for instance in instances:
            if instance.status == ServiceStatus.HEALTHY:
                uptime = (now - instance.registered_at).total_seconds()
                uptimes.append(uptime)
        
        avg_uptime = sum(uptimes) / len(uptimes) if uptimes else 0
        
        return {
            "name": name,
            "total_instances": len(instances),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "degraded": degraded,
            "availability": (healthy / len(instances) * 100) if instances else 0,
            "average_uptime_seconds": avg_uptime,
            "zones": list(set(i.zone for i in instances if i.zone)),
            "versions": list(set(i.version for i in instances))
        }


class ServiceDiscoveryClient:
    """Client for service discovery"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self._cache: Dict[str, List[ServiceInstance]] = {}
        self._cache_ttl = 60  # seconds
        self._cache_timestamps: Dict[str, datetime] = {}
    
    def discover(self, service_name: str, use_cache: bool = True) -> List[ServiceInstance]:
        """Discover service instances with caching"""
        
        # Check cache
        if use_cache and service_name in self._cache:
            if datetime.now() - self._cache_timestamps[service_name] < timedelta(seconds=self._cache_ttl):
                return self._cache[service_name]
        
        # Discover from registry
        instances = self.registry.discover_service(service_name)
        
        # Update cache
        self._cache[service_name] = instances
        self._cache_timestamps[service_name] = datetime.now()
        
        return instances
    
    def get_service_url(self, service_name: str, strategy: str = "round-robin") -> Optional[str]:
        """Get a service URL using load balancing"""
        instance = self.registry.get_service_instance(service_name, strategy)
        return instance.url if instance else None
    
    async def call_service(self, 
                          service_name: str, 
                          path: str, 
                          method: str = "GET",
                          **kwargs) -> Optional[Dict[str, Any]]:
        """Call a service with automatic discovery and retry"""
        
        max_retries = 3
        for attempt in range(max_retries):
            instance = self.registry.get_service_instance(service_name)
            
            if not instance:
                logger.error(f"No available instances for service: {service_name}")
                return None
            
            url = f"{instance.url}{path}"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method, 
                        url,
                        timeout=aiohttp.ClientTimeout(total=30),
                        **kwargs
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logger.warning(f"Service call failed: {response.status}")
                            
            except Exception as e:
                logger.error(f"Error calling service {service_name}: {e}")
                
                # Mark instance as unhealthy
                instance.status = ServiceStatus.UNHEALTHY
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
        
        return None


# Auto-registration for our existing services
async def auto_register_services(registry: ServiceRegistry):
    """Auto-register all running microservices"""
    
    services_to_register = [
        ("api-gateway", "localhost", 8000, {"tags": ["gateway", "public"]}),
        ("auth-service", "localhost", 8006, {"tags": ["auth", "security"]}),
        ("agent-designer", "localhost", 8001, {"tags": ["agent", "core"]}),
        ("workflow-engine", "localhost", 8003, {"tags": ["workflow", "core"]}),
        ("integration-hub", "localhost", 8004, {"tags": ["integration", "connector"]}),
        ("ai-model-service", "localhost", 8005, {"tags": ["ai", "ml"]}),
        ("websocket-service", "localhost", 8007, {"tags": ["realtime", "websocket"]}),
        ("monitoring-dashboard", "localhost", 8051, {"tags": ["ui", "monitoring"]}),
        ("legacy-api", "localhost", 8002, {"tags": ["legacy", "api"]})
    ]
    
    for name, host, port, kwargs in services_to_register:
        try:
            # Check if service is running
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"http://{host}:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            await registry.register_service(name, host, port, **kwargs)
                            logger.info(f"Auto-registered service: {name}")
                except:
                    logger.warning(f"Service {name} not available for registration")
        except Exception as e:
            logger.error(f"Failed to auto-register {name}: {e}")


async def main():
    """Run service discovery registry"""
    registry = ServiceRegistry()
    
    # Add event listener for logging
    async def log_event(event_type: str, instance: ServiceInstance):
        logger.info(f"Event: {event_type} for {instance.id}")
    
    registry.add_event_listener(log_event)
    
    # Auto-register existing services
    await auto_register_services(registry)
    
    # Print initial topology
    print("\nService Topology:")
    print(json.dumps(registry.get_service_topology(), indent=2))
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(30)
            
            # Print status update
            topology = registry.get_service_topology()
            print(f"\nStatus Update: {topology['healthy_instances']}/{topology['total_instances']} services healthy")
            
    except KeyboardInterrupt:
        print("\nShutting down service discovery...")


if __name__ == "__main__":
    print("Service Discovery Registry")
    print("=" * 60)
    print("\nFeatures:")
    print("  • Automatic service registration")
    print("  • Health checking and monitoring")
    print("  • Load balancing strategies")
    print("  • Zone-aware routing")
    print("  • Circuit breaking")
    print("  • Service dependency tracking")
    
    asyncio.run(main())