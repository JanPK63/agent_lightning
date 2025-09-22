"""
Comprehensive Health Check System for Agent Lightning

Provides liveness, readiness, and deep health checks for all services.
Supports dependency checking, metrics collection, and monitoring integration.
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check"""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    response_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "response_time": self.response_time,
            "timestamp": self.timestamp
        }


@dataclass
class HealthCheckSummary:
    """Summary of all health checks"""
    status: HealthStatus
    total_checks: int
    healthy_checks: int
    unhealthy_checks: int
    degraded_checks: int
    response_time: float
    timestamp: float
    checks: List[HealthCheckResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "status": self.status.value,
            "total_checks": self.total_checks,
            "healthy_checks": self.healthy_checks,
            "unhealthy_checks": self.unhealthy_checks,
            "degraded_checks": self.degraded_checks,
            "response_time": self.response_time,
            "timestamp": self.timestamp,
            "checks": [check.to_dict() for check in self.checks]
        }


class HealthCheck:
    """Base class for health checks"""

    def __init__(self, name: str, description: str = "", timeout: float = 5.0):
        self.name = name
        self.description = description
        self.timeout = timeout

    async def check(self) -> HealthCheckResult:
        """Perform the health check"""
        raise NotImplementedError("Subclasses must implement check()")


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity"""

    def __init__(self, name: str = "database", db_type: str = "write"):
        super().__init__(name, f"Database connectivity check ({db_type})")
        self.db_type = db_type

    async def check(self) -> HealthCheckResult:
        start_time = time.time()

        try:
            from .database import db_manager

            # Test database connection
            if self.db_type == "read":
                # Test read database
                session = db_manager.get_read_db()
                # Simple query to test connectivity
                session.execute("SELECT 1")
                session.close()
            else:
                # Test write database
                session = db_manager.get_write_db()
                # Simple query to test connectivity
                session.execute("SELECT 1")
                session.close()

            response_time = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"{self.db_type.capitalize()} database connection successful",
                details={"db_type": self.db_type},
                response_time=response_time
            )

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.db_type.capitalize()} database connection failed: {str(e)}",
                details={"db_type": self.db_type, "error": str(e)},
                response_time=response_time
            )


class SystemHealthCheck(HealthCheck):
    """Health check for system resources"""

    def __init__(self, name: str = "system", cpu_threshold: float = 80.0, memory_threshold: float = 80.0):
        super().__init__(name, "System resource check")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold

    async def check(self) -> HealthCheckResult:
        start_time = time.time()

        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3)
            }

            # Determine status based on thresholds
            if cpu_percent > 95 or memory.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "System resources critically high"
            elif cpu_percent > self.cpu_threshold or memory.percent > self.memory_threshold:
                status = HealthStatus.DEGRADED
                message = "System resources elevated"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"

            response_time = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                response_time=response_time
            )

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"System health check failed: {str(e)}",
                details={"error": str(e)},
                response_time=response_time
            )


class ExternalServiceHealthCheck(HealthCheck):
    """Health check for external services"""

    def __init__(self, name: str, url: str, timeout: float = 5.0):
        super().__init__(name, f"External service check for {url}", timeout)
        self.url = url

    async def check(self) -> HealthCheckResult:
        start_time = time.time()

        try:
            import aiohttp

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        response_time = time.time() - start_time
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message=f"External service {self.url} is healthy",
                            details={"url": self.url, "response": data},
                            response_time=response_time
                        )
                    else:
                        response_time = time.time() - start_time
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"External service {self.url} returned status {response.status}",
                            details={"url": self.url, "status_code": response.status},
                            response_time=response_time
                        )

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"External service {self.url} check failed: {str(e)}",
                details={"url": self.url, "error": str(e)},
                response_time=response_time
            )


class HealthCheckService:
    """Centralized health check service"""

    def __init__(self, service_name: str = "unknown"):
        self.service_name = service_name
        self.checks: List[HealthCheck] = []
        self._last_check_time: Optional[float] = None
        self._last_result: Optional[HealthCheckSummary] = None

    def add_check(self, check: HealthCheck):
        """Add a health check"""
        self.checks.append(check)

    def add_database_check(self, db_type: str = "write"):
        """Add database health check"""
        self.add_check(DatabaseHealthCheck(f"database_{db_type}", db_type))

    def add_system_check(self, cpu_threshold: float = 80.0, memory_threshold: float = 80.0):
        """Add system resource health check"""
        self.add_check(SystemHealthCheck("system", cpu_threshold, memory_threshold))

    def add_external_service_check(self, name: str, url: str, timeout: float = 5.0):
        """Add external service health check"""
        self.add_check(ExternalServiceHealthCheck(name, url, timeout))

    async def run_checks(self, check_type: str = "all") -> HealthCheckSummary:
        """Run all health checks"""
        start_time = time.time()

        results = []
        healthy_count = 0
        unhealthy_count = 0
        degraded_count = 0

        # Run checks concurrently for better performance
        tasks = []
        for check in self.checks:
            if check_type == "all" or check_type in check.name:
                tasks.append(check.check())

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    # Handle exceptions in checks
                    results.append(HealthCheckResult(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed with exception: {str(result)}",
                        details={"error": str(result)}
                    ))
                    unhealthy_count += 1
                else:
                    results.append(result)
                    if result.status == HealthStatus.HEALTHY:
                        healthy_count += 1
                    elif result.status == HealthStatus.UNHEALTHY:
                        unhealthy_count += 1
                    elif result.status == HealthStatus.DEGRADED:
                        degraded_count += 1

        # Determine overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        elif healthy_count > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        response_time = time.time() - start_time

        summary = HealthCheckSummary(
            status=overall_status,
            total_checks=len(results),
            healthy_checks=healthy_count,
            unhealthy_checks=unhealthy_count,
            degraded_checks=degraded_count,
            response_time=response_time,
            timestamp=time.time(),
            checks=results
        )

        # Cache result
        self._last_result = summary
        self._last_check_time = time.time()

        return summary

    async def liveness_check(self) -> Dict[str, Any]:
        """Basic liveness check - service is running"""
        return {
            "status": "alive",
            "service": self.service_name,
            "timestamp": time.time()
        }

    async def readiness_check(self) -> Dict[str, Any]:
        """Readiness check - service is ready to accept traffic"""
        summary = await self.run_checks()
        return {
            "status": summary.status.value,
            "service": self.service_name,
            "ready": summary.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED],
            "timestamp": summary.timestamp,
            "checks": len(summary.checks)
        }

    async def deep_health_check(self) -> Dict[str, Any]:
        """Deep health check with detailed information"""
        summary = await self.run_checks()
        return summary.to_dict()

    def get_cached_result(self) -> Optional[HealthCheckSummary]:
        """Get the last cached health check result"""
        return self._last_result


# Global health check service instance
health_check_service = HealthCheckService("agent-lightning")


def init_health_checks(service_name: str = "agent-lightning"):
    """Initialize health checks for a service"""
    global health_check_service
    health_check_service = HealthCheckService(service_name)

    # Add default checks
    health_check_service.add_system_check()

    # Add database checks if available
    try:
        from .database import db_manager
        if db_manager.is_db_splitting_enabled():
            health_check_service.add_database_check("read")
            health_check_service.add_database_check("write")
        else:
            health_check_service.add_database_check("write")
    except ImportError:
        logger.warning("Database module not available for health checks")

    return health_check_service