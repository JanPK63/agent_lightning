# Comprehensive Health Check System

This document describes the comprehensive health check system implemented for Agent Lightning, providing liveness, readiness, and deep health checks for all services.

## Overview

The health check system provides three types of health probes:

1. **Liveness Probe** (`/health`) - Basic check if the service is running
2. **Readiness Probe** (`/health/ready`) - Check if the service is ready to accept traffic
3. **Deep Health Check** (`/health/deep`) - Detailed health check with dependency validation

## Architecture

### Health Check Service

The system is built around a centralized `HealthCheckService` class that:

- Manages multiple health checks
- Runs checks concurrently for performance
- Aggregates results with overall status
- Provides caching for repeated checks

### Health Check Types

#### 1. Database Health Check
- Tests database connectivity for read and write databases
- Validates SQLAlchemy session creation and basic queries
- Supports both single database and read/write splitting configurations

#### 2. System Health Check
- Monitors CPU and memory usage
- Checks disk space utilization
- Provides configurable thresholds for alerts

#### 3. External Service Health Check
- Tests connectivity to external APIs and services
- Validates HTTP responses and status codes
- Supports custom timeout and retry logic

## API Endpoints

### Agent Lightning Server

The main server provides three health check endpoints:

#### GET `/health` - Liveness Probe
**Purpose**: Basic liveness check for container orchestration systems (Kubernetes, Docker Swarm)

**Response**:
```json
{
  "status": "alive",
  "service": "agent-lightning-server",
  "timestamp": 1640995200.123
}
```

**Use Case**: Container orchestrators use this to determine if the service needs to be restarted.

#### GET `/health/ready` - Readiness Probe
**Purpose**: Check if the service is ready to accept traffic

**Response**:
```json
{
  "status": "healthy",
  "service": "agent-lightning-server",
  "ready": true,
  "checks": 3,
  "timestamp": 1640995200.123
}
```

**Use Case**: Load balancers and orchestrators use this to determine if the service should receive traffic.

#### GET `/health/deep` - Deep Health Check
**Purpose**: Comprehensive health check with detailed information about all dependencies

**Response**:
```json
{
  "status": "healthy",
  "total_checks": 3,
  "healthy_checks": 3,
  "unhealthy_checks": 0,
  "degraded_checks": 0,
  "response_time": 0.045,
  "timestamp": 1640995200.123,
  "checks": [
    {
      "name": "database_write",
      "status": "healthy",
      "message": "Write database connection successful",
      "details": {"db_type": "write"},
      "response_time": 0.012,
      "timestamp": 1640995200.123
    },
    {
      "name": "system",
      "status": "healthy",
      "message": "System resources normal",
      "details": {
        "cpu_percent": 45.2,
        "memory_percent": 62.1,
        "disk_percent": 73.8
      },
      "response_time": 0.033,
      "timestamp": 1640995200.123
    }
  ]
}
```

**Use Case**: Monitoring systems and administrators use this for detailed diagnostics.

## Configuration

### Environment Variables

```bash
# Health check thresholds
HEALTH_CHECK_CPU_THRESHOLD=80.0      # CPU usage threshold (%)
HEALTH_CHECK_MEMORY_THRESHOLD=80.0   # Memory usage threshold (%)

# External service checks
HEALTH_CHECK_EXTERNAL_SERVICES=api.example.com:8080,db.example.com:5432
```

### Programmatic Configuration

```python
from shared.health_check import init_health_checks

# Initialize health checks for a service
health_service = init_health_checks("my-service")

# Add custom checks
health_service.add_database_check("read")
health_service.add_database_check("write")
health_service.add_system_check(cpu_threshold=90.0, memory_threshold=85.0)
health_service.add_external_service_check("api", "http://api.example.com")
```

## Health Status Definitions

### Status Levels

- **HEALTHY**: All checks passed, service is fully operational
- **DEGRADED**: Some checks failed but service can still operate (e.g., high resource usage)
- **UNHEALTHY**: Critical checks failed, service may not function properly
- **UNKNOWN**: No checks configured or unable to determine status

### Check Results

Each health check returns:
- **Name**: Identifier for the check
- **Status**: HealthStatus enum value
- **Message**: Human-readable description
- **Details**: Structured data with metrics/details
- **Response Time**: Time taken to perform the check
- **Timestamp**: When the check was performed

## Usage Examples

### Kubernetes Configuration

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: agent-lightning
spec:
  containers:
  - name: server
    image: agent-lightning:latest
    ports:
    - containerPort: 8000
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 5
```

### Docker Compose Health Checks

```yaml
version: '3.8'
services:
  agent-lightning:
    image: agent-lightning:latest
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Monitoring Integration

```python
import requests
import time

def monitor_health():
    while True:
        try:
            response = requests.get("http://localhost:8000/health/deep")
            data = response.json()

            if data["status"] != "healthy":
                # Alert monitoring system
                alert_system.send_alert(f"Service unhealthy: {data['status']}")

            # Log metrics
            logger.info(f"Health check: {data['healthy_checks']}/{data['total_checks']} healthy")

        except Exception as e:
            logger.error(f"Health check failed: {e}")

        time.sleep(60)  # Check every minute
```

## Best Practices

### 1. Check Frequency
- **Liveness**: Every 10-30 seconds (quick checks)
- **Readiness**: Every 5-10 seconds (moderate checks)
- **Deep**: Every 1-5 minutes (comprehensive checks)

### 2. Timeout Configuration
- Set appropriate timeouts for external service checks
- Consider network latency and service response times
- Use timeouts that allow for graceful degradation

### 3. Resource Monitoring
- Set CPU/memory thresholds based on your infrastructure
- Monitor disk space to prevent storage issues
- Consider historical trends for threshold tuning

### 4. Alert Configuration
- Alert on UNHEALTHY status immediately
- Consider alerting on DEGRADED status for proactive monitoring
- Use different alert severities for different check types

### 5. Database Checks
- Use lightweight queries for health checks
- Avoid long-running operations
- Test both read and write connections when using database splitting

## Troubleshooting

### Common Issues

1. **Slow Health Checks**
   - Check external service timeouts
   - Review database query performance
   - Consider caching results for expensive checks

2. **False Positives**
   - Tune thresholds based on normal operating conditions
   - Account for peak usage periods
   - Use rolling averages for resource metrics

3. **Database Connection Issues**
   - Verify database credentials and connectivity
   - Check connection pool settings
   - Monitor database server resources

4. **External Service Failures**
   - Implement retry logic with backoff
   - Use circuit breaker patterns
   - Consider service mesh for resilience

### Debugging

Enable detailed logging:

```python
import logging
logging.getLogger('shared.health_check').setLevel(logging.DEBUG)
```

Check individual health check results:

```python
from shared.health_check import health_check_service
import asyncio

async def debug_checks():
    summary = await health_check_service.run_checks()
    for check in summary.checks:
        print(f"{check.name}: {check.status.value} - {check.message}")
        if check.details:
            print(f"  Details: {check.details}")
```

## Extensibility

### Custom Health Checks

Create custom health checks by extending the `HealthCheck` base class:

```python
from shared.health_check import HealthCheck, HealthStatus, HealthCheckResult

class CustomHealthCheck(HealthCheck):
    def __init__(self, name: str, custom_param: str):
        super().__init__(name, f"Custom check for {custom_param}")
        self.custom_param = custom_param

    async def check(self) -> HealthCheckResult:
        # Implement your custom health check logic
        try:
            # Your check logic here
            result = perform_custom_check(self.custom_param)

            if result.success:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Custom check passed",
                    details={"result": result.data}
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Custom check failed: {result.error}",
                    details={"error": result.error}
                )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Custom check error: {str(e)}",
                details={"exception": str(e)}
            )
```

### Integration with Monitoring Systems

The health check system integrates with popular monitoring solutions:

- **Prometheus**: Expose metrics via `/metrics` endpoint
- **Grafana**: Create dashboards based on health check data
- **ELK Stack**: Send health check logs for analysis
- **AlertManager**: Configure alerts based on health status

## Performance Considerations

- Health checks run concurrently to minimize response time
- Results are cached to reduce load on dependencies
- Lightweight checks are prioritized for liveness/readiness probes
- Deep checks can be resource-intensive but provide comprehensive diagnostics

## Security

- Health check endpoints should be accessible to monitoring systems
- Consider authentication for sensitive health information
- Avoid exposing sensitive configuration details in responses
- Use HTTPS for health check endpoints in production

## Future Enhancements

- **Metrics Export**: Integration with Prometheus metrics
- **Webhook Notifications**: Real-time alerts for status changes
- **Historical Tracking**: Store health check history for trend analysis
- **Auto-healing**: Automatic recovery actions for common issues
- **Distributed Health Checks**: Cross-service dependency validation