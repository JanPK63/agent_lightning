# Agent Lightning - Prometheus Metrics Implementation

## Overview

This document provides comprehensive documentation for the Prometheus metrics implementation in the Agent Lightning system. The metrics system provides observability, monitoring, and alerting capabilities for all services in the Agent Lightning platform.

## Architecture

### Components

1. **Metrics Collection** - Each service exposes Prometheus metrics via HTTP endpoints
2. **Prometheus Server** - Scrapes metrics from all services and stores time-series data
3. **Alerting Rules** - Defines conditions for generating alerts based on metrics
4. **Grafana Dashboards** - Visualizes metrics data for monitoring and analysis

### Service Groups

The system is organized into 6 service groups with metrics implemented:

1. **Agent Coordination Services**
   - `agent-coordinator` (port 8001)
   - `agent-designer` (port 8002)

2. **AI Model Services**
   - `ai-model-service` (port 8003)
   - `langchain-integration` (port 8004)

3. **Memory & Knowledge Services**
   - `memory-manager` (port 8005)
   - `knowledge-manager` (port 8006)

4. **Workflow & Orchestration Services**
   - `workflow-engine` (port 8007)
   - `rl-orchestrator` (port 8008)

5. **Monitoring & Dashboard Services**
   - `monitoring-dashboard` (port 8009)
   - `performance-metrics` (port 8010)

6. **Communication Services**
   - `websocket-service` (port 8011)
   - `event-replay-debugger` (port 8012)

## Metrics Implementation

### Shared Metrics Infrastructure

All services use a shared `MetricsCollector` class located in `monitoring/metrics.py`:

```python
from monitoring.metrics import MetricsCollector

# Initialize metrics collector
metrics_collector = MetricsCollector()

# Use in endpoints
@router.get("/health")
async def health_check():
    try:
        # Your health check logic
        result = {"status": "healthy"}
        metrics_collector.increment_request("health", "GET", "200")
        return result
    except Exception as e:
        metrics_collector.increment_error("health", "GET", str(type(e).__name__))
        raise
```

### Metrics Types

#### Request Metrics
- **Name**: `agent_lightning_requests_total`
- **Type**: Counter
- **Labels**: `method`, `endpoint`, `status`
- **Description**: Total number of HTTP requests processed

#### Error Metrics
- **Name**: `agent_lightning_requests_total` (with status codes 5xx)
- **Type**: Counter
- **Labels**: `method`, `endpoint`, `status`
- **Description**: Total number of error responses

#### Service Health Metrics
- **Name**: `up`
- **Type**: Gauge
- **Labels**: `job`
- **Description**: Service availability (1 = up, 0 = down)

#### System Metrics (automatically collected)
- **Name**: `process_cpu_user_seconds_total`
- **Type**: Counter
- **Description**: CPU usage in user mode

- **Name**: `process_resident_memory_bytes`
- **Type**: Gauge
- **Description**: Resident memory usage

## Configuration Files

### Prometheus Configuration

**File**: `monitoring/working_prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Service configurations for all 12 services
  - job_name: 'agent-coordinator'
    static_configs:
      - targets: ['host.docker.internal:8001']
        labels:
          service: 'agent-coordinator'
          component: 'coordination'
    scrape_interval: 10s
    metrics_path: /metrics
  # ... additional service configurations
```

### Alerting Rules

**File**: `monitoring/alert_rules.yml`

Contains 8 alerting rules covering:
- Service down detection
- High error rates (>10%)
- High memory usage (>90%)
- High CPU usage (>80%)
- WebSocket connection errors
- RL training stalls
- Memory retrieval errors
- Workflow execution errors

## Grafana Dashboards

### Main Dashboard

**File**: `monitoring/grafana/dashboards/agent-lightning-overview.json`

**Features**:
- System health overview with service status indicators
- Total requests by service (bar chart)
- Error rate by service (bar chart)
- Individual service sections with detailed metrics
- System resources monitoring (CPU and memory)

**Dashboard Structure**:
1. **System Overview** - Health status and aggregate metrics
2. **Agent Coordination Services** - Coordinator and designer metrics
3. **AI Model Services** - AI model and LangChain metrics
4. **Memory & Knowledge Services** - Memory and knowledge metrics
5. **Workflow & Orchestration Services** - Workflow and RL metrics
6. **Monitoring & Dashboard Services** - Dashboard and performance metrics
7. **Communication Services** - WebSocket and event replay metrics
8. **System Resources** - CPU and memory usage across services

## Testing and Validation

### Metrics Validation Script

**File**: `monitoring/test_metrics_validation.py`

**Capabilities**:
- Validates Prometheus server connectivity
- Tests all 12 services for health and metrics availability
- Generates traffic to create metrics data
- Validates alerting rules configuration
- Provides detailed reporting and recommendations

**Usage**:
```bash
cd monitoring
python test_metrics_validation.py
```

**Expected Output**:
- Service health status for all 12 services
- Metrics collection validation
- Alerting rules status
- Recommendations for issues found

## Deployment and Operations

### Prerequisites

1. **Python Environment**: Python 3.10+ with required dependencies
2. **Prometheus Server**: Version 2.40+ recommended
3. **Grafana**: Version 9.0+ recommended
4. **Alertmanager**: For alert notifications (optional)

### Service Startup

Each service must be started with metrics enabled:

```bash
# Example service startup
uvicorn services.agent_coordinator_service:app --host 0.0.0.0 --port 8001
```

### Prometheus Startup

```bash
# Start Prometheus with configuration
prometheus --config.file=monitoring/working_prometheus.yml
```

### Grafana Setup

1. Import dashboard from `monitoring/grafana/dashboards/agent-lightning-overview.json`
2. Configure Prometheus as data source
3. Set up alerting notifications (optional)

## Monitoring Best Practices

### Service Health Checks

- All services expose `/health` endpoints for health monitoring
- Health checks are used by Prometheus for service availability
- Failed health checks trigger alerts

### Metrics Naming Convention

- **Prefix**: `agent_lightning_` for all custom metrics
- **Suffix**: `_total` for counters
- **Labels**: Use consistent labels (`method`, `endpoint`, `status`, `job`)

### Alert Thresholds

- **Error Rate**: >10% over 5 minutes triggers warning
- **Memory Usage**: >90% over 10 minutes triggers warning
- **CPU Usage**: >80% over 10 minutes triggers warning
- **Service Down**: >5 minutes triggers critical alert

### Dashboard Organization

- **System Overview**: High-level health and performance
- **Service Groups**: Detailed metrics per service category
- **Resources**: System resource utilization
- **Alerts**: Active alerts and notifications

## Troubleshooting

### Common Issues

#### Services Not Appearing in Prometheus
- Check service is running and accessible on configured port
- Verify `/metrics` endpoint is responding
- Check Prometheus scrape target configuration

#### Missing Metrics Data
- Ensure `MetricsCollector` is properly initialized
- Verify metrics are being incremented in endpoint handlers
- Check service logs for metrics-related errors

#### Alerting Not Working
- Verify `alert_rules.yml` is in correct location
- Check Prometheus configuration references correct rules file
- Validate alerting rule syntax

#### Grafana Dashboard Issues
- Ensure Prometheus data source is correctly configured
- Check dashboard JSON for syntax errors
- Verify metric names match between services and dashboard

### Debugging Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Query specific metrics
curl "http://localhost:9090/api/v1/query?query=agent_lightning_requests_total"

# Check alerting rules
curl http://localhost:9090/api/v1/rules

# Validate service metrics endpoint
curl http://localhost:8001/metrics
```

## Maintenance and Updates

### Adding New Services

1. Add service configuration to `monitoring/working_prometheus.yml`
2. Implement metrics in service using `MetricsCollector`
3. Add service to validation script
4. Update Grafana dashboard if needed

### Updating Alert Rules

1. Modify `monitoring/alert_rules.yml`
2. Reload Prometheus configuration
3. Test alert conditions
4. Update documentation

### Dashboard Updates

1. Export current dashboard JSON
2. Modify dashboard configuration
3. Import updated dashboard to Grafana
4. Test all panels and queries

## Performance Considerations

### Metrics Overhead

- Metrics collection adds minimal overhead (<1% CPU)
- Memory usage scales with number of metrics
- Network traffic from Prometheus scraping is minimal

### Storage Requirements

- Prometheus storage grows with metrics volume
- Typical retention: 15-30 days
- Compression reduces storage by ~70%

### Scalability

- Supports up to 1000+ services with proper configuration
- Horizontal scaling possible with Prometheus federation
- Grafana can handle high query loads with caching

## Security Considerations

### Metrics Endpoint Security

- Metrics endpoints should not expose sensitive information
- Consider authentication for production deployments
- Use HTTPS for metrics scraping in production

### Network Security

- Prometheus scraping should use internal networks
- Grafana access should be properly authenticated
- Alert notifications should use secure channels

## Future Enhancements

### Planned Improvements

1. **Custom Metrics**: Add domain-specific metrics for RL training
2. **Distributed Tracing**: Integration with Jaeger/OpenTelemetry
3. **Anomaly Detection**: ML-based anomaly detection for metrics
4. **Automated Scaling**: Metrics-driven auto-scaling policies
5. **Advanced Alerting**: Multi-condition and time-based alerts

### Integration Opportunities

1. **Kubernetes Integration**: Service discovery and metrics collection
2. **Cloud Monitoring**: Integration with AWS CloudWatch/GCP Stackdriver
3. **Log Aggregation**: Correlation with ELK stack
4. **CI/CD Integration**: Automated metrics validation in pipelines

## Support and Resources

### Documentation Links

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)

### Community Resources

- [Prometheus Community](https://prometheus.io/community/)
- [Grafana Community](https://community.grafana.com/)
- [Agent Lightning Repository](https://github.com/agent-lightning)

### Contact

For issues or questions regarding the metrics implementation:
- **Team**: Agent Lightning Team
- **Documentation**: This document
- **Issues**: GitHub repository issues

---

**Last Updated**: 2025-09-19
**Version**: 1.0
**Authors**: Agent Lightning Team