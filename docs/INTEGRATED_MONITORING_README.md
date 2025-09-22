# Integrated Monitoring Dashboard Service

A unified monitoring service that consolidates all Agent Lightning monitoring microservices into a single, cohesive system. This service reduces overall file size by modularizing dependencies, optimizing code structure, and minimizing redundancy.

## Overview

The Integrated Monitoring Dashboard Service provides:

- **Unified Monitoring Interface**: Single entry point for all monitoring functionality
- **Service Discovery**: Automatic detection and registration of monitoring services
- **Metrics Aggregation**: Centralized collection and storage of metrics from all services
- **Alert Management**: Configurable alerting rules with escalation policies
- **Health Monitoring**: Real-time health checks for all dependent services
- **Dashboard Integration**: Web-based dashboard with real-time updates
- **API Gateway**: RESTful API for programmatic access to monitoring data
- **Configuration Management**: Centralized configuration with hot-reloading
- **Scalability**: Auto-scaling capabilities based on load and requirements

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Integrated Monitoring Service            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │Service      │ │Metrics      │ │Alert        │           │
│  │Discovery    │ │Aggregator   │ │Manager      │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │Health       │ │Dashboard    │ │API Gateway  │           │
│  │Monitor      │ │Renderer     │ │             │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                    ┌─────────────┐
                    │Web Dashboard│
                    │  (Streamlit) │
                    └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- FastAPI
- Uvicorn
- Streamlit (for dashboard)
- PostgreSQL (optional, for advanced features)

### Installation

1. **Clone and setup the project:**
   ```bash
   git clone <repository>
   cd agent-lightning-main
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the service:**
   ```bash
   cp config/integrated_monitoring_config.json config/my_config.json
   # Edit my_config.json with your settings
   ```

4. **Start the service:**
   ```bash
   # Using the deployment script
   ./scripts/deploy_integrated_monitoring.sh start

   # Or directly
   python services/integrated_monitoring_dashboard_service.py --config config/my_config.json
   ```

### Access Points

- **Main API**: `http://localhost:8051`
- **Dashboard**: `http://localhost:8051/dashboard` (redirects to Streamlit)
- **Streamlit Dashboard**: `http://localhost:8501`
- **Health Check**: `http://localhost:8051/health`
- **Metrics**: `http://localhost:8051/metrics`

## Configuration

The service uses a comprehensive JSON configuration file. Key sections:

### Server Configuration
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8051,
    "dashboard_port": 8501,
    "workers": 4,
    "reload": false
  }
}
```

### Monitoring Configuration
```json
{
  "monitoring": {
    "refresh_interval": 5,
    "max_data_points": 1000,
    "metrics_retention_hours": 24,
    "alert_cooldown_minutes": 5,
    "health_check_interval": 30
  }
}
```

### Alert Rules
```json
{
  "alerting": {
    "rules": {
      "cpu_high": {
        "name": "High CPU Usage",
        "condition": ">",
        "threshold": 90.0,
        "severity": "warning",
        "enabled": true
      }
    }
  }
}
```

## API Reference

### Service Management

#### Register a Service
```http
POST /api/v1/services/register
Content-Type: application/json

{
  "id": "agent-service-1",
  "name": "Web Developer Agent",
  "type": "agent",
  "url": "http://localhost:9001",
  "health_endpoint": "/health"
}
```

#### List Services
```http
GET /api/v1/services
```

Response:
```json
{
  "services": [
    {
      "id": "agent-service-1",
      "name": "Web Developer Agent",
      "type": "agent",
      "url": "http://localhost:9001",
      "status": "healthy",
      "last_heartbeat": "2025-09-22T05:00:00Z"
    }
  ],
  "count": 1
}
```

### Metrics

#### Get Metrics
```http
GET /api/v1/metrics
```

#### Get Dashboard Metrics
```http
GET /dashboard/metrics
```

### Alerts

#### Get Active Alerts
```http
GET /api/v1/alerts
```

#### Create Alert Rule
```http
POST /api/v1/alerts/rules
Content-Type: application/json

{
  "id": "memory_high",
  "name": "High Memory Usage",
  "condition": ">",
  "threshold": 85.0,
  "severity": "warning",
  "enabled": true,
  "cooldown_minutes": 10
}
```

### Health Monitoring

#### Overall Health
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "total_services": 5,
  "healthy_services": 5,
  "timestamp": "2025-09-22T05:00:00Z",
  "service_health": {
    "agent-service-1": {
      "healthy": true,
      "response_time": 0.15,
      "status_code": 200
    }
  }
}
```

## Dashboard Features

### Main Dashboard
- Real-time system health overview
- Active services count
- Alert summary
- System resource usage

### Training Metrics
- Loss curves over time
- Reward progression
- Accuracy trends
- Learning rate monitoring

### Agent Performance
- Agent status grid
- Performance comparison charts
- Response time distribution
- Task completion heatmaps

### System Resources
- CPU, Memory, GPU usage gauges
- Resource usage timelines
- Database connection pool metrics

### Task Assignment
- Interactive task creation and assignment
- Agent capability matching
- Task progress tracking
- Feedback collection

### Agent Knowledge
- Knowledge base management
- Search and retrieval
- Usage statistics
- Import/export functionality

### Project Configuration
- Multi-project management
- Deployment target configuration
- Directory structure setup
- Tech stack management

### Visual Code Builder
- Drag-and-drop code construction
- Multiple language support
- Code validation and generation
- Visual debugging

### Spec-Driven Development
- Specification creation and management
- Implementation planning
- Progress tracking
- GitHub Spec-Kit integration

## Deployment Options

### Local Development
```bash
./scripts/deploy_integrated_monitoring.sh start
```

### Docker Deployment
```bash
# Build the image
docker build -f deployments/Dockerfile.integrated_monitoring -t integrated-monitoring .

# Run the container
docker run -p 8051:8051 -p 8501:8501 -v $(pwd)/config:/app/config integrated-monitoring
```

### Docker Compose
```yaml
version: '3.8'
services:
  integrated-monitoring:
    build:
      context: .
      dockerfile: deployments/Dockerfile.integrated_monitoring
    ports:
      - "8051:8051"
      - "8501:8501"
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
```

### Production Deployment
```bash
# Using systemd
sudo cp deployments/integrated-monitoring.service /etc/systemd/system/
sudo systemctl enable integrated-monitoring
sudo systemctl start integrated-monitoring

# Using supervisor
pip install supervisor
sudo cp deployments/supervisord.conf /etc/supervisor/conf.d/
sudo supervisorctl reread
sudo supervisorctl update
```

## Monitoring and Maintenance

### Health Checks
The service provides comprehensive health monitoring:

- **Service Health**: Individual service status and response times
- **System Health**: CPU, memory, disk usage
- **Dependency Health**: Database connections, external services
- **Alert Health**: Alert rule evaluation and notification delivery

### Logging
Logs are written to `logs/integrated_monitoring.log` with rotation:

```
2025-09-22 05:00:00,123 - INFO - Integrated Monitoring Service initialized
2025-09-22 05:00:05,456 - INFO - Registered service: agent-service-1
2025-09-22 05:00:10,789 - WARNING - Alert triggered: High CPU Usage
```

### Metrics Collection
Prometheus-compatible metrics are available at `/metrics`:

```
# HELP monitoring_services_total Total number of registered services
# TYPE monitoring_services_total gauge
monitoring_services_total 5

# HELP monitoring_alerts_active Number of active alerts
# TYPE monitoring_alerts_active gauge
monitoring_alerts_active 2
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check Python version
python3 --version

# Check dependencies
pip list | grep fastapi

# Check configuration
python3 -c "import json; json.load(open('config/integrated_monitoring_config.json'))"
```

#### Dashboard Not Loading
```bash
# Check if Streamlit is running
ps aux | grep streamlit

# Check dashboard port
netstat -tlnp | grep 8501

# Check logs
tail -f logs/integrated_monitoring.log
```

#### Services Not Registering
```bash
# Check service URLs
curl http://localhost:9001/health

# Check network connectivity
telnet localhost 9001

# Check service logs
tail -f logs/agent_service_1.log
```

### Performance Tuning

#### Memory Optimization
```json
{
  "monitoring": {
    "max_data_points": 500,
    "metrics_retention_hours": 12
  }
}
```

#### CPU Optimization
```json
{
  "server": {
    "workers": 2
  },
  "monitoring": {
    "refresh_interval": 10
  }
}
```

## Integration Examples

### Python Client
```python
import requests

# Register a service
response = requests.post("http://localhost:8051/api/v1/services/register", json={
    "id": "my-service",
    "name": "My Custom Service",
    "type": "custom",
    "url": "http://localhost:9005"
})

# Get metrics
metrics = requests.get("http://localhost:8051/api/v1/metrics").json()

# Check health
health = requests.get("http://localhost:8051/health").json()
```

### JavaScript Client
```javascript
// Register service
fetch('http://localhost:8051/api/v1/services/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        id: 'frontend-service',
        name: 'Frontend Service',
        type: 'frontend',
        url: 'http://localhost:3000'
    })
});

// Get real-time updates
const ws = new WebSocket('ws://localhost:8051/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
black . && ruff .

# Start in development mode
./scripts/deploy_integrated_monitoring.sh setup
PYTHONPATH=. python services/integrated_monitoring_dashboard_service.py --reload
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: See `/docs` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@agent-lightning.dev

---

**Version**: 1.0.0
**Last Updated**: 2025-09-22
**Maintainer**: Agent Lightning Team