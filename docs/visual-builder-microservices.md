# Visual Builder Microservices Architecture

## Overview

The Agent Lightning Visual Builder has been refactored from a monolithic service into a microservices architecture to improve scalability, maintainability, and deployment flexibility.

## Architecture

### Services Overview

```
┌─────────────────┐    ┌─────────────────────┐
│   API Gateway   │────│ Workflow Engine     │
│   (Port 8006)   │    │ (Port 8007)         │
└─────────────────┘    └─────────────────────┘
          │                       │
          ├───────────────────────┼─────────────────────┐
          │                       │                     │
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│ Component       │    │ Code Generator      │    │ Debugger        │
│ Registry        │    │ (Port 8009)         │    │ (Port 8010)     │
│ (Port 8008)     │    └─────────────────────┘    └─────────────────┘
└─────────────────┘              │                     │
          │                      │                     │
          └──────────────────────┼─────────────────────┘
                                 │
                    ┌─────────────────────┐    ┌─────────────────┐
                    │ Deployment          │    │ AI Assistant    │
                    │ (Port 8011)         │    │ (Port 8012)     │
                    └─────────────────────┘    └─────────────────┘
```

### Service Responsibilities

#### 1. API Gateway (`visual_builder_service_integrated.py`)
- **Port**: 8006
- **Purpose**: Unified entry point for all visual builder operations
- **Responsibilities**:
  - Route requests to appropriate microservices
  - Aggregate health checks from all services
  - Maintain backward compatibility with existing clients
  - Handle cross-cutting concerns (CORS, logging, etc.)

#### 2. Workflow Engine (`visual_workflow_engine_service.py`)
- **Port**: 8007
- **Purpose**: Core workflow management and execution
- **Responsibilities**:
  - Project CRUD operations
  - Component and connection management
  - Workflow validation and execution
  - Real-time collaboration via WebSockets

#### 3. Component Registry (`visual_component_registry_service.py`)
- **Port**: 8008
- **Purpose**: Component and template management
- **Responsibilities**:
  - Store and retrieve visual components
  - Manage project templates
  - Component library administration
  - Template CRUD operations

#### 4. Code Generator (`visual_code_generator_service.py`)
- **Port**: 8009
- **Purpose**: Generate code from visual projects
- **Responsibilities**:
  - Translate visual workflows to executable code
  - Support multiple programming languages
  - Code optimization and formatting
  - Generated code storage and retrieval

#### 5. Debugger (`visual_debugger_service.py`)
- **Port**: 8010
- **Purpose**: Visual workflow debugging
- **Responsibilities**:
  - Step-through execution
  - Breakpoint management
  - Variable inspection
  - Real-time debugging sessions

#### 6. Deployment Service (`visual_deployment_service.py`)
- **Port**: 8011
- **Purpose**: Deployment configuration and execution
- **Responsibilities**:
  - Generate deployment configurations
  - Manage deployment lifecycle
  - Support multiple deployment targets
  - Deployment status tracking

#### 7. AI Assistant (`visual_ai_assistant_service.py`)
- **Port**: 8012
- **Purpose**: AI-powered development assistance
- **Responsibilities**:
  - Generate code suggestions
  - Provide best practices recommendations
  - Code optimization suggestions
  - Context-aware assistance

## API Specifications

### Common Response Format

All services follow a consistent response format:

```json
{
  "status": "success|error",
  "data": { ... },
  "message": "Optional message",
  "timestamp": "ISO 8601 timestamp"
}
```

### Health Check Endpoint

All services expose a `/health` endpoint:

```http
GET /health
```

Response:
```json
{
  "service": "service_name",
  "status": "healthy|degraded|unhealthy",
  "version": "1.0.0",
  "timestamp": "2025-09-18T06:00:00.000Z"
}
```

### API Gateway Endpoints

The API Gateway maintains backward compatibility by proxying requests to appropriate services:

#### Projects
- `POST /projects` → Workflow Engine
- `GET /projects` → Workflow Engine
- `GET /projects/{id}` → Workflow Engine
- `DELETE /projects/{id}` → Workflow Engine

#### Components
- `POST /components/add` → Workflow Engine
- `POST /connections/create` → Workflow Engine
- `GET /components/library` → Component Registry

#### Code Generation
- `POST /generate/code` → Code Generator
- `GET /download/{code_id}` → Code Generator

#### Workflow Execution
- `POST /workflows/execute` → Workflow Engine

#### Debugging
- `POST /debug` → Debugger
- `POST /debug/step` → Debugger
- `GET /debug/{session_id}` → Debugger

#### Deployment
- `POST /deploy` → Deployment Service
- `GET /deployments/{deployment_id}` → Deployment Service

#### AI Assistance
- `POST /ai/suggest` → AI Assistant
- `POST /ai/optimize` → AI Assistant

## Inter-Service Communication

### Event-Driven Architecture

Services communicate asynchronously through the shared event bus:

```python
from shared.events import EventChannel

# Emit event
dal.event_bus.emit(EventChannel.PROJECT_UPDATED, {
    "project_id": "123",
    "action": "component_added"
})

# Listen for events
def on_project_updated(event):
    print(f"Project {event.data['project_id']} updated")

dal.event_bus.on(EventChannel.PROJECT_UPDATED, on_project_updated)
```

### Available Events

- `PROJECT_CREATED`
- `PROJECT_UPDATED`
- `PROJECT_DELETED`
- `COMPONENT_ADDED`
- `COMPONENT_USED`
- `CONNECTION_CREATED`
- `WORKFLOW_EXECUTED`
- `CODE_GENERATED`
- `DEBUG_SESSION_STARTED`
- `DEBUG_STEP_EXECUTED`
- `DEPLOYMENT_STARTED`
- `DEPLOYMENT_COMPLETED`
- `AI_SUGGESTION_GENERATED`

### HTTP Communication

Services communicate synchronously via HTTP for immediate responses:

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(f"{service_url}/endpoint")
    data = response.json()
```

## Configuration

### Environment Variables

```bash
# Service Ports
VISUAL_BUILDER_PORT=8006
VISUAL_WORKFLOW_ENGINE_PORT=8007
VISUAL_COMPONENT_REGISTRY_PORT=8008
VISUAL_CODE_GENERATOR_PORT=8009
VISUAL_DEBUGGER_PORT=8010
VISUAL_DEPLOYMENT_PORT=8011
VISUAL_AI_ASSISTANT_PORT=8012

# Service URLs (for inter-service communication)
VISUAL_WORKFLOW_ENGINE_URL=http://localhost:8007
VISUAL_COMPONENT_REGISTRY_URL=http://localhost:8008
VISUAL_CODE_GENERATOR_URL=http://localhost:8009
VISUAL_DEBUGGER_URL=http://localhost:8010
VISUAL_DEPLOYMENT_URL=http://localhost:8011
VISUAL_AI_ASSISTANT_URL=http://localhost:8012
```

### Docker Compose

All services are defined in `docker-compose.yml` with proper dependencies and networking.

## Development Setup

### Local Development

1. Start all services:
```bash
docker-compose up -d
```

2. Check service health:
```bash
curl http://localhost:8006/health
```

3. Access individual services:
- API Gateway: http://localhost:8006
- Workflow Engine: http://localhost:8007
- Component Registry: http://localhost:8008
- Code Generator: http://localhost:8009
- Debugger: http://localhost:8010
- Deployment: http://localhost:8011
- AI Assistant: http://localhost:8012

### Testing

Run integration tests:
```bash
python -m pytest tests/test_microservices_integration.py -v
```

## Migration Guide

### From Monolithic Service

The API Gateway maintains backward compatibility, so existing clients continue to work without changes. However, for optimal performance, update clients to use individual service endpoints directly.

### Breaking Changes

None - the API Gateway ensures backward compatibility.

## Monitoring and Observability

### Health Checks

The API Gateway provides aggregated health status:
```bash
curl http://localhost:8006/health
```

### Metrics

Each service exposes metrics via the shared monitoring infrastructure (InfluxDB + Grafana).

### Logging

All services use structured logging with consistent log levels and formats.

## Security Considerations

- Services communicate over internal Docker network
- API Gateway handles external authentication/authorization
- Sensitive configuration via environment variables
- No direct external access to individual microservices

## Performance Optimization

- Asynchronous request handling
- Connection pooling for inter-service communication
- Caching layer for frequently accessed data
- Horizontal scaling support via Docker Compose

## Troubleshooting

### Common Issues

1. **Service Unavailable**: Check Docker container status
   ```bash
   docker-compose ps
   ```

2. **Health Check Failures**: Check individual service logs
   ```bash
   docker-compose logs <service_name>
   ```

3. **Inter-Service Communication**: Verify environment variables and network connectivity

### Debugging

1. Enable debug logging:
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. Check service dependencies:
   ```bash
   docker-compose config
   ```

## Future Enhancements

- Service mesh integration (Istio/Linkerd)
- Advanced service discovery (Consul/Etcd)
- Circuit breaker patterns
- Distributed tracing expansion
- Auto-scaling policies
- Multi-region deployment support