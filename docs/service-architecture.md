# Agent Lightning Service Architecture Documentation

## Overview

Agent Lightning is a sophisticated microservices-based AI agent platform that implements a stateful ReAct (Reasoning and Acting) pattern. The system orchestrates intelligent task assignment, execution, and learning through a distributed architecture of specialized services.

## Architecture Components

### Core Services

#### 1. RL Orchestrator Service (Port 8025)
**Primary Function**: Intelligent task assignment and reinforcement learning orchestration

**Key Responsibilities:**
- Receives task requests via REST API (`/assign-task`)
- Performs capability matching using Q-learning algorithms
- Assigns tasks to appropriate agents based on expertise and performance history
- Maintains Q-tables for agent performance tracking
- Implements circuit breaker patterns for resilient cross-service communication
- Provides task status tracking via `/tasks` endpoint

**Technical Stack:**
- FastAPI for REST API
- Custom Q-learning implementation for agent selection
- SQLite/PostgreSQL for Q-table persistence
- Async/await patterns for concurrent task processing

**API Endpoints:**
- `POST /assign-task` - Assign tasks to agents
- `GET /tasks` - Query task status with filters
- `GET /health` - Service health check
- `GET /docs` - OpenAPI documentation

#### 2. Enhanced Production API (Port 8002)
**Primary Function**: Comprehensive agent execution and task processing

**Key Responsibilities:**
- Executes assigned tasks using specialized AI agents
- Manages agent lifecycle and context
- Provides knowledge management and memory systems
- Handles visual planning and deployment workflows
- Supports multiple LLM providers (OpenAI, Anthropic)
- Implements RL training integration
- Provides task status and result retrieval

**Technical Stack:**
- FastAPI with comprehensive middleware
- Multiple AI agent implementations
- Knowledge base integration
- Memory management systems
- Deployment pipeline integration

**API Endpoints:**
- `POST /api/v2/agents/execute` - Execute tasks with agents
- `GET /api/v1/tasks/{task_id}` - Retrieve task status and results
- `GET /agents` - List available agents
- `GET /health` - Service health check
- `POST /deploy` - Deployment operations

### Supporting Services

#### 3. Agent Designer Service (Port 8002 - Conflicts with Enhanced Production API)
**Status**: Currently blocked by port conflict with Enhanced Production API

**Intended Function**: Agent design and configuration management

**Note**: This service is redundant with Enhanced Production API capabilities and requires port reconfiguration.

#### 4. AI Model Service (Port 8005)
**Function**: Dedicated AI model management and inference

#### 5. RL Server (Port 8010)
**Function**: Reinforcement learning training and optimization

## Data Flow Architecture

### Task Execution Pipeline

```
User Request → RL Orchestrator → Enhanced Production API → Agent Execution → Result Return
     ↓              ↓                    ↓                    ↓            ↓
   Task ID      Capability Matching   Task Processing     AI Processing  Response
   Priority     Agent Selection       Context Management  LLM Calls      Status Update
   Description  Confidence Scoring    Memory Access       Result Generation
```

### Detailed Flow:

1. **Task Submission**
   - User submits task via RL Orchestrator `/assign-task` endpoint
   - Task includes: `task_id`, `description`, `priority`

2. **Intelligent Assignment**
   - RL Orchestrator analyzes task description
   - Matches against agent capabilities using Q-learning
   - Selects optimal agent based on performance history
   - Returns assignment confirmation with confidence score

3. **Task Execution**
   - RL Orchestrator triggers execution via Enhanced Production API
   - Primary endpoint: `/api/v2/agents/execute`
   - Fallback endpoint: `/agents/execute` (if primary fails)
   - Enhanced Production API processes task with assigned agent

4. **Result Processing**
   - Agent completes task execution
   - Results stored in Enhanced Production API
   - Status updated to "completed"
   - Results available via task status endpoints

## Service Communication Patterns

### Synchronous Communication
- REST API calls between RL Orchestrator and Enhanced Production API
- Health checks and status polling
- Immediate task assignment and execution triggers

### Asynchronous Processing
- Task execution happens asynchronously
- Results stored and retrieved on-demand
- Status updates via polling mechanism

### Error Handling
- Circuit breaker pattern for service resilience
- Automatic fallback to alternative endpoints
- Comprehensive error logging and monitoring

## Configuration Management

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Service URLs
RL_ORCHESTRATOR_URL=http://localhost:8025
ENHANCED_API_URL=http://localhost:8002

# Database
DATABASE_URL=sqlite:///agent_lightning.db

# Monitoring
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_token
```

### Service Configuration
Each service maintains its own configuration for:
- Port assignments
- Database connections
- External API integrations
- Logging levels
- Performance tuning parameters

## Deployment Architecture

### Local Development
```bash
# Start services individually
python -m services.rl_orchestrator_improved
python -m enhanced_production_api

# Or use docker-compose
docker-compose -f docker-compose.dev.yml up
```

### Production Deployment
- Containerized deployment using Docker
- Kubernetes orchestration for scaling
- Load balancing for high availability
- Monitoring with Grafana/InfluxDB

## Monitoring and Observability

### Health Checks
- All services provide `/health` endpoints
- Comprehensive health metrics including:
  - Service status
  - Database connectivity
  - External API availability
  - Queue lengths and processing metrics

### Logging
- Structured logging across all services
- Centralized log aggregation
- Error tracking and alerting
- Performance monitoring

### Metrics
- Task completion rates
- Agent performance statistics
- API response times
- Error rates and patterns

## Troubleshooting Guide

### Common Issues

#### Port Conflicts
**Problem**: Services fail to start due to occupied ports
**Solution**:
1. Check which services are running: `lsof -i :8025`
2. Stop conflicting services: `pkill -f service_name`
3. Reconfigure ports in service configuration
4. Restart services in correct order

#### API Endpoint Failures
**Problem**: 404 errors when calling service endpoints
**Solution**:
1. Verify service is running and healthy
2. Check endpoint URLs in configuration
3. Use fallback endpoints if available
4. Review service logs for detailed error messages

#### Task Assignment Failures
**Problem**: Tasks not being assigned to agents
**Solution**:
1. Verify RL Orchestrator is running
2. Check agent capability database
3. Review Q-learning parameters
4. Validate task description format

### Debugging Commands

```bash
# Check service status
curl http://localhost:8025/health
curl http://localhost:8002/health

# View service logs
tail -f logs/rl_orchestrator.log
tail -f logs/enhanced_api.log

# Test task assignment
curl -X POST http://localhost:8025/assign-task \
  -H "Content-Type: application/json" \
  -d '{"task_id": "test", "description": "Test task", "priority": 5}'

# Check task status
curl http://localhost:8002/api/v1/tasks/test
```

## Security Considerations

### API Security
- Input validation on all endpoints
- Rate limiting implementation
- Authentication for sensitive operations
- HTTPS encryption in production

### Data Protection
- PII masking in logs and responses
- Secure credential management
- Database encryption for sensitive data
- Audit logging for compliance

## Performance Optimization

### Caching Strategies
- Agent capability caching
- Task result caching
- Database query optimization
- API response caching

### Scalability Considerations
- Horizontal scaling of agent execution services
- Database connection pooling
- Async processing for high-throughput scenarios
- Load balancing for multiple service instances

## Future Enhancements

### Planned Improvements
- Service mesh implementation (Istio/Linkerd)
- Event-driven architecture with message queues
- Advanced monitoring with distributed tracing
- Auto-scaling based on load patterns
- Multi-region deployment support

### Integration Points
- Additional LLM provider support
- External knowledge base integration
- Workflow orchestration engines
- Advanced analytics and reporting

## Maintenance Procedures

### Regular Tasks
- Log rotation and cleanup
- Database maintenance and backups
- Dependency updates and security patches
- Performance monitoring and optimization

### Emergency Procedures
- Service restart procedures
- Database recovery processes
- Incident response protocols
- Communication plans for outages

## Contact and Support

**Development Team**: Agent Lightning Team
**Documentation**: This document and inline code comments
**Issue Tracking**: GitHub repository issues
**Support Channels**: Team communication platforms

---

**Last Updated**: 2025-09-19
**Version**: 1.0
**Status**: Production Ready