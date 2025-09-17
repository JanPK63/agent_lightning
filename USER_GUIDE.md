# Agent Lightning⚡ User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Agent Management](#agent-management)
4. [Reinforcement Learning](#reinforcement-learning)
5. [Workflow Management](#workflow-management)
6. [Monitoring & Analytics](#monitoring--analytics)
7. [API Usage](#api-usage)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites
- Python 3.10 or later
- Docker and Docker Compose
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### 5-Minute Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/microsoft/agent-lightning
   cd agent-lightning
   pip install -e .
   ```

2. **Start Infrastructure**
   ```bash
   # Start PostgreSQL, Redis, and monitoring stack
   docker-compose -f monitoring/docker-compose-rl.yml up -d
   ```

3. **Initialize Database**
   ```bash
   python json_to_postgres_migration.py
   ```

4. **Launch API Server**
   ```bash
   python enhanced_production_api.py
   ```

5. **Access Dashboards**
   - **API Explorer**: http://localhost:8000/docs
   - **RL Dashboard**: http://localhost:8501
   - **Monitoring**: http://localhost:3000 (admin/admin)

### First Agent Task
```bash
curl -X POST "http://localhost:8000/api/v2/agents/assign" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze the current market trends for AI technology",
    "agent_type": "research_agent",
    "priority": "high"
  }'
```

## System Overview

### Architecture Components

#### 1. API Gateway (`enhanced_production_api.py`)
- **Purpose**: Main entry point for all system interactions
- **Features**: JWT authentication, rate limiting, request routing
- **Endpoints**: 15+ REST API endpoints for complete system control

#### 2. Agent Management System
- **31 Production Agents**: Specialized for different tasks
- **LangChain Integration**: Advanced memory and conversation management
- **Tool Framework**: 7 integrated tools for enhanced capabilities
- **Internet Access**: Real web browsing and search capabilities

#### 3. RL Orchestrator (`rl_orch/`)
- **Auto-RL System**: Intelligent automatic training (94.2% success rate)
- **Algorithms**: PPO, DQN, SAC with real Gymnasium environments
- **Distributed Training**: Ray-based parallel processing
- **CLI Management**: `rlctl` command-line interface

#### 4. Data Layer
- **PostgreSQL**: Production database with 1,421 knowledge items
- **Redis**: High-performance caching (50%+ hit rate)
- **Migration Tools**: Seamless data migration utilities

#### 5. Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: 4 specialized dashboards for visualization
- **Health Checks**: Real-time system status monitoring

## Agent Management

### Available Agents (31 Total)

#### Core Agents
- **research_agent**: Web research and data analysis
- **code_agent**: Code generation, review, and debugging
- **data_agent**: Data processing and SQL operations
- **content_agent**: Content creation and editing
- **analysis_agent**: Business analysis and reporting

#### Specialized Agents
- **sql_agent**: Database queries and optimization
- **web_agent**: Web scraping and API interactions
- **file_agent**: File operations and management
- **security_agent**: Security analysis and compliance
- **workflow_agent**: Process automation and orchestration

### Agent Assignment

#### Basic Assignment
```python
import requests

response = requests.post("http://localhost:8000/api/v2/agents/assign", json={
    "task": "Create a Python function to calculate fibonacci numbers",
    "agent_type": "code_agent",
    "priority": "medium",
    "context": {
        "language": "python",
        "complexity": "intermediate"
    }
})

result = response.json()
print(f"Task ID: {result['task_id']}")
print(f"Agent: {result['assigned_agent']}")
```

#### Advanced Assignment with Tools
```python
response = requests.post("http://localhost:8000/api/v2/agents/assign", json={
    "task": "Research current AI trends and create a summary report",
    "agent_type": "research_agent",
    "tools_enabled": ["search_web", "write_file", "query_knowledge"],
    "output_format": "markdown",
    "max_tokens": 2000
})
```

### Agent Tools Framework

#### Available Tools
1. **execute_python_code**: Run Python code in sandboxed environment
2. **search_web**: Internet search with result filtering
3. **read_file**: Read files from the system
4. **write_file**: Create and modify files
5. **list_directory**: Browse directory structures
6. **query_agents**: Interact with other agents
7. **query_knowledge**: Access knowledge base

#### Tool Usage Example
```python
# Agent automatically selects appropriate tools based on task
response = requests.post("http://localhost:8000/api/v2/agents/assign", json={
    "task": "Find the latest Python version, download release notes, and summarize key features",
    "agent_type": "research_agent",
    "auto_tools": True  # Agent selects tools automatically
})
```

## Reinforcement Learning

### Auto-RL System

The Auto-RL system automatically analyzes tasks and triggers reinforcement learning when beneficial.

#### How It Works
1. **Task Analysis**: Evaluates task complexity and type
2. **Performance Baseline**: Measures current agent performance
3. **RL Decision**: Determines if RL training would be beneficial
4. **Automatic Training**: Launches RL training without user intervention
5. **Performance Validation**: Verifies improvement after training

#### Monitoring Auto-RL
```python
# Check Auto-RL status
response = requests.get("http://localhost:8000/api/v2/rl/auto-status")
status = response.json()

print(f"Auto-RL Active: {status['auto_rl_active']}")
print(f"Success Rate: {status['success_rate']}%")
print(f"Active Sessions: {status['active_sessions']}")
```

#### Manual RL Trigger
```python
# Manually trigger RL for specific agent
response = requests.post("http://localhost:8000/api/v2/rl/auto-trigger", json={
    "agent_id": "code_agent",
    "task_type": "code_generation",
    "performance_threshold": 0.85
})
```

### RL Orchestrator CLI

#### Basic Commands
```bash
# Launch RL experiment
rlctl launch --config configs/experiment.yaml

# Check experiment status
rlctl status --experiment ppo_cartpole_mvp

# Resume training from checkpoint
rlctl resume --experiment ppo_cartpole_mvp --checkpoint latest

# Run hyperparameter sweep
rlctl sweep --config configs/sweep.yaml

# Promote model to production
rlctl promote --experiment ppo_cartpole_mvp --version best
```

#### Configuration Example
```yaml
# configs/custom_experiment.yaml
experiment:
  name: "custom_agent_training"
  algorithm: "ppo"
  
environment:
  name: "AgentEnvironment"
  max_episode_steps: 1000
  
policy:
  network_arch: [256, 256]
  learning_rate: 3e-4
  
training:
  total_timesteps: 1000000
  batch_size: 64
  n_epochs: 10
```

## Workflow Management

### Enterprise Workflow Engine

#### Creating Workflows
```python
workflow_config = {
    "name": "Data Analysis Pipeline",
    "description": "Complete data analysis workflow",
    "steps": [
        {
            "id": "data_collection",
            "agent_type": "data_agent",
            "task": "Collect data from specified sources",
            "dependencies": []
        },
        {
            "id": "data_cleaning",
            "agent_type": "data_agent", 
            "task": "Clean and preprocess collected data",
            "dependencies": ["data_collection"]
        },
        {
            "id": "analysis",
            "agent_type": "analysis_agent",
            "task": "Perform statistical analysis",
            "dependencies": ["data_cleaning"]
        },
        {
            "id": "report_generation",
            "agent_type": "content_agent",
            "task": "Generate analysis report",
            "dependencies": ["analysis"]
        }
    ],
    "fault_tolerance": True,
    "max_retries": 3
}

response = requests.post("http://localhost:8000/api/v2/workflows", json=workflow_config)
workflow_id = response.json()["workflow_id"]
```

#### Executing Workflows
```python
# Start workflow execution
response = requests.post(f"http://localhost:8000/api/v2/workflows/{workflow_id}/execute", json={
    "input_data": {
        "data_sources": ["database", "api", "files"],
        "analysis_type": "trend_analysis"
    }
})

execution_id = response.json()["execution_id"]
```

#### Monitoring Workflows
```python
# Check workflow status
response = requests.get(f"http://localhost:8000/api/v2/workflows/{workflow_id}/status")
status = response.json()

print(f"Status: {status['status']}")
print(f"Progress: {status['progress']}%")
print(f"Current Step: {status['current_step']}")
```

## Monitoring & Analytics

### Grafana Dashboards

#### 1. RL Overview Dashboard
- **URL**: http://localhost:3000/d/rl-overview
- **Metrics**: Session counts, success rates, training progress
- **Use Case**: High-level RL system monitoring

#### 2. RL Performance Dashboard  
- **URL**: http://localhost:3000/d/rl-performance
- **Metrics**: Performance gains, training rates, algorithm comparison
- **Use Case**: Performance optimization and analysis

#### 3. RL Agents Dashboard
- **URL**: http://localhost:3000/d/rl-agents
- **Metrics**: Agent activity, response times, success rates
- **Use Case**: Individual agent performance monitoring

#### 4. RL System Health Dashboard
- **URL**: http://localhost:3000/d/rl-system-health
- **Metrics**: System resources, error rates, service health
- **Use Case**: Infrastructure monitoring during RL training

### Prometheus Metrics

#### Key Metrics
- `rl_sessions_total`: Total RL training sessions
- `rl_auto_triggered_total`: Auto-triggered RL sessions
- `rl_success_rate`: RL training success rate
- `rl_performance_gain_percent`: Performance improvement percentage
- `rl_sessions_active`: Currently active RL sessions
- `agent_requests_total`: Total agent requests
- `agent_response_time_seconds`: Agent response times
- `system_cpu_usage_percent`: CPU utilization
- `system_memory_usage_percent`: Memory utilization

#### Custom Alerts
```yaml
# prometheus_alerts.yml
groups:
  - name: agent_lightning
    rules:
      - alert: HighRLFailureRate
        expr: rl_success_rate < 0.8
        for: 5m
        annotations:
          summary: "RL success rate below 80%"
          
      - alert: HighAgentResponseTime
        expr: agent_response_time_seconds > 10
        for: 2m
        annotations:
          summary: "Agent response time above 10 seconds"
```

### Smart RL Dashboard

Access the Streamlit dashboard at http://localhost:8501

#### Features
- **Real-time Metrics**: Live RL system performance
- **Agent Management**: Start/stop agents, view status
- **Training Control**: Launch RL experiments, monitor progress
- **Performance Analytics**: Historical performance data
- **System Health**: Infrastructure monitoring

## API Usage

### Authentication

#### Login
```python
import requests

# Login to get JWT token
response = requests.post("http://localhost:8000/auth/login", json={
    "username": "admin",
    "password": "your_password"
})

token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
```

#### Role-Based Access
- **Admin**: Full system access, user management
- **Developer**: Agent management, RL training, monitoring
- **User**: Basic agent interaction, read-only monitoring

### Core API Endpoints

#### Agent Operations
```python
# List all agents
response = requests.get("http://localhost:8000/api/v2/agents", headers=headers)
agents = response.json()["agents"]

# Get agent details
response = requests.get(f"http://localhost:8000/api/v2/agents/{agent_id}", headers=headers)
agent_info = response.json()

# Update agent configuration
response = requests.put(f"http://localhost:8000/api/v2/agents/{agent_id}", 
                       json={"config": {"max_tokens": 2000}}, 
                       headers=headers)
```

#### Knowledge Management
```python
# Add knowledge item
response = requests.post("http://localhost:8000/api/v2/knowledge", json={
    "title": "API Documentation",
    "content": "Detailed API usage instructions...",
    "tags": ["api", "documentation"],
    "category": "technical"
}, headers=headers)

# Search knowledge base
response = requests.get("http://localhost:8000/api/v2/knowledge/search", 
                       params={"query": "API usage", "limit": 10}, 
                       headers=headers)
```

#### System Management
```python
# System health check
response = requests.get("http://localhost:8000/api/v2/health")
health = response.json()

# System statistics
response = requests.get("http://localhost:8000/api/v2/system/stats", headers=headers)
stats = response.json()

# Update system configuration
response = requests.put("http://localhost:8000/api/v2/system/config", json={
    "auto_rl_enabled": True,
    "max_concurrent_agents": 50,
    "cache_ttl": 3600
}, headers=headers)
```

## Troubleshooting

### Common Issues

#### 1. Agent Not Responding
**Symptoms**: Agent requests timeout or return errors
**Solutions**:
```bash
# Check agent status
curl http://localhost:8000/api/v2/agents/status

# Restart specific agent
curl -X POST http://localhost:8000/api/v2/agents/{agent_id}/restart

# Check system resources
curl http://localhost:8000/api/v2/health
```

#### 2. RL Training Failures
**Symptoms**: RL training sessions fail or hang
**Solutions**:
```bash
# Check RL system status
curl http://localhost:8000/api/v2/rl/auto-status

# View RL logs
docker logs agent-lightning-rl

# Restart RL orchestrator
rlctl restart --experiment {experiment_name}
```

#### 3. Database Connection Issues
**Symptoms**: Database errors or connection timeouts
**Solutions**:
```bash
# Check PostgreSQL status
docker ps | grep postgres

# Restart database
docker-compose restart postgres

# Run database migration
python json_to_postgres_migration.py --verify
```

#### 4. Cache Performance Issues
**Symptoms**: Slow response times, low cache hit rate
**Solutions**:
```bash
# Check Redis status
docker exec -it redis redis-cli ping

# Clear cache
docker exec -it redis redis-cli flushall

# Monitor cache performance
curl http://localhost:8000/metrics | grep cache
```

### Performance Optimization

#### 1. Agent Pool Sizing
```python
# Optimize agent pool based on workload
response = requests.put("http://localhost:8000/api/v2/system/config", json={
    "agent_pool_size": 20,  # Adjust based on concurrent requests
    "agent_timeout": 30,    # Seconds
    "max_queue_size": 100
})
```

#### 2. Cache Configuration
```python
# Optimize Redis cache settings
cache_config = {
    "cache_ttl": 3600,      # 1 hour default TTL
    "max_memory": "2gb",    # Maximum cache memory
    "eviction_policy": "allkeys-lru"
}
```

#### 3. RL Training Optimization
```yaml
# Optimize RL training performance
training:
  batch_size: 128         # Increase for better GPU utilization
  n_workers: 8           # Match CPU cores
  buffer_size: 100000    # Larger buffer for better sampling
  checkpoint_frequency: 1000  # Save checkpoints more frequently
```

### Logging and Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export AGENT_LIGHTNING_LOG_LEVEL=DEBUG
```

#### View System Logs
```bash
# API server logs
tail -f logs/api_server.log

# RL orchestrator logs  
tail -f logs/rl_orchestrator.log

# Agent logs
tail -f logs/agents/*.log

# Database logs
docker logs postgres
```

#### Performance Profiling
```python
# Enable performance profiling
response = requests.post("http://localhost:8000/api/v2/system/profiling", json={
    "enabled": True,
    "sample_rate": 0.1,  # 10% sampling
    "output_format": "json"
})
```

### Support Resources

#### Documentation
- **API Reference**: http://localhost:8000/docs
- **Technical Docs**: `/docs` directory
- **Examples**: `/examples` directory

#### Community Support
- **Discord**: https://discord.gg/RYk7CdvDR7
- **GitHub Issues**: https://github.com/microsoft/agent-lightning/issues
- **Stack Overflow**: Tag `agent-lightning`

#### Professional Support
- **Email**: support@agentlightning.com
- **Enterprise Support**: Available with commercial license
- **Training**: Professional training programs available

---

*For additional help, consult the [System Documentation](SYSTEM_DOCUMENTATION.md) or contact our support team.*