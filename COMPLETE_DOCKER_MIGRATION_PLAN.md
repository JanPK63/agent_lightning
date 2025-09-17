# Complete Agent Lightning Docker Migration Plan

## Current System Analysis

### Core Services Identified
1. **Agent Dashboard** - `agent_dashboard.py` (Streamlit UI)
2. **Production APIs** - Multiple FastAPI services
3. **RL Training System** - Lightning server/client architecture
4. **Agent Coordination** - Multi-agent system management
5. **Memory Systems** - Shared memory and knowledge management
6. **Workflow Engine** - Enterprise workflow management
7. **Monitoring Stack** - Grafana, Prometheus, InfluxDB
8. **Database Layer** - PostgreSQL with Redis cache

## Migration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  Frontend Services                          │
├─────────────────────────────────────────────────────────────┤
│ • Agent Dashboard (Streamlit) - Port 8051                  │
│ • Visual Builder - Port 8052                               │
│ • Monitoring Dashboard - Port 8053                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                    API Gateway                              │
├─────────────────────────────────────────────────────────────┤
│ • Production API - Port 8001                               │
│ • Agent API - Port 8002                                    │
│ • RL API - Port 8003                                       │
│ • Workflow API - Port 8004                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  Core Services                              │
├─────────────────────────────────────────────────────────────┤
│ • RL Training Server - Port 8010                           │
│ • Agent Coordinator - Port 8011                            │
│ • Memory Manager - Port 8012                               │
│ • Workflow Engine - Port 8013                              │
│ • Knowledge Manager - Port 8014                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  Data Layer                                 │
├─────────────────────────────────────────────────────────────┤
│ • PostgreSQL - Port 5432                                   │
│ • Redis - Port 6379                                        │
│ • InfluxDB - Port 8086                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 Monitoring Stack                            │
├─────────────────────────────────────────────────────────────┤
│ • Prometheus - Port 9090                                   │
│ • Grafana - Port 3000                                      │
│ • Alertmanager - Port 9093                                 │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Core Service Containerization

### 1.1 Comprehensive Monitoring Dashboard Service
```dockerfile
# docker/services/monitoring-dashboard/Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY monitoring_dashboard_integrated.py .
EXPOSE 8053
CMD ["streamlit", "run", "monitoring_dashboard_integrated.py", "--server.port=8053", "--server.address=0.0.0.0", "--server.headless=true"]
```

### 1.2 Agent Dashboard Service
```dockerfile
# docker/services/agent-dashboard/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt streamlit
COPY agent_dashboard.py .
EXPOSE 8051
CMD ["streamlit", "run", "agent_dashboard.py", "--server.port=8051", "--server.address=0.0.0.0"]
```

### 1.3 Enhanced Production API Service
```dockerfile
# docker/services/enhanced-api/Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY enhanced_production_api.py .
EXPOSE 8002
CMD ["python", "enhanced_production_api.py"]
```

### 1.4 RL Training Service
```dockerfile
# docker/services/rl-training/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-rl.txt .
RUN pip install -r requirements-rl.txt
COPY lightning_server_rl.py .
COPY lightning_client_rl.py .
EXPOSE 8010
CMD ["python", "lightning_server_rl.py"]
```

### 1.5 Agent Coordination Service
```dockerfile
# docker/services/agent-coordinator/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY multi_agent_system.py .
COPY agent_coordination_state_machine.py .
COPY agent_client.py .
EXPOSE 8011
CMD ["python", "multi_agent_system.py"]
```

### 1.6 Memory Management Service
```dockerfile
# docker/services/memory-manager/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY memory_manager.py .
COPY shared_memory_system.py .
COPY postgres_memory_manager.py .
EXPOSE 8012
CMD ["python", "memory_manager.py"]
```

### 1.7 Workflow Engine Service
```dockerfile
# docker/services/workflow-engine/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY enterprise_workflow_engine.py .
COPY workflow_engine_service.py .
COPY workflow_api_service.py .
EXPOSE 8013
CMD ["python", "enterprise_workflow_engine.py"]
```

### 1.8 Knowledge Manager Service
```dockerfile
# docker/services/knowledge-manager/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY knowledge_manager.py .
COPY knowledge_trainer.py .
EXPOSE 8014
CMD ["python", "knowledge_manager.py"]
```

## Phase 2: Complete Docker Compose

### 2.1 Production Docker Compose
```yaml
# docker-compose.complete.yml
version: '3.8'

networks:
  agent-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  postgres_data:
  redis_data:
  influxdb_data:
  grafana_data:
  prometheus_data:
  agent_logs:
  rl_checkpoints:

services:
  # Frontend Services
  agent-dashboard:
    build: ./docker/services/agent-dashboard
    ports:
      - "8051:8051"
    networks:
      - frontend
      - api
    environment:
      - API_BASE_URL=http://production-api:8001
    depends_on:
      - production-api

  visual-builder:
    build: ./docker/services/visual-builder
    ports:
      - "8052:8052"
    networks:
      - frontend
      - api
    depends_on:
      - production-api

  monitoring-dashboard:
    build: ./docker/services/monitoring-dashboard
    ports:
      - "8053:8053"
    networks:
      - frontend
      - monitoring
    depends_on:
      - prometheus
      - influxdb

  # API Gateway
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/configs/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./docker/configs/ssl:/etc/nginx/ssl:ro
    networks:
      - frontend
    depends_on:
      - agent-dashboard
      - production-api

  # API Services
  production-api:
    build: ./docker/services/production-api
    ports:
      - "8001:8001"
    networks:
      - api
      - backend
      - database
    environment:
      - DATABASE_URL=postgresql://agent_user:agent_pass@postgres:5432/agent_lightning
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  agent-api:
    build: ./docker/services/agent-api
    ports:
      - "8002:8002"
    networks:
      - api
      - backend
      - database
    environment:
      - DATABASE_URL=postgresql://agent_user:agent_pass@postgres:5432/agent_lightning
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
      - agent-coordinator

  rl-api:
    build: ./docker/services/rl-api
    ports:
      - "8003:8003"
    networks:
      - api
      - rl
    depends_on:
      - rl-training-server

  workflow-api:
    build: ./docker/services/workflow-api
    ports:
      - "8004:8004"
    networks:
      - api
      - backend
      - database
    depends_on:
      - workflow-engine

  # Core Services
  rl-training-server:
    build: ./docker/services/rl-training
    ports:
      - "8010:8010"
    networks:
      - backend
    volumes:
      - rl_checkpoints:/app/checkpoints
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  agent-coordinator:
    build: ./docker/services/agent-coordinator
    ports:
      - "8011:8011"
    networks:
      - backend
      - database
    environment:
      - DATABASE_URL=postgresql://agent_user:agent_pass@postgres:5432/agent_lightning
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
      - memory-manager

  memory-manager:
    build: ./docker/services/memory-manager
    ports:
      - "8012:8012"
    networks:
      - backend
      - database
    environment:
      - DATABASE_URL=postgresql://agent_user:agent_pass@postgres:5432/agent_lightning
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  workflow-engine:
    build: ./docker/services/workflow-engine
    ports:
      - "8013:8013"
    networks:
      - backend
      - database
    environment:
      - DATABASE_URL=postgresql://agent_user:agent_pass@postgres:5432/agent_lightning
    depends_on:
      - postgres

  knowledge-manager:
    build: ./docker/services/knowledge-manager
    ports:
      - "8014:8014"
    networks:
      - backend
      - database
    environment:
      - DATABASE_URL=postgresql://agent_user:agent_pass@postgres:5432/agent_lightning
    depends_on:
      - postgres

  spec-driven:
    build: ./docker/services/spec-driven
    ports:
      - "8029:8029"
    networks:
      - backend

  # Data Layer
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: agent_lightning
      POSTGRES_USER: agent_user
      POSTGRES_PASSWORD: agent_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    networks:
      - database

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass redis_pass --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - databaseabase

  influxdb:
    image: influxdb:2.7-alpine
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: admin_pass
      DOCKER_INFLUXDB_INIT_ORG: agent-lightning
      DOCKER_INFLUXDB_INIT_BUCKET: metrics
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2
    networks:
      - monitoring

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    volumes:
      - ./docker/configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - monitoring
      - backend

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin_pass
      GF_USERS_ALLOW_SIGN_UP: false
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker/configs/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./docker/configs/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - monitoring
    depends_on:
      - prometheus
      - influxdb

  alertmanager:
    image: prom/alertmanager:latest
    command:
      - '--config.file=/etc/alertmanager/config.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    ports:
      - "9093:9093"
    volumes:
      - ./docker/configs/alertmanager.yml:/etc/alertmanager/config.yml:ro
    networks:
      - monitoring

networks:
  frontend:
    driver: bridge
  api:
    driver: bridge
  backend:
    driver: bridge
  database:
    driver: bridge
  monitoring:
    driver: bridge
  rl:
    driver: bridge
```

## Phase 3: Configuration Files

### 3.1 Nginx Configuration
```nginx
# docker/configs/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream dashboard {
        server agent-dashboard:8051;
    }
    
    upstream api {
        server production-api:8001;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://dashboard;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /api/ {
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

### 3.2 Prometheus Configuration
```yaml
# docker/configs/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agent-services'
    static_configs:
      - targets:
        - 'production-api:8001'
        - 'agent-coordinator:8011'
        - 'memory-manager:8012'
        - 'workflow-engine:8013'
        - 'knowledge-manager:8014'
        - 'rl-training-server:8010'
```

### 3.3 Database Schema
```sql
-- database/schema.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    config JSONB,
    status VARCHAR(50) DEFAULT 'inactive',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    definition JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE training_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id),
    config JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id),
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Phase 4: Deployment Scripts

### 4.1 Production Deployment
```bash
#!/bin/bash
# scripts/deploy-production.sh

set -e

echo "Starting Agent Lightning production deployment..."

# Build all services
docker-compose -f docker-compose.complete.yml build

# Start infrastructure services first
docker-compose -f docker-compose.complete.yml up -d postgres redis influxdb

# Wait for databases to be ready
echo "Waiting for databases..."
sleep 30

# Start core services
docker-compose -f docker-compose.complete.yml up -d \
  memory-manager \
  agent-coordinator \
  workflow-engine \
  knowledge-manager \
  rl-training-server

# Start API services
docker-compose -f docker-compose.complete.yml up -d \
  production-api \
  agent-api \
  rl-api \
  workflow-api

# Start monitoring
docker-compose -f docker-compose.complete.yml up -d \
  prometheus \
  grafana \
  alertmanager

# Start frontend services
docker-compose -f docker-compose.complete.yml up -d \
  agent-dashboard \
  visual-builder \
  monitoring-dashboard \
  nginx

echo "Deployment complete! Services available at:"
echo "  Dashboard: http://localhost"
echo "  API: http://localhost/api"
echo "  Monitoring: http://localhost:3000"
```

### 4.2 Development Setup
```bash
#!/bin/bash
# scripts/setup-dev.sh

set -e

echo "Setting up Agent Lightning development environment..."

# Create development docker-compose override
cat > docker-compose.dev.yml << EOF
version: '3.8'
services:
  production-api:
    volumes:
      - ./src:/app/src
    environment:
      - DEBUG=true
      - LOG_LEVEL=debug
  
  agent-dashboard:
    volumes:
      - ./dashboard:/app
    environment:
      - STREAMLIT_SERVER_RELOAD_ON_CHANGE=true
EOF

# Start development environment
docker-compose -f docker-compose.complete.yml -f docker-compose.dev.yml up -d

echo "Development environment ready!"
```

## Phase 5: Migration Checklist

### 5.1 Pre-Migration Tasks
- [ ] Backup existing data and configurations
- [ ] Test all Dockerfiles locally
- [ ] Verify network connectivity between services
- [ ] Validate environment variables and secrets
- [ ] Test database migrations

### 5.2 Migration Steps
1. [ ] Deploy infrastructure services (postgres, redis, influxdb)
2. [ ] Run database migrations
3. [ ] Deploy core services in dependency order
4. [ ] Deploy API services
5. [ ] Deploy monitoring stack
6. [ ] Deploy frontend services
7. [ ] Configure load balancer/nginx
8. [ ] Verify all health checks pass

### 5.3 Post-Migration Validation
- [ ] All services respond to health checks
- [ ] Agent creation and training workflows work
- [ ] Dashboard displays correct data
- [ ] Monitoring alerts are configured
- [ ] Performance metrics are within acceptable ranges

## Phase 6: Maintenance and Operations

### 6.1 Backup Strategy
```bash
#!/bin/bash
# scripts/backup.sh

# Backup PostgreSQL
docker exec postgres pg_dump -U agent_user agent_lightning > backup_$(date +%Y%m%d).sql

# Backup Redis
docker exec redis redis-cli BGSAVE
docker cp redis:/data/dump.rdb backup_redis_$(date +%Y%m%d).rdb

# Backup InfluxDB
docker exec influxdb influx backup /tmp/backup
docker cp influxdb:/tmp/backup ./backup_influx_$(date +%Y%m%d)/
```

### 6.2 Health Monitoring
```bash
#!/bin/bash
# scripts/health-check.sh

services=("production-api" "agent-coordinator" "memory-manager" "workflow-engine")

for service in "${services[@]}"; do
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        echo "✓ $service is healthy"
    else
        echo "✗ $service is unhealthy"
    fi
done
```

This completes the comprehensive Docker migration plan for Agent Lightning, providing a production-ready containerized architecture with proper monitoring, scaling, and maintenance procedures.kend
      - database

  influxdb:
    image: influxdb:2.7-alpine
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: admin_pass
      DOCKER_INFLUXDB_INIT_ORG: agent-lightning
      DOCKER_INFLUXDB_INIT_BUCKET: metrics
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2
    networks:
      - monitoring

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9090:9090"
    volumes:
      - ./docker/configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - monitoring
      - backend

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker/configs/grafana:/etc/grafana/provisioning:ro
    networks:
      - monitoring
    depends_on:
      - prometheus
      - influxdb

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./docker/configs/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
    networks:
      - monitoring
```

## Phase 3: Service Integration Scripts

### 3.1 Complete Deployment Script
```bash
#!/bin/bash
# deploy-complete.sh

set -e

ENV=${1:-production}
echo "🚀 Deploying Complete Agent Lightning System - Environment: $ENV"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."
docker --version
docker-compose --version

# Build all services
echo "🏗️ Building all services..."
docker-compose -f docker-compose.complete.yml build

# Deploy based on environment
if [ "$ENV" = "dev" ]; then
    docker-compose -f docker-compose.complete.yml -f docker-compose.dev-overrides.yml up -d
else
    docker-compose -f docker-compose.complete.yml up -d
fi

# Health checks
echo "🏥 Running comprehensive health checks..."
sleep 30

SERVICES=(
    "agent-dashboard:8051"
    "production-api:8001"
    "agent-coordinator:8011"
    "memory-manager:8012"
    "workflow-engine:8013"
    "postgres:5432"
    "redis:6379"
)

for service_port in "${SERVICES[@]}"; do
    service=$(echo $service_port | cut -d: -f1)
    port=$(echo $service_port | cut -d: -f2)
    
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "✅ $service is healthy"
    else
        echo "❌ $service is not responding"
    fi
done

echo "🎉 Complete deployment finished!"
echo "📊 Access points:"
echo "  - Agent Dashboard: http://localhost:8051"
echo "  - Production API: http://localhost:8001"
echo "  - Grafana: http://localhost:3000"
echo "  - Prometheus: http://localhost:9090"
```

## Phase 4: Migration Execution Plan

### 4.1 Service Migration Order
1. **Data Layer** (PostgreSQL, Redis, InfluxDB)
2. **Core Services** (Memory, Knowledge, Workflow)
3. **Agent Services** (Coordinator, RL Training)
4. **API Layer** (Production, Agent, RL APIs)
5. **Frontend Services** (Dashboard, Visual Builder)
6. **Monitoring Stack** (Prometheus, Grafana)
7. **Load Balancer** (Nginx)

### 4.2 Validation Checklist
- [ ] All existing Python services containerized
- [ ] Database connections working
- [ ] Agent dashboard accessible
- [ ] RL training functional
- [ ] Multi-agent coordination working
- [ ] Memory systems operational
- [ ] Workflow engine running
- [ ] Monitoring stack active
- [ ] SSL/TLS configured
- [ ] Backup systems in place

### 4.3 Rollback Plan
```bash
#!/bin/bash
# rollback.sh
echo "🔄 Rolling back to previous system..."
docker-compose -f docker-compose.complete.yml down
# Restore from backup
tar -xzf agent-lightning-backup-$(date +%Y%m%d).tar.gz
echo "✅ Rollback complete"
```

## Phase 5: Post-Migration Tasks

### 5.1 Performance Optimization
- Resource limits configuration
- Auto-scaling setup
- Load balancing optimization
- Database connection pooling

### 5.2 Security Hardening
- Container security scanning
- Network segmentation
- Secrets management
- SSL certificate automation

### 5.3 Monitoring Enhancement
- Custom metrics collection
- Alert rule configuration
- Dashboard customization
- Log aggregation setup

## Estimated Timeline
- **Phase 1**: 2 days (Service containerization)
- **Phase 2**: 1 day (Docker Compose setup)
- **Phase 3**: 1 day (Integration scripts)
- **Phase 4**: 1 day (Migration execution)
- **Phase 5**: 1 day (Post-migration tasks)

**Total: 6 days for complete migration**

## Success Criteria
✅ All existing functionality preserved
✅ Agent dashboard fully operational
✅ RL training system working
✅ Multi-agent coordination active
✅ Memory and knowledge systems functional
✅ Workflow engine operational
✅ Monitoring stack complete
✅ Production-ready deployment