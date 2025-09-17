#!/bin/bash

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

# Deploy
echo "🚀 Starting all services..."
docker-compose -f docker-compose.complete.yml up -d

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
echo "  - Visual Builder: http://localhost:8052"
echo "  - Monitoring Dashboard: http://localhost:8053"
echo "  - Production API: http://localhost:8001"
echo "  - Grafana: http://localhost:3000"
echo "  - Prometheus: http://localhost:9090"