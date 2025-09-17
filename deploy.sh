#!/bin/bash

set -e

ENV=${1:-production}
MODE=${2:-secure}

echo "🚀 Deploying Agent Lightning - Environment: $ENV, Mode: $MODE"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."
docker --version
docker-compose --version

# Generate SSL certificates if needed
if [ ! -f "docker/configs/ssl/nginx.crt" ]; then
    echo "🔒 Generating SSL certificates..."
    ./docker/configs/ssl/generate-certs.sh
fi

# Initialize Docker Swarm and secrets for secure mode
if [ "$MODE" = "secure" ]; then
    echo "🔐 Checking Docker Swarm status..."
    if ! docker info | grep -q "Swarm: active"; then
        echo "🔄 Initializing Docker Swarm..."
        docker swarm init
    fi
    echo "🔐 Initializing secrets..."
    ./docker/security/init-secrets.sh
fi

# Build and deploy
echo "🏗️ Building and deploying services..."
if [ "$ENV" = "dev" ]; then
    docker-compose -f docker-compose.dev.yml down --remove-orphans
    docker-compose -f docker-compose.dev.yml up --build -d
elif [ "$MODE" = "secure" ]; then
    docker stack rm agent-lightning 2>/dev/null || true
    sleep 10
    docker stack deploy -c docker-compose.swarm.yml agent-lightning
else
    docker-compose -f docker-compose.production.yml down --remove-orphans
    docker-compose -f docker-compose.production.yml up --build -d
fi

# Health checks
echo "🏥 Running health checks..."
sleep 30

# Check core services
SERVICES=("postgres" "redis")
if [ "$ENV" != "dev" ]; then
    SERVICES+=("nginx")
fi

for service in "${SERVICES[@]}"; do
    if [ "$MODE" = "secure" ]; then
        if docker service ls | grep -q "agent-lightning_$service"; then
            echo "✅ $service is deployed"
        else
            echo "❌ $service is not deployed"
        fi
    else
        if docker-compose ps | grep -q "$service.*Up"; then
            echo "✅ $service is healthy"
        else
            echo "❌ $service is not healthy"
            exit 1
        fi
    fi
done

echo "🎉 Deployment complete!"
echo "📊 Access points:"
if [ "$MODE" = "secure" ]; then
    echo "  - Application: https://localhost"
    echo "  - Grafana: https://localhost:3000"
else
    echo "  - Application: http://localhost"
    echo "  - API: http://localhost:8001"
    echo "  - Dashboard: http://localhost:8051"
    echo "  - Grafana: http://localhost:3000"
fi