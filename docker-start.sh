#!/bin/bash

set -e

echo "🚀 Starting Agent Lightning Docker Environment..."

# Load environment variables
if [ -f .env.docker ]; then
    export $(cat .env.docker | grep -v '^#' | xargs)
fi

# Check if development or production
ENV=${1:-production}

if [ "$ENV" = "dev" ] || [ "$ENV" = "development" ]; then
    echo "📦 Starting development environment..."
    docker-compose -f docker-compose.dev.yml up --build -d
    echo "✅ Development environment started!"
    echo "🌐 API: http://localhost:8001"
    echo "📊 Dashboard: http://localhost:8051"
else
    echo "🏭 Starting production environment..."
    docker-compose -f docker-compose.production.yml up --build -d
    echo "✅ Production environment started!"
    echo "🌐 Application: http://localhost"
    echo "📊 Grafana: http://localhost:3000"
    echo "📈 Prometheus: http://localhost:9090"
fi

echo "🔍 Checking service health..."
sleep 10
docker-compose ps