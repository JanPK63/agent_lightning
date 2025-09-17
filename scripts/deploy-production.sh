#!/bin/bash

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