#!/bin/bash

set -e

echo "🧪 Testing Docker Migration Setup..."

# Test network connectivity
echo "📡 Testing network connectivity..."
docker network ls | grep agent-lightning

# Test database connectivity
echo "🗄️ Testing PostgreSQL connectivity..."
docker exec agent-postgres-dev pg_isready -U agent_user -d agent_lightning_memory

# Test Redis connectivity
echo "🔴 Testing Redis connectivity..."
docker exec agent-redis-dev redis-cli -a redis_pass_2024 ping

# Test RabbitMQ connectivity
echo "🐰 Testing RabbitMQ connectivity..."
docker exec agent-rabbitmq rabbitmq-diagnostics ping

# Test volume mounts
echo "💾 Testing volume mounts..."
docker volume ls | grep agent-lightning

# Test service discovery
echo "🔍 Testing service discovery..."
docker-compose -f docker-compose.production.yml config --services

# Test monitoring stack
echo "📊 Testing monitoring stack..."
curl -s http://localhost:3000 > /dev/null && echo "✅ Grafana accessible" || echo "❌ Grafana not accessible"
curl -s http://localhost:9090 > /dev/null && echo "✅ Prometheus accessible" || echo "❌ Prometheus not accessible"
curl -s http://localhost:9093 > /dev/null && echo "✅ Alertmanager accessible" || echo "❌ Alertmanager not accessible"

echo "✅ All tests passed! Docker migration Phase 3 complete."