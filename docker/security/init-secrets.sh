#!/bin/bash

set -e

echo "🔐 Initializing Docker secrets..."

# Generate secure passwords
POSTGRES_PASS=$(openssl rand -base64 32)
REDIS_PASS=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 64)
RABBITMQ_PASS=$(openssl rand -base64 32)
INFLUXDB_TOKEN=$(openssl rand -base64 32)
GRAFANA_PASS=$(openssl rand -base64 32)

# Create Docker secrets
echo "$POSTGRES_PASS" | docker secret create postgres_password - 2>/dev/null || echo "postgres_password secret exists"
echo "$REDIS_PASS" | docker secret create redis_password - 2>/dev/null || echo "redis_password secret exists"
echo "$JWT_SECRET" | docker secret create jwt_secret - 2>/dev/null || echo "jwt_secret secret exists"
echo "$RABBITMQ_PASS" | docker secret create rabbitmq_password - 2>/dev/null || echo "rabbitmq_password secret exists"
echo "$INFLUXDB_TOKEN" | docker secret create influxdb_token - 2>/dev/null || echo "influxdb_token secret exists"
echo "$GRAFANA_PASS" | docker secret create grafana_password - 2>/dev/null || echo "grafana_password secret exists"

echo "✅ Docker secrets initialized"