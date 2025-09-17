#!/bin/bash

set -e

echo "🔒 Generating SSL certificates..."

# Create self-signed certificate for development
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout docker/configs/ssl/nginx.key \
    -out docker/configs/ssl/nginx.crt \
    -subj "/C=US/ST=State/L=City/O=AgentLightning/CN=localhost"

# Set proper permissions
chmod 600 docker/configs/ssl/nginx.key
chmod 644 docker/configs/ssl/nginx.crt

echo "✅ SSL certificates generated"