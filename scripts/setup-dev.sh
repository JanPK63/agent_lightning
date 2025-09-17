#!/bin/bash

set -e

echo "Setting up Agent Lightning development environment..."

# Create development docker-compose override
cat > docker-compose.dev.yml << EOF
version: '3.8'
services:
  production-api:
    volumes:
      - ./:/app/src
    environment:
      - DEBUG=true
      - LOG_LEVEL=debug
  
  agent-dashboard:
    volumes:
      - ./:/app
    environment:
      - STREAMLIT_SERVER_RELOAD_ON_CHANGE=true
EOF

# Start development environment
docker-compose -f docker-compose.complete.yml -f docker-compose.dev.yml up -d

echo "Development environment ready!"