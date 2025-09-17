#!/bin/bash

services=("production-api:8001" "agent-coordinator:8011" "memory-manager:8012" "workflow-engine:8013")

for service_port in "${services[@]}"; do
    service=$(echo $service_port | cut -d: -f1)
    port=$(echo $service_port | cut -d: -f2)
    
    if curl -f http://localhost:$port/health > /dev/null 2>&1; then
        echo "✓ $service is healthy"
    else
        echo "✗ $service is unhealthy"
    fi
done