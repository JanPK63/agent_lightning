#!/bin/bash

set -e

echo "🛑 Stopping Agent Lightning Docker Environment..."

ENV=${1:-production}

if [ "$ENV" = "dev" ] || [ "$ENV" = "development" ]; then
    echo "📦 Stopping development environment..."
    docker-compose -f docker-compose.dev.yml down
else
    echo "🏭 Stopping production environment..."
    docker-compose -f docker-compose.production.yml down
fi

echo "✅ Environment stopped!"

# Optional: Remove volumes (uncomment if needed)
# echo "🗑️  Removing volumes..."
# docker volume prune -f