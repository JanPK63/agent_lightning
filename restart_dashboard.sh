#!/bin/bash

echo "🔄 Restarting Agent Lightning Dashboard..."

# Stop existing dashboard container if running
docker stop agent-dashboard 2>/dev/null || true
docker rm agent-dashboard 2>/dev/null || true

# Rebuild and start the dashboard
cd /Users/jankootstra/agent-lightning-main
docker-compose up -d --build agent-dashboard

echo "✅ Dashboard restarted successfully!"
echo "🌐 Access your dashboard at: http://localhost:8051"