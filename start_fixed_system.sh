#!/bin/bash
# Agent Lightning - Fixed System Startup

echo "🚀 Starting Fixed Agent Lightning System..."

# Kill any existing processes on port 8888
lsof -ti:8888 | xargs kill -9 2>/dev/null || true

# Start the fixed API
echo "Starting Fixed Agent API on port 8888..."
python3 fixed_agent_api.py &

# Wait for startup
sleep 3

# Test the system
echo "Testing system..."
curl -s http://localhost:8888/health > /dev/null && echo "✅ System is running!" || echo "❌ System failed to start"

echo "🎯 Fixed Agent Lightning is ready!"
echo "   • API: http://localhost:8888"
echo "   • Agents: http://localhost:8888/agents"
echo "   • Execute: http://localhost:8888/execute"
