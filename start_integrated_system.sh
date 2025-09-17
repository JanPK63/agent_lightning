#!/bin/bash

# Lightning System - Integrated Services Startup Script
# This script starts all integrated services with shared database and Redis

echo "========================================="
echo "🚀 Lightning System - Integrated Startup"
echo "========================================="

# Set environment variables
export REDIS_PASSWORD='redis_secure_password_123'
export DATABASE_URL='postgresql://agent_user:agent_pass_123@localhost:5432/agent_lightning'
export JWT_SECRET_KEY='your-secret-key-change-in-production'

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if service is running
check_service() {
    local port=$1
    local name=$2
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name is running on port $port${NC}"
        return 0
    else
        echo -e "${RED}❌ $name is not running on port $port${NC}"
        return 1
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    if lsof -i:$port > /dev/null 2>&1; then
        echo "Stopping service on port $port..."
        kill -9 $(lsof -t -i:$port) 2>/dev/null
        sleep 1
    fi
}

# Check command line arguments
if [ "$1" == "stop" ]; then
    echo -e "${YELLOW}Stopping all services...${NC}"
    
    # Stop integrated services
    kill_port 8102  # Agent Designer
    kill_port 8103  # Workflow Engine
    kill_port 8104  # Integration Hub
    kill_port 8105  # AI Model
    kill_port 8106  # Auth Service
    kill_port 8107  # WebSocket
    
    # Stop old services if running
    kill_port 8001  # Old Agent Designer
    kill_port 8002  # Old Agent Designer
    kill_port 8003  # Old Workflow Engine
    kill_port 8004  # Old Integration Hub
    kill_port 8005  # Old AI Model
    kill_port 8006  # Old Auth
    kill_port 8007  # Old WebSocket
    
    # Stop API Gateway and Dashboard
    kill_port 8000  # API Gateway
    kill_port 8051  # Dashboard
    
    echo -e "${GREEN}All services stopped${NC}"
    exit 0
fi

if [ "$1" == "status" ]; then
    echo -e "${YELLOW}Service Status:${NC}"
    echo "-------------------"
    check_service 5432 "PostgreSQL"
    check_service 6379 "Redis"
    echo "-------------------"
    check_service 8102 "Agent Designer (Integrated)"
    check_service 8103 "Workflow Engine (Integrated)"
    check_service 8104 "Integration Hub (Integrated)"
    check_service 8105 "AI Model Service (Integrated)"
    check_service 8106 "Auth Service (Integrated)"
    check_service 8107 "WebSocket Service (Integrated)"
    echo "-------------------"
    check_service 8000 "API Gateway"
    check_service 8051 "Monitoring Dashboard"
    exit 0
fi

echo -e "${YELLOW}Step 1: Checking Prerequisites${NC}"
echo "--------------------------------"

# Check PostgreSQL
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo -e "${RED}❌ PostgreSQL is not running!${NC}"
    echo "Please start PostgreSQL first:"
    echo "  brew services start postgresql@14"
    exit 1
fi
echo -e "${GREEN}✅ PostgreSQL is running${NC}"

# Check Redis
if ! redis-cli -a "$REDIS_PASSWORD" ping > /dev/null 2>&1; then
    echo -e "${RED}❌ Redis is not running or password is incorrect!${NC}"
    echo "Please start Redis first:"
    echo "  redis-server --requirepass $REDIS_PASSWORD"
    exit 1
fi
echo -e "${GREEN}✅ Redis is running${NC}"

# Check database exists
if ! psql -U agent_user -d agent_lightning -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${YELLOW}Creating database...${NC}"
    createdb -U agent_user agent_lightning 2>/dev/null || true
fi
echo -e "${GREEN}✅ Database 'agent_lightning' exists${NC}"

echo ""
echo -e "${YELLOW}Step 2: Stopping Old Services${NC}"
echo "------------------------------"

# Stop any existing services on old ports
for port in 8001 8002 8003 8004 8005 8006 8007; do
    kill_port $port
done

# Stop existing integrated services
for port in 8102 8103 8104 8105 8106 8107; do
    kill_port $port
done

echo -e "${GREEN}✅ Old services stopped${NC}"

echo ""
echo -e "${YELLOW}Step 3: Starting Integrated Services${NC}"
echo "------------------------------------"

# Navigate to services directory
cd services || exit 1

# Start integrated services
echo "Starting Agent Designer (Port 8102)..."
REDIS_PASSWORD="$REDIS_PASSWORD" AGENT_DESIGNER_PORT=8102 python agent_designer_service_integrated.py > /tmp/agent_designer.log 2>&1 &
sleep 2

echo "Starting Workflow Engine (Port 8103)..."
REDIS_PASSWORD="$REDIS_PASSWORD" WORKFLOW_ENGINE_PORT=8103 python workflow_engine_service_integrated.py > /tmp/workflow_engine.log 2>&1 &
sleep 2

echo "Starting Integration Hub (Port 8104)..."
REDIS_PASSWORD="$REDIS_PASSWORD" INTEGRATION_HUB_PORT=8104 python integration_hub_service_integrated.py > /tmp/integration_hub.log 2>&1 &
sleep 2

echo "Starting AI Model Service (Port 8105)..."
REDIS_PASSWORD="$REDIS_PASSWORD" AI_MODEL_PORT=8105 python ai_model_service_integrated.py > /tmp/ai_model.log 2>&1 &
sleep 2

echo "Starting Auth Service (Port 8106)..."
REDIS_PASSWORD="$REDIS_PASSWORD" AUTH_PORT=8106 python auth_service_integrated.py > /tmp/auth.log 2>&1 &
sleep 2

echo "Starting WebSocket Service (Port 8107)..."
REDIS_PASSWORD="$REDIS_PASSWORD" WEBSOCKET_PORT=8107 python websocket_service_integrated.py > /tmp/websocket.log 2>&1 &
sleep 2

# Go back to main directory
cd ..

echo ""
echo -e "${YELLOW}Step 4: Verifying Services${NC}"
echo "--------------------------"

# Check all services are running
all_running=true
check_service 8102 "Agent Designer" || all_running=false
check_service 8103 "Workflow Engine" || all_running=false
check_service 8104 "Integration Hub" || all_running=false
check_service 8105 "AI Model Service" || all_running=false
check_service 8106 "Auth Service" || all_running=false
check_service 8107 "WebSocket Service" || all_running=false

if [ "$all_running" = false ]; then
    echo -e "${RED}⚠️  Some services failed to start. Check logs:${NC}"
    echo "  tail -f /tmp/agent_designer.log"
    echo "  tail -f /tmp/workflow_engine.log"
    echo "  tail -f /tmp/integration_hub.log"
    echo "  tail -f /tmp/ai_model.log"
    echo "  tail -f /tmp/auth.log"
    echo "  tail -f /tmp/websocket.log"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 5: Starting API Gateway (Updated)${NC}"
echo "---------------------------------------"

# Stop existing API Gateway
kill_port 8000

# Start API Gateway with updated configuration
echo "Starting API Gateway (Port 8000)..."
python api_gateway_integrated.py > /tmp/api_gateway.log 2>&1 &
sleep 3

check_service 8000 "API Gateway"

echo ""
echo -e "${YELLOW}Step 6: Starting Monitoring Dashboard${NC}"
echo "-------------------------------------"

# Stop existing dashboard
kill_port 8051

echo "Starting Dashboard (Port 8051)..."
streamlit run monitoring_dashboard_integrated.py --server.port 8051 --server.address localhost > /tmp/dashboard.log 2>&1 &
sleep 3

check_service 8051 "Monitoring Dashboard"

echo ""
echo "========================================="
echo -e "${GREEN}✨ Lightning System Ready!${NC}"
echo "========================================="
echo ""
echo "Service URLs:"
echo "  • Dashboard:       http://localhost:8051"
echo "  • API Gateway:     http://localhost:8000"
echo "  • Agent Designer:  http://localhost:8102"
echo "  • Workflow Engine: http://localhost:8103"
echo "  • Integration Hub: http://localhost:8104"
echo "  • AI Model:        http://localhost:8105"
echo "  • Auth Service:    http://localhost:8106"
echo "  • WebSocket:       ws://localhost:8107/ws"
echo ""
echo "Useful Commands:"
echo "  • Check status:    ./start_integrated_system.sh status"
echo "  • Stop all:        ./start_integrated_system.sh stop"
echo "  • View logs:       tail -f /tmp/*.log"
echo ""
echo "Test the system:"
echo "  1. Open http://localhost:8051 in your browser"
echo "  2. Try creating an agent or running a workflow"
echo ""
echo -e "${GREEN}🚀 System is fully integrated and operational!${NC}"