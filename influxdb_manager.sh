#!/bin/bash

# InfluxDB Manager Script
# Manages InfluxDB and Grafana containers for the Agent System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env.influxdb ]; then
    export $(cat .env.influxdb | grep -v '^#' | xargs)
fi

function print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

function print_error() {
    echo -e "${RED}✗${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

function start_services() {
    echo "Starting InfluxDB and Grafana..."
    
    # Start containers
    docker-compose up -d
    
    # Wait for InfluxDB to be ready
    echo "Waiting for InfluxDB to be ready..."
    sleep 5
    
    # Check if InfluxDB is running
    if curl -s http://localhost:8086/health | grep -q "pass"; then
        print_status "InfluxDB is running at http://localhost:8086"
    else
        print_error "InfluxDB failed to start"
        exit 1
    fi
    
    # Check if Grafana is running
    if curl -s http://localhost:3000/api/health | grep -q "ok"; then
        print_status "Grafana is running at http://localhost:3000"
    else
        print_warning "Grafana is still starting..."
    fi
    
    echo ""
    print_status "Services started successfully!"
    echo ""
    echo "Access URLs:"
    echo "  InfluxDB: http://localhost:8086"
    echo "    Username: admin"
    echo "    Password: supersecret123"
    echo ""
    echo "  Grafana: http://localhost:3000"
    echo "    Username: admin"
    echo "    Password: admin123"
    echo ""
    echo "Environment variables have been saved to .env.influxdb"
}

function stop_services() {
    echo "Stopping InfluxDB and Grafana..."
    docker-compose down
    print_status "Services stopped"
}

function restart_services() {
    stop_services
    sleep 2
    start_services
}

function status_services() {
    echo "Service Status:"
    echo ""
    
    # Check InfluxDB
    if docker ps | grep -q agent-influxdb; then
        print_status "InfluxDB is running"
        
        # Check health
        if curl -s http://localhost:8086/health | grep -q "pass"; then
            echo "    Health: PASS"
        else
            echo "    Health: FAIL"
        fi
    else
        print_error "InfluxDB is not running"
    fi
    
    # Check Grafana
    if docker ps | grep -q agent-grafana; then
        print_status "Grafana is running"
    else
        print_error "Grafana is not running"
    fi
    
    echo ""
    echo "Container Details:"
    docker ps --filter "name=agent-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

function logs_services() {
    echo "Showing logs (press Ctrl+C to exit)..."
    docker-compose logs -f
}

function clean_services() {
    echo "This will remove all containers and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        docker-compose down -v
        print_status "All services and data removed"
    else
        echo "Cancelled"
    fi
}

function test_connection() {
    echo "Testing InfluxDB connection..."
    
    # Test InfluxDB API
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Token ${INFLUXDB_TOKEN}" \
        "${INFLUXDB_URL}/api/v2/buckets?org=${INFLUXDB_ORG}")
    
    if [ "$response" = "200" ]; then
        print_status "Successfully connected to InfluxDB"
        
        # Show buckets
        echo ""
        echo "Available buckets:"
        curl -s -H "Authorization: Token ${INFLUXDB_TOKEN}" \
            "${INFLUXDB_URL}/api/v2/buckets?org=${INFLUXDB_ORG}" | \
            jq -r '.buckets[] | "  - \(.name)"' 2>/dev/null || echo "  (install jq to see bucket list)"
    else
        print_error "Failed to connect to InfluxDB (HTTP $response)"
        echo "Make sure InfluxDB is running and credentials are correct"
    fi
}

function create_sample_data() {
    echo "Inserting sample performance metrics..."
    
    # Create sample data point
    timestamp=$(date +%s%N)
    data="performance_metrics,host=agent-system,metric_type=cpu_usage value=45.2 ${timestamp}"
    
    # Send to InfluxDB
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${INFLUXDB_URL}/api/v2/write?org=${INFLUXDB_ORG}&bucket=${INFLUXDB_BUCKET}" \
        -H "Authorization: Token ${INFLUXDB_TOKEN}" \
        -H "Content-Type: text/plain; charset=utf-8" \
        --data-raw "$data")
    
    if [ "$response" = "204" ]; then
        print_status "Sample data inserted successfully"
    else
        print_error "Failed to insert sample data (HTTP $response)"
    fi
}

# Main menu
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        status_services
        ;;
    logs)
        logs_services
        ;;
    clean)
        clean_services
        ;;
    test)
        test_connection
        ;;
    sample)
        create_sample_data
        ;;
    *)
        echo "InfluxDB Manager for Agent System"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|clean|test|sample}"
        echo ""
        echo "Commands:"
        echo "  start    - Start InfluxDB and Grafana containers"
        echo "  stop     - Stop all containers"
        echo "  restart  - Restart all containers"
        echo "  status   - Show container status"
        echo "  logs     - Show container logs"
        echo "  clean    - Remove all containers and volumes"
        echo "  test     - Test InfluxDB connection"
        echo "  sample   - Insert sample data"
        ;;
esac