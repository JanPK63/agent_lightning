#!/bin/bash

# Agent Lightning Monitoring Stack Startup Script
# This script provides an easy way for non-IT users to start the monitoring stack

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not installed."
        exit 1
    fi
}

# Function to wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to start up..."

    # Wait for Prometheus
    print_status "Waiting for Prometheus..."
    for i in {1..30}; do
        if curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
            print_success "Prometheus is ready!"
            break
        fi
        sleep 2
    done

    # Wait for Grafana
    print_status "Waiting for Grafana..."
    for i in {1..30}; do
        if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
            print_success "Grafana is ready!"
            break
        fi
        sleep 2
    done

    # Wait for Alertmanager
    print_status "Waiting for Alertmanager..."
    for i in {1..30}; do
        if curl -s http://localhost:9093/-/healthy >/dev/null 2>&1; then
            print_success "Alertmanager is ready!"
            break
        fi
        sleep 2
    done
}

# Function to display access information
show_access_info() {
    echo ""
    echo "=================================================="
    echo "🎉 Agent Lightning Monitoring Stack is Running!"
    echo "=================================================="
    echo ""
    echo "📊 Access your monitoring dashboards:"
    echo ""
    echo "   🌐 Grafana (Main Dashboard):"
    echo "      http://localhost:3000"
    echo "      Username: admin"
    echo "      Password: admin123"
    echo ""
    echo "   📈 Prometheus (Metrics):"
    echo "      http://localhost:9090"
    echo ""
    echo "   🚨 Alertmanager (Alerts):"
    echo "      http://localhost:9093"
    echo ""
    echo "   📊 Node Exporter (System Metrics):"
    echo "      http://localhost:9100"
    echo ""
    echo "=================================================="
    echo ""
    print_warning "⚠️  Important: Change the default Grafana password after first login!"
    echo ""
    print_status "To stop monitoring: ./stop_monitoring.sh"
    echo ""
}

# Function to start services
start_services() {
    print_status "Starting Agent Lightning Monitoring Stack..."

    # Use docker compose (newer versions) or docker-compose (older versions)
    if docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi

    # Start services
    $DOCKER_COMPOSE_CMD -f docker-compose.monitoring.yml up -d

    if [ $? -eq 0 ]; then
        print_success "Monitoring services started successfully!"
        wait_for_services
        show_access_info
    else
        print_error "Failed to start monitoring services."
        exit 1
    fi
}

# Function to check service status
check_status() {
    print_status "Checking monitoring services status..."

    echo ""
    if docker compose version >/dev/null 2>&1; then
        docker compose -f docker-compose.monitoring.yml ps
    else
        docker-compose -f docker-compose.monitoring.yml ps
    fi
    echo ""
}

# Function to show logs
show_logs() {
    print_status "Showing monitoring services logs..."
    echo "Press Ctrl+C to stop viewing logs"
    echo ""

    if docker compose version >/dev/null 2>&1; then
        docker compose -f docker-compose.monitoring.yml logs -f
    else
        docker-compose -f docker-compose.monitoring.yml logs -f
    fi
}

# Main script logic
case "${1:-start}" in
    "start")
        print_status "Agent Lightning Monitoring Stack"
        print_status "=================================="
        check_docker
        check_docker_compose
        start_services
        ;;
    "status")
        check_status
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        print_status "Stopping monitoring services..."
        if docker compose version >/dev/null 2>&1; then
            docker compose -f docker-compose.monitoring.yml down
        else
            docker-compose -f docker-compose.monitoring.yml down
        fi
        print_success "Monitoring services stopped."
        ;;
    "restart")
        print_status "Restarting monitoring services..."
        if docker compose version >/dev/null 2>&1; then
            docker compose -f docker-compose.monitoring.yml restart
        else
            docker-compose -f docker-compose.monitoring.yml restart
        fi
        print_success "Monitoring services restarted."
        ;;
    "help"|"-h"|"--help")
        echo "Agent Lightning Monitoring Stack Control Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start   - Start the monitoring stack (default)"
        echo "  stop    - Stop the monitoring stack"
        echo "  restart - Restart the monitoring stack"
        echo "  status  - Show status of monitoring services"
        echo "  logs    - Show logs from monitoring services"
        echo "  help    - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 start    # Start monitoring"
        echo "  $0 status   # Check if services are running"
        echo "  $0 logs     # View service logs"
        echo "  $0 stop     # Stop monitoring"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for available commands."
        exit 1
        ;;
esac