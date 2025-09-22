#!/bin/bash

# Integrated Monitoring Dashboard Service Deployment Script
# This script deploys the unified monitoring service for Agent Lightning

set -e

# Configuration
SERVICE_NAME="integrated_monitoring"
CONFIG_FILE="config/integrated_monitoring_config.json"
SERVICE_FILE="services/integrated_monitoring_dashboard_service.py"
LOG_DIR="logs"
VENV_DIR=".venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.10+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
        log_success "Python $PYTHON_VERSION is compatible"
    else
        log_error "Python $PYTHON_VERSION is not supported. Please use Python 3.10+"
        exit 1
    fi

    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file $CONFIG_FILE not found"
        exit 1
    fi

    # Check if service file exists
    if [ ! -f "$SERVICE_FILE" ]; then
        log_error "Service file $SERVICE_FILE not found"
        exit 1
    fi
}

setup_environment() {
    log_info "Setting up environment..."

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    if [ -f "requirements.txt" ]; then
        log_info "Installing dependencies..."
        pip install -r requirements.txt
    else
        log_warning "requirements.txt not found. Installing basic dependencies..."
        pip install fastapi uvicorn aiohttp psutil pydantic
    fi

    # Install the service itself (if it's a package)
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        pip install -e .
    fi
}

start_service() {
    log_info "Starting Integrated Monitoring Service..."

    # Load configuration
    if [ -f "$CONFIG_FILE" ]; then
        HOST=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['server']['host'])")
        PORT=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['server']['port'])")
    else
        HOST="0.0.0.0"
        PORT="8051"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Start service
    log_info "Starting service on $HOST:$PORT"
    nohup python3 "$SERVICE_FILE" --host "$HOST" --port "$PORT" --config "$CONFIG_FILE" > "$LOG_DIR/service.log" 2>&1 &

    SERVICE_PID=$!
    echo $SERVICE_PID > "$LOG_DIR/service.pid"

    log_success "Service started with PID: $SERVICE_PID"
    log_info "Logs available at: $LOG_DIR/service.log"
    log_info "Service URL: http://$HOST:$PORT"
    log_info "Dashboard URL: http://$HOST:$PORT/dashboard"
}

stop_service() {
    log_info "Stopping Integrated Monitoring Service..."

    if [ -f "$LOG_DIR/service.pid" ]; then
        SERVICE_PID=$(cat "$LOG_DIR/service.pid")

        if kill -0 "$SERVICE_PID" 2>/dev/null; then
            log_info "Stopping service (PID: $SERVICE_PID)..."
            kill "$SERVICE_PID"

            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done

            # Force kill if still running
            if kill -0 "$SERVICE_PID" 2>/dev/null; then
                log_warning "Service didn't stop gracefully, force killing..."
                kill -9 "$SERVICE_PID"
            fi
        else
            log_warning "Service PID $SERVICE_PID not found or already stopped"
        fi

        rm -f "$LOG_DIR/service.pid"
        log_success "Service stopped"
    else
        log_warning "Service PID file not found. Service may not be running."
    fi
}

restart_service() {
    log_info "Restarting Integrated Monitoring Service..."
    stop_service
    sleep 2
    start_service
}

check_status() {
    log_info "Checking service status..."

    if [ -f "$LOG_DIR/service.pid" ]; then
        SERVICE_PID=$(cat "$LOG_DIR/service.pid")

        if kill -0 "$SERVICE_PID" 2>/dev/null; then
            log_success "Service is running (PID: $SERVICE_PID)"

            # Load configuration to check URLs
            if [ -f "$CONFIG_FILE" ]; then
                HOST=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['server']['host'])")
                PORT=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['server']['port'])")
                log_info "Service URL: http://$HOST:$PORT"
                log_info "Dashboard URL: http://$HOST:$PORT/dashboard"
                log_info "API URL: http://$HOST:$PORT/api/v1"
            fi
        else
            log_error "Service PID $SERVICE_PID exists but process is not running"
            rm -f "$LOG_DIR/service.pid"
        fi
    else
        log_warning "Service is not running (no PID file found)"
    fi
}

show_logs() {
    if [ -f "$LOG_DIR/service.log" ]; then
        log_info "Showing recent logs (last 50 lines):"
        echo "----------------------------------------"
        tail -50 "$LOG_DIR/service.log"
        echo "----------------------------------------"
        log_info "Full logs available at: $LOG_DIR/service.log"
    else
        log_warning "Log file not found"
    fi
}

cleanup() {
    log_info "Cleaning up..."

    # Stop service if running
    if [ -f "$LOG_DIR/service.pid" ]; then
        stop_service
    fi

    # Remove log files
    if [ -d "$LOG_DIR" ]; then
        rm -rf "$LOG_DIR"
        log_info "Removed log directory"
    fi

    # Remove virtual environment
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        log_info "Removed virtual environment"
    fi

    log_success "Cleanup completed"
}

# Main script logic
case "${1:-help}" in
    "start")
        check_dependencies
        setup_environment
        start_service
        ;;
    "stop")
        stop_service
        ;;
    "restart")
        restart_service
        ;;
    "status")
        check_status
        ;;
    "logs")
        show_logs
        ;;
    "cleanup")
        cleanup
        ;;
    "setup")
        check_dependencies
        setup_environment
        log_success "Environment setup completed"
        ;;
    "help"|*)
        echo "Integrated Monitoring Dashboard Service Control Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start    - Start the monitoring service"
        echo "  stop     - Stop the monitoring service"
        echo "  restart  - Restart the monitoring service"
        echo "  status   - Check service status"
        echo "  logs     - Show service logs"
        echo "  setup    - Setup environment without starting service"
        echo "  cleanup  - Remove all service files and stop service"
        echo "  help     - Show this help message"
        echo ""
        echo "Configuration:"
        echo "  Config file: $CONFIG_FILE"
        echo "  Service file: $SERVICE_FILE"
        echo "  Log directory: $LOG_DIR"
        echo "  Virtual env: $VENV_DIR"
        ;;
esac