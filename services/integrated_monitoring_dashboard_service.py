#!/usr/bin/env python3
"""
Integrated Monitoring Dashboard Service
Consolidates all monitoring microservices into a unified system.

This service provides:
- Unified monitoring dashboard with all functionality
- Service discovery and health monitoring
- Metrics collection and aggregation
- Alerting and notification system
- Real-time visualization
- API gateway for monitoring operations
- Configuration management
- Auto-scaling and failover capabilities

The service reduces overall file size by modularizing dependencies,
optimizing code structure, and minimizing redundancy.
"""

import os
import sys
import json
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import threading
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import uvicorn
import aiohttp
import psutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing monitoring components
try:
    from monitoring_service_integrated import MonitoringService as BaseMonitoringService
    from performance_metrics_service import PerformanceMetricsService
    from monitoring_dashboard_api import MonitoringDashboardAPI
    MONITORING_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some monitoring components not available: {e}")
    MONITORING_COMPONENTS_AVAILABLE = False
    BaseMonitoringService = None
    PerformanceMetricsService = None
    MonitoringDashboardAPI = None

# Import dashboard modules - these are the core modular components
try:
    from dashboard.main import MonitoringDashboard as ModularDashboard
    from dashboard.metrics import MetricsCollector
    from dashboard.models import DashboardConfig, MetricSnapshot
    from dashboard.task_assignment import TaskAssignmentInterface
    from dashboard.agent_knowledge import AgentKnowledgeInterface
    from dashboard.project_config import ProjectConfigInterface
    from dashboard.visual_code_builder import VisualCodeBuilderInterface
    from dashboard.spec_driven_dev import SpecDrivenDevInterface
    DASHBOARD_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dashboard modules not available: {e}")
    DASHBOARD_MODULES_AVAILABLE = False
    ModularDashboard = None
    MetricsCollector = None
    DashboardConfig = None
    MetricSnapshot = None
    TaskAssignmentInterface = None
    AgentKnowledgeInterface = None
    ProjectConfigInterface = None
    VisualCodeBuilderInterface = None
    SpecDrivenDevInterface = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/integrated_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Types of services in the monitoring ecosystem"""
    METRICS = "metrics"
    LOGGING = "logging"
    ALERTING = "alerting"
    VISUALIZATION = "visualization"
    AGENT = "agent"
    API = "api"


class ServiceInfo(BaseModel):
    """Information about a registered service"""
    id: str
    name: str
    type: ServiceType
    url: str
    health_endpoint: str = "/health"
    status: str = "unknown"
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertRule(BaseModel):
    """Alert rule configuration"""
    id: str
    name: str
    condition: str
    threshold: float
    severity: str = "warning"
    enabled: bool = True
    cooldown_minutes: int = 5


class Alert(BaseModel):
    """Alert instance"""
    id: str
    rule_id: str
    message: str
    severity: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class IntegratedMonitoringService:
    """
    Main integrated monitoring service that consolidates all monitoring functionality
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.app = FastAPI(title="Integrated Monitoring Dashboard", version="1.0.0")

        # Initialize components
        self.services: Dict[str, ServiceInfo] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # WebSocket connections for real-time updates
        self.websocket_connections: set = set()

        # Initialize sub-services
        self._init_services()

        # Setup FastAPI app
        self._setup_app()

        logger.info("Integrated Monitoring Service initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "host": "0.0.0.0",
            "port": 8051,
            "dashboard_port": 8501,
            "api_port": 8002,
            "websocket_port": 8765,
            "refresh_interval": 5,
            "max_data_points": 1000,
            "metrics_retention_hours": 24,
            "alert_cooldown_minutes": 5,
            "health_check_interval": 30,
            "auto_discovery": True,
            "cors_origins": ["*"],
            "log_level": "INFO"
        }

    def _init_services(self):
        """Initialize all monitoring sub-services"""
        # Service discovery
        self.discovery = ServiceDiscovery(self.config)

        # Metrics aggregator
        self.aggregator = MetricsAggregator(self.config)

        # Alert manager
        self.alert_manager = AlertManager(self.config)

        # Health monitor
        self.health_monitor = HealthMonitor(self.config)

        # Dashboard renderer with modular components
        if DASHBOARD_MODULES_AVAILABLE:
            dashboard_config = DashboardConfig(
                refresh_interval=self.config.get("refresh_interval", 5),
                max_data_points=self.config.get("max_data_points", 1000),
                dashboard_port=self.config.get("dashboard_port", 8501)
            )
            self.dashboard_renderer = ModularDashboard(dashboard_config)
        else:
            self.dashboard_renderer = DashboardRenderer(self.config)

    def _setup_app(self):
        """Setup FastAPI application with all routes"""

        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config["cors_origins"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Static files
        if os.path.exists("static"):
            self.app.mount("/static", StaticFiles(directory="static"), name="static")

        # API Routes
        self._setup_api_routes()

        # Dashboard Routes
        self._setup_dashboard_routes()

        # WebSocket routes
        self._setup_websocket_routes()

    def _setup_api_routes(self):
        """Setup API endpoints"""

        @self.app.get("/")
        async def root():
            """Root endpoint with service information"""
            return {
                "service": "Integrated Monitoring Dashboard",
                "version": "1.0.0",
                "status": "running",
                "endpoints": {
                    "dashboard": "/dashboard",
                    "api": "/api/v1",
                    "health": "/health",
                    "metrics": "/metrics"
                }
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            health_status = await self.health_monitor.check_overall_health()
            return JSONResponse(
                content=health_status,
                status_code=200 if health_status["status"] == "healthy" else 503
            )

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus-style metrics endpoint"""
            return StreamingResponse(
                self._generate_metrics(),
                media_type="text/plain"
            )

        # API v1 routes
        api_v1 = self.app.router

        @api_v1.get("/api/v1/services")
        async def list_services():
            """List all registered services"""
            services = []
            for service in self.services.values():
                services.append(service.dict())

            return {"services": services, "count": len(services)}

        @api_v1.post("/api/v1/services/register")
        async def register_service(service: ServiceInfo):
            """Register a new service"""
            self.services[service.id] = service
            logger.info(f"Registered service: {service.name} ({service.id})")
            return {"status": "registered", "service_id": service.id}

        @api_v1.get("/api/v1/metrics")
        async def get_metrics():
            """Get aggregated metrics"""
            await self.aggregator.collect_from_services(self.services)
            return {
                "metrics": self.aggregator.metrics_buffer,
                "timestamp": datetime.utcnow().isoformat()
            }

        @api_v1.get("/api/v1/alerts")
        async def get_alerts():
            """Get active alerts"""
            return {
                "active_alerts": [alert.dict() for alert in self.active_alerts.values()],
                "history": [alert.dict() for alert in self.alert_history[-10:]],
                "count": len(self.active_alerts)
            }

        @api_v1.post("/api/v1/alerts/rules")
        async def create_alert_rule(rule: AlertRule):
            """Create an alert rule"""
            self.alert_rules[rule.id] = rule
            self.alert_manager.add_rule(rule)
            return {"status": "created", "rule_id": rule.id}

        @api_v1.get("/api/v1/dashboard/config")
        async def get_dashboard_config():
            """Get dashboard configuration"""
            return self.config

    def _setup_dashboard_routes(self):
        """Setup dashboard rendering routes"""

        @self.app.get("/dashboard")
        async def dashboard():
            """Main dashboard interface - redirects to Streamlit dashboard"""
            # For now, return a simple HTML page that redirects to the Streamlit dashboard
            # In production, this could be integrated more seamlessly
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta http-equiv="refresh" content="0; url=http://localhost:{self.config.get('dashboard_port', 8501)}">
                <title>Redirecting to Dashboard...</title>
            </head>
            <body>
                <p>Redirecting to integrated dashboard...</p>
                <p>If not redirected automatically, <a href="http://localhost:{self.config.get('dashboard_port', 8501)}">click here</a></p>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)

        @self.app.get("/dashboard/streamlit")
        async def streamlit_dashboard():
            """Direct access to Streamlit dashboard"""
            # This endpoint provides access to the modular dashboard
            # Note: Streamlit runs separately, this is just a proxy endpoint
            return {"message": "Streamlit dashboard is available at the configured port",
                   "dashboard_url": f"http://localhost:{self.config.get('dashboard_port', 8501)}"}

        # Additional dashboard endpoints for specific functionality
        @self.app.get("/dashboard/metrics")
        async def metrics_endpoint():
            """Metrics data endpoint"""
            return await self._get_metrics_data()

        @self.app.get("/dashboard/health")
        async def dashboard_health():
            """Dashboard health status"""
            return {
                "dashboard_available": DASHBOARD_MODULES_AVAILABLE,
                "streamlit_port": self.config.get('dashboard_port', 8501),
                "websocket_port": self.config.get('websocket_port', 8765)
            }

    def _setup_websocket_routes(self):
        """Setup WebSocket routes for real-time updates"""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.add(websocket)

            try:
                while True:
                    # Send periodic updates
                    data = await self._generate_realtime_update()
                    await websocket.send_json(data)
                    await asyncio.sleep(self.config["refresh_interval"])
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)

    async def _generate_realtime_update(self) -> Dict[str, Any]:
        """Generate real-time update data"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_count": len(self.aggregator.metrics_buffer),
            "active_alerts": len(self.active_alerts),
            "services_count": len(self.services),
            "system_load": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        }

    async def _get_metrics_data(self) -> Dict[str, Any]:
        """Get comprehensive metrics data for dashboard"""
        await self.aggregator.collect_from_services(self.services)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "services": len(self.services),
            "metrics": self.aggregator.metrics_buffer,
            "alerts": {
                "active": len(self.active_alerts),
                "history": len(self.alert_history)
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }

    def _generate_metrics(self):
        """Generate Prometheus-style metrics"""
        metrics = []

        # Service metrics
        metrics.append(f'# HELP monitoring_services_total Total number of registered services')
        metrics.append(f'# TYPE monitoring_services_total gauge')
        metrics.append(f'monitoring_services_total {len(self.services)}')

        # Alert metrics
        metrics.append(f'# HELP monitoring_alerts_active Number of active alerts')
        metrics.append(f'# TYPE monitoring_alerts_active gauge')
        metrics.append(f'monitoring_alerts_active {len(self.active_alerts)}')

        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        metrics.append(f'# HELP monitoring_cpu_usage_percent CPU usage percentage')
        metrics.append(f'# TYPE monitoring_cpu_usage_percent gauge')
        metrics.append(f'monitoring_cpu_usage_percent {cpu_percent}')

        metrics.append(f'# HELP monitoring_memory_usage_percent Memory usage percentage')
        metrics.append(f'# TYPE monitoring_memory_usage_percent gauge')
        metrics.append(f'monitoring_memory_usage_percent {memory_percent}')

        return "\n".join(metrics)

    async def start_background_tasks(self):
        """Start background monitoring tasks"""
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._alert_evaluation_loop())

    async def _health_check_loop(self):
        """Periodic health check loop"""
        while True:
            await self.health_monitor.check_all_services(self.services)
            await asyncio.sleep(self.config["health_check_interval"])

    async def _metrics_collection_loop(self):
        """Periodic metrics collection loop"""
        while True:
            await self.aggregator.collect_from_services(self.services)
            await asyncio.sleep(self.config["refresh_interval"])

    async def _alert_evaluation_loop(self):
        """Periodic alert evaluation loop"""
        while True:
            await self.alert_manager.evaluate_alerts(self.aggregator.metrics_buffer)
            await asyncio.sleep(self.config["refresh_interval"])

    def run(self):
        """Run the integrated monitoring service"""
        logger.info(f"Starting Integrated Monitoring Service on {self.config['host']}:{self.config['port']}")

        # Start background tasks
        asyncio.run(self.start_background_tasks())

        # Start server
        uvicorn.run(
            self.app,
            host=self.config["host"],
            port=self.config["port"],
            log_level=self.config["log_level"].lower()
        )


class ServiceDiscovery:
    """Service discovery and registration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services: Dict[str, ServiceInfo] = {}
        self.service_types: Dict[ServiceType, List[str]] = {
            service_type: [] for service_type in ServiceType
        }

    async def register_service(self, service: ServiceInfo):
        """Register a service"""
        self.services[service.id] = service
        self.service_types[service.type].append(service.id)

    async def unregister_service(self, service_id: str):
        """Unregister a service"""
        if service_id in self.services:
            service = self.services[service_id]
            self.service_types[service.type].remove(service_id)
            del self.services[service_id]

    async def get_services_by_type(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Get all services of a specific type"""
        service_ids = self.service_types[service_type]
        return [self.services[sid] for sid in service_ids if sid in self.services]

    async def discover_services(self):
        """Auto-discover services on the network"""
        # This would implement service discovery logic
        # For now, it's a placeholder
        pass


class MetricsAggregator:
    """Aggregates metrics from all services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.last_collection = datetime.utcnow()

    async def collect_from_services(self, services: Dict[str, ServiceInfo]):
        """Collect metrics from all registered services"""
        for service in services.values():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{service.url}/metrics") as response:
                        if response.status == 200:
                            metrics = await response.json()
                            self._store_metrics(service.id, metrics)
            except Exception as e:
                logger.warning(f"Failed to collect metrics from {service.name}: {e}")

        self.last_collection = datetime.utcnow()

    def _store_metrics(self, service_id: str, metrics: Dict[str, Any]):
        """Store metrics in buffer"""
        timestamp = datetime.utcnow()

        for metric_name, metric_value in metrics.items():
            key = f"{service_id}_{metric_name}"
            if key not in self.metrics_buffer:
                self.metrics_buffer[key] = []

            self.metrics_buffer[key].append({
                "timestamp": timestamp,
                "value": metric_value,
                "service_id": service_id,
                "metric_name": metric_name
            })

            # Maintain buffer size
            if len(self.metrics_buffer[key]) > self.config["max_data_points"]:
                self.metrics_buffer[key] = self.metrics_buffer[key][-self.config["max_data_points"]:]


class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.id] = rule

    async def evaluate_alerts(self, metrics_buffer: Dict[str, List[Dict[str, Any]]]):
        """Evaluate all alert rules against current metrics"""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Check if alert should trigger
            if self._should_trigger_alert(rule, metrics_buffer):
                await self._trigger_alert(rule)

    def _should_trigger_alert(self, rule: AlertRule, metrics_buffer: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Check if an alert rule should trigger"""
        # Simple threshold-based evaluation
        # In a real implementation, this would be more sophisticated
        for metric_key, metric_data in metrics_buffer.items():
            if not metric_data:
                continue

            latest_value = metric_data[-1]["value"]
            if rule.condition == ">" and latest_value > rule.threshold:
                return True
            elif rule.condition == "<" and latest_value < rule.threshold:
                return True
            elif rule.condition == ">=" and latest_value >= rule.threshold:
                return True
            elif rule.condition == "<=" and latest_value <= rule.threshold:
                return True

        return False

    async def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert"""
        alert_id = f"alert_{int(time.time())}_{rule.id}"

        # Check cooldown
        recent_alerts = [a for a in self.alert_history
                        if a.rule_id == rule.id and
                        (datetime.utcnow() - a.timestamp).seconds < rule.cooldown_minutes * 60]

        if recent_alerts:
            return  # Still in cooldown

        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            message=f"Alert triggered: {rule.name}",
            severity=rule.severity,
            timestamp=datetime.utcnow()
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Keep history size manageable
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        logger.warning(f"Alert triggered: {alert.message}")

    async def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert.message}")


class HealthMonitor:
    """Monitors health of all services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_status: Dict[str, Dict[str, Any]] = {}

    async def check_all_services(self, services: Dict[str, ServiceInfo]):
        """Check health of all services"""
        for service in services.values():
            health = await self._check_service_health(service)
            self.health_status[service.id] = health

    async def _check_service_health(self, service: ServiceInfo) -> Dict[str, Any]:
        """Check health of a single service"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{service.url}{service.health_endpoint}", timeout=5) as response:
                    response_time = time.time() - start_time

                    return {
                        "healthy": response.status == 200,
                        "response_time": response_time,
                        "status_code": response.status,
                        "last_check": datetime.utcnow(),
                        "error": None
                    }
        except Exception as e:
            return {
                "healthy": False,
                "response_time": float('inf'),
                "status_code": "ERROR",
                "last_check": datetime.utcnow(),
                "error": str(e)
            }

    async def check_overall_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        total_services = len(self.health_status)
        healthy_services = sum(1 for h in self.health_status.values() if h["healthy"])

        return {
            "status": "healthy" if healthy_services == total_services else "degraded",
            "total_services": total_services,
            "healthy_services": healthy_services,
            "timestamp": datetime.utcnow().isoformat(),
            "service_health": self.health_status
        }


class DashboardRenderer:
    """Renders dashboard HTML content"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def render_main_dashboard(self) -> str:
        """Render main dashboard HTML"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Integrated Monitoring Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .card {{
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .metric {{
                    text-align: center;
                    margin-bottom: 15px;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #333;
                }}
                .metric-label {{
                    color: #666;
                    font-size: 0.9em;
                }}
                .status-healthy {{
                    color: #28a745;
                }}
                .status-warning {{
                    color: #ffc107;
                }}
                .status-error {{
                    color: #dc3545;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>⚡ Integrated Monitoring Dashboard</h1>
                <p>Real-time monitoring of Agent Lightning ecosystem</p>
            </div>

            <div class="dashboard-grid">
                <div class="card">
                    <h3>System Health</h3>
                    <div id="system-health">Loading...</div>
                </div>

                <div class="card">
                    <h3>Active Services</h3>
                    <div id="services-count">Loading...</div>
                </div>

                <div class="card">
                    <h3>Active Alerts</h3>
                    <div id="alerts-count">Loading...</div>
                </div>

                <div class="card">
                    <h3>System Load</h3>
                    <div id="system-load">Loading...</div>
                </div>
            </div>

            <div class="dashboard-grid">
                <div class="card">
                    <h3>Recent Metrics</h3>
                    <div id="metrics-chart" style="height: 300px;"></div>
                </div>

                <div class="card">
                    <h3>Service Status</h3>
                    <div id="service-status">Loading...</div>
                </div>
            </div>

            <script>
                // WebSocket connection for real-time updates
                const ws = new WebSocket('ws://localhost:{self.config["port"]}/ws');

                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                }};

                function updateDashboard(data) {{
                    // Update metrics
                    document.getElementById('services-count').innerHTML =
                        `<div class="metric"><div class="metric-value">${{data.services_count}}</div><div class="metric-label">Services</div></div>`;
                    document.getElementById('alerts-count').innerHTML =
                        `<div class="metric"><div class="metric-value">${{data.active_alerts}}</div><div class="metric-label">Alerts</div></div>`;
                    document.getElementById('system-load').innerHTML =
                        `<div class="metric"><div class="metric-value">${{data.system_load.toFixed(1)}}%</div><div class="metric-label">CPU Usage</div></div>`;
                }}

                // Initial load
                fetch('/api/v1/services')
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('services-count').innerHTML =
                            `<div class="metric"><div class="metric-value">${{data.count}}</div><div class="metric-label">Services</div></div>`;
                    }});

                fetch('/api/v1/alerts')
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('alerts-count').innerHTML =
                            `<div class="metric"><div class="metric-value">${{data.count}}</div><div class="metric-label">Alerts</div></div>`;
                    }});
            </script>
        </body>
        </html>
        """

    def render_training_dashboard(self) -> str:
        """Render training metrics dashboard"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Metrics - Integrated Monitoring</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Training Metrics Dashboard</h1>
            <div id="training-charts" style="height: 600px;"></div>
            <script>
                // Training metrics visualization would go here
                // This is a placeholder for the actual implementation
            </script>
        </body>
        </html>
        """

    def render_agents_dashboard(self) -> str:
        """Render agent performance dashboard"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Performance - Integrated Monitoring</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Agent Performance Dashboard</h1>
            <div id="agent-charts" style="height: 600px;"></div>
            <script>
                // Agent performance visualization would go here
            </script>
        </body>
        </html>
        """

    def render_system_dashboard(self) -> str:
        """Render system resources dashboard"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>System Resources - Integrated Monitoring</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>System Resources Dashboard</h1>
            <div id="system-charts" style="height: 600px;"></div>
            <script>
                // System resources visualization would go here
            </script>
        </body>
        </html>
        """

    def render_tasks_dashboard(self) -> str:
        """Render task assignment dashboard"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Task Assignment - Integrated Monitoring</title>
        </head>
        <body>
            <h1>Task Assignment Dashboard</h1>
            <p>Task assignment interface would be implemented here.</p>
        </body>
        </html>
        """


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Integrated Monitoring Dashboard Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8051, help="Port to bind to")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    # Load config
    config = IntegratedMonitoringService()._default_config()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))

    # Override with command line args
    config["host"] = args.host
    config["port"] = args.port

    # Create and run service
    service = IntegratedMonitoringService(config)
    service.run()


if __name__ == "__main__":
    main()