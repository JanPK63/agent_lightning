#!/usr/bin/env python3
"""
Enterprise Monitoring Service - Integrated with InfluxDB and Grafana
Provides comprehensive observability for the entire AI Agent system
Based on SA-008: Complete System Integration
"""

import os
import sys
import json
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque, defaultdict
import statistics

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import psutil
import aiohttp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared data access layer
from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ServiceStatus(str, Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# Pydantic models
class MetricSubmit(BaseModel):
    """Submit metric data"""
    service: str = Field(description="Service name")
    metric_name: str = Field(description="Metric name")
    value: float = Field(description="Metric value")
    tags: Optional[Dict[str, str]] = Field(default=None, description="Metric tags")
    timestamp: Optional[str] = Field(default=None, description="Timestamp")


class AlertRule(BaseModel):
    """Alert rule configuration"""
    name: str = Field(description="Rule name")
    metric: str = Field(description="Metric to monitor")
    condition: str = Field(description="Condition (gt, lt, eq)")
    threshold: float = Field(description="Threshold value")
    duration: int = Field(default=60, description="Duration in seconds")
    severity: AlertSeverity = Field(description="Alert severity")
    notification_channels: List[str] = Field(default_factory=list, description="Notification channels")


class DashboardCreate(BaseModel):
    """Create monitoring dashboard"""
    name: str = Field(description="Dashboard name")
    description: str = Field(description="Dashboard description")
    panels: List[Dict[str, Any]] = Field(description="Dashboard panels configuration")
    refresh_interval: int = Field(default=30, description="Refresh interval in seconds")


class ServiceHealthCheck(BaseModel):
    """Service health check configuration"""
    service_name: str = Field(description="Service name")
    endpoint: str = Field(description="Health check endpoint")
    interval: int = Field(default=30, description="Check interval in seconds")
    timeout: int = Field(default=5, description="Timeout in seconds")


class MetricsAggregator:
    """Aggregates metrics before writing to InfluxDB"""
    
    def __init__(self, flush_interval: int = 10):
        self.metrics_buffer = defaultdict(list)
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        
    def add_metric(self, service: str, metric: str, value: float, tags: dict = None):
        """Add metric to buffer"""
        key = f"{service}.{metric}"
        self.metrics_buffer[key].append({
            "value": value,
            "tags": tags or {},
            "timestamp": time.time()
        })
        
    def should_flush(self) -> bool:
        """Check if buffer should be flushed"""
        return time.time() - self.last_flush >= self.flush_interval
        
    def get_aggregated_metrics(self) -> List[Dict]:
        """Get aggregated metrics and clear buffer"""
        aggregated = []
        
        for key, values in self.metrics_buffer.items():
            service, metric = key.rsplit(".", 1)
            
            # Calculate aggregations
            metric_values = [v["value"] for v in values]
            if metric_values:
                aggregated.append({
                    "service": service,
                    "metric": metric,
                    "count": len(metric_values),
                    "sum": sum(metric_values),
                    "mean": statistics.mean(metric_values),
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "p50": statistics.median(metric_values),
                    "p95": statistics.quantiles(metric_values, n=20)[18] if len(metric_values) > 1 else metric_values[0],
                    "p99": statistics.quantiles(metric_values, n=100)[98] if len(metric_values) > 1 else metric_values[0],
                    "timestamp": time.time()
                })
                
        self.metrics_buffer.clear()
        self.last_flush = time.time()
        return aggregated


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_rules = {}
        self.alert_history = deque(maxlen=1000)
        
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
        
    def check_metric(self, metric_name: str, value: float, tags: dict = None):
        """Check metric against alert rules"""
        for rule_name, rule in self.alert_rules.items():
            if rule.metric != metric_name:
                continue
                
            triggered = False
            if rule.condition == "gt" and value > rule.threshold:
                triggered = True
            elif rule.condition == "lt" and value < rule.threshold:
                triggered = True
            elif rule.condition == "eq" and value == rule.threshold:
                triggered = True
                
            if triggered:
                self._trigger_alert(rule, metric_name, value, tags)
            else:
                self._clear_alert(rule_name)
                
    def _trigger_alert(self, rule: AlertRule, metric: str, value: float, tags: dict):
        """Trigger an alert"""
        alert_key = f"{rule.name}_{metric}"
        
        if alert_key not in self.active_alerts:
            alert = {
                "rule": rule.name,
                "metric": metric,
                "value": value,
                "threshold": rule.threshold,
                "severity": rule.severity.value,
                "triggered_at": datetime.utcnow().isoformat(),
                "tags": tags or {}
            }
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            for channel in rule.notification_channels:
                self._send_notification(channel, alert)
                
            logger.warning(f"Alert triggered: {rule.name} - {metric}={value} (threshold={rule.threshold})")
            
    def _clear_alert(self, rule_name: str):
        """Clear an alert if it exists"""
        keys_to_remove = [k for k in self.active_alerts.keys() if k.startswith(rule_name)]
        for key in keys_to_remove:
            del self.active_alerts[key]
            logger.info(f"Alert cleared: {key}")
            
    def _send_notification(self, channel: str, alert: dict):
        """Send alert notification"""
        # In production, integrate with Slack, PagerDuty, email, etc.
        logger.info(f"Sending alert to {channel}: {alert}")
        
    def get_active_alerts(self) -> List[dict]:
        """Get all active alerts"""
        return list(self.active_alerts.values())


class ServiceMonitor:
    """Monitors service health"""
    
    def __init__(self):
        self.service_configs = {}
        self.service_status = {}
        self.check_tasks = {}
        
    def register_service(self, config: ServiceHealthCheck):
        """Register service for monitoring"""
        self.service_configs[config.service_name] = config
        self.service_status[config.service_name] = ServiceStatus.UNKNOWN
        logger.info(f"Registered service for monitoring: {config.service_name}")
        
    async def start_monitoring(self):
        """Start monitoring all registered services"""
        for service_name, config in self.service_configs.items():
            if service_name not in self.check_tasks:
                task = asyncio.create_task(self._monitor_service(service_name, config))
                self.check_tasks[service_name] = task
                
    async def _monitor_service(self, service_name: str, config: ServiceHealthCheck):
        """Monitor a single service"""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        config.endpoint,
                        timeout=aiohttp.ClientTimeout(total=config.timeout)
                    ) as response:
                        if response.status == 200:
                            self.service_status[service_name] = ServiceStatus.HEALTHY
                        else:
                            self.service_status[service_name] = ServiceStatus.DEGRADED
                            
            except asyncio.TimeoutError:
                self.service_status[service_name] = ServiceStatus.UNHEALTHY
                logger.error(f"Service {service_name} health check timeout")
                
            except Exception as e:
                self.service_status[service_name] = ServiceStatus.UNHEALTHY
                logger.error(f"Service {service_name} health check failed: {e}")
                
            await asyncio.sleep(config.interval)
            
    def get_service_status(self) -> Dict[str, str]:
        """Get status of all monitored services"""
        return {
            service: status.value
            for service, status in self.service_status.items()
        }


class MonitoringService:
    """Enterprise Monitoring Service - Main class"""
    
    def __init__(self):
        self.app = FastAPI(title="Enterprise Monitoring Service", version="2.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("monitoring_service")
        self.cache = get_cache()
        
        # Initialize InfluxDB client
        self.influx_client = None
        self.influx_write_api = None
        self.influx_query_api = None
        self._init_influxdb()
        
        # Initialize monitoring components
        self.aggregator = MetricsAggregator()
        self.alert_manager = AlertManager()
        self.service_monitor = ServiceMonitor()
        
        # System metrics collection
        self.system_metrics_task = None
        
        # Grafana dashboards
        self.dashboards = {}
        
        logger.info("✅ Monitoring Service initialized")
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
        self._setup_default_alerts()
        self._register_services()
        
    def _init_influxdb(self):
        """Initialize InfluxDB connection"""
        try:
            influx_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
            influx_token = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
            influx_org = os.getenv("INFLUXDB_ORG", "agent-lightning")
            influx_bucket = os.getenv("INFLUXDB_BUCKET", "metrics")
            
            self.influx_client = InfluxDBClient(
                url=influx_url,
                token=influx_token,
                org=influx_org
            )
            
            self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            self.influx_query_api = self.influx_client.query_api()
            self.influx_bucket = influx_bucket
            self.influx_org = influx_org
            
            logger.info(f"Connected to InfluxDB at {influx_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            logger.warning("Running without InfluxDB - metrics will not be persisted")
            
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            health_status = self.dal.health_check()
            influx_healthy = self.influx_client is not None
            
            return {
                "service": "monitoring",
                "status": "healthy" if health_status['database'] and influx_healthy else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "influxdb": influx_healthy,
                "active_alerts": len(self.alert_manager.active_alerts),
                "monitored_services": len(self.service_monitor.service_configs),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        @self.app.get("/")
        async def root():
            """Monitoring dashboard"""
            return HTMLResponse(content="""
            <html>
                <head>
                    <title>Enterprise Monitoring</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .metric { display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                        .healthy { color: green; }
                        .unhealthy { color: red; }
                        .warning { color: orange; }
                    </style>
                </head>
                <body>
                    <h1>Enterprise Monitoring Service</h1>
                    <div id="services"></div>
                    <div id="metrics"></div>
                    <div id="alerts"></div>
                    <script>
                        async function updateDashboard() {
                            // Update services status
                            const servicesResp = await fetch('/services/status');
                            const services = await servicesResp.json();
                            document.getElementById('services').innerHTML = '<h2>Services</h2>' + 
                                Object.entries(services).map(([name, status]) => 
                                    `<div class="metric ${status}">${name}: ${status}</div>`
                                ).join('');
                            
                            // Update alerts
                            const alertsResp = await fetch('/alerts/active');
                            const alerts = await alertsResp.json();
                            document.getElementById('alerts').innerHTML = '<h2>Active Alerts</h2>' + 
                                (alerts.length > 0 ? 
                                    alerts.map(a => `<div class="metric warning">${a.rule}: ${a.metric}=${a.value}</div>`).join('') :
                                    '<div>No active alerts</div>');
                        }
                        
                        updateDashboard();
                        setInterval(updateDashboard, 5000);
                    </script>
                </body>
            </html>
            """)
            
        @self.app.post("/metrics")
        async def submit_metric(metric: MetricSubmit, background_tasks: BackgroundTasks):
            """Submit a metric"""
            try:
                # Add to aggregator
                self.aggregator.add_metric(
                    metric.service,
                    metric.metric_name,
                    metric.value,
                    metric.tags
                )
                
                # Check alerts
                self.alert_manager.check_metric(
                    f"{metric.service}.{metric.metric_name}",
                    metric.value,
                    metric.tags
                )
                
                # Write to InfluxDB if available
                if self.influx_write_api:
                    point = Point(metric.metric_name) \
                        .tag("service", metric.service) \
                        .field("value", metric.value) \
                        .time(datetime.utcnow(), WritePrecision.NS)
                    
                    if metric.tags:
                        for key, value in metric.tags.items():
                            point = point.tag(key, value)
                            
                    self.influx_write_api.write(
                        bucket=self.influx_bucket,
                        org=self.influx_org,
                        record=point
                    )
                    
                return {"status": "accepted", "metric": metric.metric_name}
                
            except Exception as e:
                logger.error(f"Failed to submit metric: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/metrics/query")
        async def query_metrics(
            metric: str,
            service: Optional[str] = None,
            start: str = "-1h",
            stop: str = "now()"
        ):
            """Query metrics from InfluxDB"""
            try:
                if not self.influx_query_api:
                    raise HTTPException(status_code=503, detail="InfluxDB not available")
                    
                # Build Flux query
                query = f'''
                from(bucket: "{self.influx_bucket}")
                    |> range(start: {start}, stop: {stop})
                    |> filter(fn: (r) => r["_measurement"] == "{metric}")
                '''
                
                if service:
                    query += f'|> filter(fn: (r) => r["service"] == "{service}")'
                    
                # Execute query
                result = self.influx_query_api.query(org=self.influx_org, query=query)
                
                # Format results
                data = []
                for table in result:
                    for record in table.records:
                        data.append({
                            "time": record.get_time().isoformat(),
                            "value": record.get_value(),
                            "service": record.values.get("service"),
                            "measurement": record.get_measurement()
                        })
                        
                return {"metric": metric, "data": data, "count": len(data)}
                
            except Exception as e:
                logger.error(f"Failed to query metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/alerts/rules")
        async def create_alert_rule(rule: AlertRule):
            """Create alert rule"""
            try:
                self.alert_manager.add_rule(rule)
                
                # Store in cache
                self.cache.set(f"alert_rule:{rule.name}", rule.dict(), ttl=None)
                
                return {"status": "created", "rule": rule.name}
                
            except Exception as e:
                logger.error(f"Failed to create alert rule: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/alerts/active")
        async def get_active_alerts():
            """Get active alerts"""
            return self.alert_manager.get_active_alerts()
            
        @self.app.get("/alerts/history")
        async def get_alert_history(limit: int = 100):
            """Get alert history"""
            return list(self.alert_manager.alert_history)[:limit]
            
        @self.app.post("/services/register")
        async def register_service(config: ServiceHealthCheck):
            """Register service for monitoring"""
            try:
                self.service_monitor.register_service(config)
                
                # Start monitoring if not already running
                await self.service_monitor.start_monitoring()
                
                return {"status": "registered", "service": config.service_name}
                
            except Exception as e:
                logger.error(f"Failed to register service: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/services/status")
        async def get_services_status():
            """Get status of all monitored services"""
            return self.service_monitor.get_service_status()
            
        @self.app.post("/dashboards")
        async def create_dashboard(dashboard: DashboardCreate):
            """Create Grafana dashboard"""
            try:
                dashboard_id = f"dashboard_{len(self.dashboards) + 1}"
                
                # Store dashboard configuration
                self.dashboards[dashboard_id] = dashboard.dict()
                self.cache.set(f"dashboard:{dashboard_id}", dashboard.dict(), ttl=None)
                
                # In production, would integrate with Grafana API
                logger.info(f"Created dashboard: {dashboard.name}")
                
                return {"dashboard_id": dashboard_id, "status": "created"}
                
            except Exception as e:
                logger.error(f"Failed to create dashboard: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/dashboards")
        async def list_dashboards():
            """List all dashboards"""
            return {
                "dashboards": list(self.dashboards.values()),
                "count": len(self.dashboards)
            }
            
        @self.app.get("/system/metrics")
        async def get_system_metrics():
            """Get system metrics"""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                return {
                    "cpu": {
                        "percent": cpu_percent,
                        "count": psutil.cpu_count()
                    },
                    "memory": {
                        "total": memory.total,
                        "used": memory.used,
                        "percent": memory.percent
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "percent": disk.percent
                    },
                    "network": {
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get system metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
    def _setup_event_handlers(self):
        """Setup event handlers for cross-service events"""
        
        def on_task_completed(event):
            """Record task completion metrics"""
            task_id = event.data.get('task_id')
            duration = event.data.get('duration_ms', 0)
            
            # Submit metric
            asyncio.create_task(self._submit_internal_metric(
                "agent_designer",
                "task_completed",
                1,
                {"task_id": task_id}
            ))
            
            if duration > 0:
                asyncio.create_task(self._submit_internal_metric(
                    "agent_designer",
                    "task_duration_ms",
                    duration,
                    {"task_id": task_id}
                ))
                
        def on_workflow_completed(event):
            """Record workflow completion metrics"""
            workflow_id = event.data.get('workflow_id')
            
            asyncio.create_task(self._submit_internal_metric(
                "workflow_engine",
                "workflow_completed",
                1,
                {"workflow_id": workflow_id}
            ))
            
        # Register handlers
        self.dal.event_bus.on(EventChannel.TASK_COMPLETED, on_task_completed)
        self.dal.event_bus.on(EventChannel.WORKFLOW_COMPLETED, on_workflow_completed)
        
        logger.info("Event handlers registered for monitoring")
        
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu",
                metric="system.cpu_percent",
                condition="gt",
                threshold=80,
                duration=300,
                severity=AlertSeverity.WARNING,
                notification_channels=["slack", "email"]
            ),
            AlertRule(
                name="high_memory",
                metric="system.memory_percent",
                condition="gt",
                threshold=90,
                duration=300,
                severity=AlertSeverity.WARNING,
                notification_channels=["slack"]
            ),
            AlertRule(
                name="service_unhealthy",
                metric="service.health",
                condition="eq",
                threshold=0,
                duration=60,
                severity=AlertSeverity.ERROR,
                notification_channels=["pagerduty", "slack"]
            ),
            AlertRule(
                name="high_error_rate",
                metric="errors.rate",
                condition="gt",
                threshold=10,
                duration=60,
                severity=AlertSeverity.ERROR,
                notification_channels=["slack", "email"]
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
            
        logger.info(f"Configured {len(default_rules)} default alert rules")
        
    def _register_services(self):
        """Register core services for monitoring"""
        services = [
            ("auth_service", "http://localhost:8001/health"),
            ("agent_designer", "http://localhost:8002/health"),
            ("workflow_engine", "http://localhost:8003/health"),
            ("integration_hub", "http://localhost:8004/health"),
            ("ai_model", "http://localhost:8005/health"),
            ("visual_builder", "http://localhost:8006/health"),
            ("websocket", "http://localhost:8009/health")
        ]
        
        for service_name, endpoint in services:
            config = ServiceHealthCheck(
                service_name=service_name,
                endpoint=endpoint,
                interval=30,
                timeout=5
            )
            self.service_monitor.register_service(config)
            
        logger.info(f"Registered {len(services)} services for monitoring")
        
    async def _submit_internal_metric(self, service: str, metric: str, value: float, tags: dict = None):
        """Submit internal metric"""
        try:
            self.aggregator.add_metric(service, metric, value, tags)
            self.alert_manager.check_metric(f"{service}.{metric}", value, tags)
        except Exception as e:
            logger.error(f"Failed to submit internal metric: {e}")
            
    async def _collect_system_metrics(self):
        """Continuously collect system metrics"""
        while True:
            try:
                # Collect CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                await self._submit_internal_metric("system", "cpu_percent", cpu_percent)
                
                # Collect memory metrics
                memory = psutil.virtual_memory()
                await self._submit_internal_metric("system", "memory_percent", memory.percent)
                await self._submit_internal_metric("system", "memory_used_bytes", memory.used)
                
                # Collect disk metrics
                disk = psutil.disk_usage('/')
                await self._submit_internal_metric("system", "disk_percent", disk.percent)
                await self._submit_internal_metric("system", "disk_used_bytes", disk.used)
                
                # Collect network metrics
                network = psutil.net_io_counters()
                await self._submit_internal_metric("system", "network_bytes_sent", network.bytes_sent)
                await self._submit_internal_metric("system", "network_bytes_recv", network.bytes_recv)
                
                # Flush aggregated metrics to InfluxDB
                if self.aggregator.should_flush():
                    metrics = self.aggregator.get_aggregated_metrics()
                    for metric in metrics:
                        if self.influx_write_api:
                            point = Point("aggregated_metrics") \
                                .tag("service", metric["service"]) \
                                .tag("metric", metric["metric"]) \
                                .field("count", metric["count"]) \
                                .field("sum", metric["sum"]) \
                                .field("mean", metric["mean"]) \
                                .field("min", metric["min"]) \
                                .field("max", metric["max"]) \
                                .field("p50", metric["p50"]) \
                                .field("p95", metric["p95"]) \
                                .field("p99", metric["p99"]) \
                                .time(datetime.utcnow(), WritePrecision.NS)
                                
                            self.influx_write_api.write(
                                bucket=self.influx_bucket,
                                org=self.influx_org,
                                record=point
                            )
                            
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                
            await asyncio.sleep(10)
            
    async def startup(self):
        """Startup tasks"""
        logger.info("Enterprise Monitoring Service starting up...")
        
        # Verify connections
        health = self.dal.health_check()
        if not health['database']:
            logger.warning("Database not available")
        if not health['cache']:
            logger.warning("Cache not available")
            
        # Start service monitoring
        await self.service_monitor.start_monitoring()
        
        # Start system metrics collection
        self.system_metrics_task = asyncio.create_task(self._collect_system_metrics())
        
        logger.info("Monitoring Service ready")
        
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Monitoring Service shutting down...")
        
        # Stop background tasks
        if self.system_metrics_task:
            self.system_metrics_task.cancel()
            
        # Stop service monitoring
        for task in self.service_monitor.check_tasks.values():
            task.cancel()
            
        # Close InfluxDB connection
        if self.influx_client:
            self.influx_client.close()
            
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = MonitoringService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("MONITORING_PORT", 8007))
    logger.info(f"Starting Enterprise Monitoring Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()