#!/usr/bin/env python3
"""
Centralized Prometheus Metrics for Agent Lightning
Provides unified metrics collection across all services
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
import logging

logger = logging.getLogger(__name__)

class AgentLightningMetrics:
    """
    Centralized metrics collection for Agent Lightning services
    Provides standardized metrics across all components
    """

    def __init__(self, service_name: str = "agent_lightning", registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics for a service

        Args:
            service_name: Name of the service (e.g., 'api', 'agent_executor', 'memory')
            registry: Custom registry or use default
        """
        self.service_name = service_name
        self.registry = registry or REGISTRY

        # Service-level metrics
        self._init_service_metrics()

        # HTTP/API metrics
        self._init_http_metrics()

        # Agent metrics
        self._init_agent_metrics()

        # Database metrics
        self._init_database_metrics()

        # Memory metrics
        self._init_memory_metrics()

        # Workflow metrics
        self._init_workflow_metrics()

        # System metrics
        self._init_system_metrics()

        # Custom business metrics
        self._init_business_metrics()

        logger.info(f"✅ Initialized Prometheus metrics for service: {service_name}")

    def _init_service_metrics(self):
        """Initialize basic service-level metrics"""
        self.service_uptime = Gauge(
            f'{self.service_name}_uptime_seconds',
            'Service uptime in seconds',
            registry=self.registry
        )

        self.service_requests_total = Counter(
            f'{self.service_name}_requests_total',
            'Total number of requests processed',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.service_errors_total = Counter(
            f'{self.service_name}_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )

        self.service_active_connections = Gauge(
            f'{self.service_name}_active_connections',
            'Number of active connections',
            registry=self.registry
        )

    def _init_http_metrics(self):
        """Initialize HTTP/API specific metrics"""
        self.http_requests_total = Counter(
            f'{self.service_name}_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )

        self.http_request_duration_seconds = Histogram(
            f'{self.service_name}_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )

        self.http_response_size_bytes = Summary(
            f'{self.service_name}_http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint'],
            registry=self.registry
        )

        self.http_active_requests = Gauge(
            f'{self.service_name}_http_active_requests',
            'Number of active HTTP requests',
            registry=self.registry
        )

    def _init_agent_metrics(self):
        """Initialize agent-specific metrics"""
        self.agent_tasks_total = Counter(
            f'{self.service_name}_agent_tasks_total',
            'Total agent tasks processed',
            ['agent_id', 'agent_type', 'status'],
            registry=self.registry
        )

        self.agent_task_duration_seconds = Histogram(
            f'{self.service_name}_agent_task_duration_seconds',
            'Agent task duration in seconds',
            ['agent_id', 'agent_type'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )

        self.agent_confidence_score = Histogram(
            f'{self.service_name}_agent_confidence_score',
            'Agent confidence scores',
            ['agent_id', 'agent_type'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )

        self.agent_memory_usage_bytes = Gauge(
            f'{self.service_name}_agent_memory_usage_bytes',
            'Agent memory usage in bytes',
            ['agent_id'],
            registry=self.registry
        )

        self.agent_active_tasks = Gauge(
            f'{self.service_name}_agent_active_tasks',
            'Number of active tasks per agent',
            ['agent_id'],
            registry=self.registry
        )

    def _init_database_metrics(self):
        """Initialize database-specific metrics"""
        self.db_connections_total = Gauge(
            f'{self.service_name}_db_connections_total',
            'Total database connections',
            ['pool_name'],
            registry=self.registry
        )

        self.db_connections_active = Gauge(
            f'{self.service_name}_db_connections_active',
            'Active database connections',
            ['pool_name'],
            registry=self.registry
        )

        self.db_query_duration_seconds = Histogram(
            f'{self.service_name}_db_query_duration_seconds',
            'Database query duration in seconds',
            ['query_type', 'table'],
            buckets=[0.001, 0.01, 0.1, 1.0, 5.0],
            registry=self.registry
        )

        self.db_connection_errors_total = Counter(
            f'{self.service_name}_db_connection_errors_total',
            'Database connection errors',
            ['error_type'],
            registry=self.registry
        )

        self.db_pool_size = Gauge(
            f'{self.service_name}_db_pool_size',
            'Database connection pool size',
            ['pool_name'],
            registry=self.registry
        )

    def _init_memory_metrics(self):
        """Initialize memory management metrics"""
        self.memory_usage_bytes = Gauge(
            f'{self.service_name}_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )

        self.memory_operations_total = Counter(
            f'{self.service_name}_memory_operations_total',
            'Memory operations (store/retrieve)',
            ['operation_type', 'memory_type'],
            registry=self.registry
        )

        self.memory_items_total = Gauge(
            f'{self.service_name}_memory_items_total',
            'Total items in memory',
            ['memory_type'],
            registry=self.registry
        )

        self.memory_hit_rate = Gauge(
            f'{self.service_name}_memory_hit_rate',
            'Memory cache hit rate',
            ['cache_type'],
            registry=self.registry
        )

    def _init_workflow_metrics(self):
        """Initialize workflow orchestration metrics"""
        self.workflow_executions_total = Counter(
            f'{self.service_name}_workflow_executions_total',
            'Total workflow executions',
            ['workflow_type', 'status'],
            registry=self.registry
        )

        self.workflow_execution_duration_seconds = Histogram(
            f'{self.service_name}_workflow_execution_duration_seconds',
            'Workflow execution duration in seconds',
            ['workflow_type'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0],
            registry=self.registry
        )

        self.workflow_active_executions = Gauge(
            f'{self.service_name}_workflow_active_executions',
            'Number of active workflow executions',
            ['workflow_type'],
            registry=self.registry
        )

        self.workflow_transitions_total = Counter(
            f'{self.service_name}_workflow_transitions_total',
            'Workflow state transitions',
            ['workflow_type', 'from_state', 'to_state'],
            registry=self.registry
        )

    def _init_system_metrics(self):
        """Initialize system-level metrics"""
        self.cpu_usage_percent = Gauge(
            f'{self.service_name}_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage_percent = Gauge(
            f'{self.service_name}_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )

        self.disk_usage_bytes = Gauge(
            f'{self.service_name}_disk_usage_bytes',
            'Disk usage in bytes',
            ['mount_point'],
            registry=self.registry
        )

        self.network_io_bytes = Counter(
            f'{self.service_name}_network_io_bytes_total',
            'Network I/O in bytes',
            ['direction', 'interface'],
            registry=self.registry
        )

        self.process_count = Gauge(
            f'{self.service_name}_process_count',
            'Number of processes',
            registry=self.registry
        )

    def _init_business_metrics(self):
        """Initialize custom business logic metrics"""
        self.business_tasks_completed = Counter(
            f'{self.service_name}_business_tasks_completed_total',
            'Business tasks completed',
            ['task_type', 'priority'],
            registry=self.registry
        )

        self.business_user_sessions = Gauge(
            f'{self.service_name}_business_user_sessions',
            'Active user sessions',
            registry=self.registry
        )

        self.business_api_calls = Counter(
            f'{self.service_name}_business_api_calls_total',
            'External API calls',
            ['api_name', 'status'],
            registry=self.registry
        )

        self.business_processing_queue_size = Gauge(
            f'{self.service_name}_business_processing_queue_size',
            'Size of processing queue',
            ['queue_type'],
            registry=self.registry
        )

    # Convenience methods for common operations
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record an HTTP request"""
        self.http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

    def record_agent_task(self, agent_id: str, agent_type: str, status: str, duration: float, confidence: float = 0.0):
        """Record an agent task"""
        self.agent_tasks_total.labels(agent_id=agent_id, agent_type=agent_type, status=status).inc()
        self.agent_task_duration_seconds.labels(agent_id=agent_id, agent_type=agent_type).observe(duration)
        if confidence > 0:
            self.agent_confidence_score.labels(agent_id=agent_id, agent_type=agent_type).observe(confidence)

    def record_database_query(self, query_type: str, table: str, duration: float):
        """Record a database query"""
        self.db_query_duration_seconds.labels(query_type=query_type, table=table).observe(duration)

    def record_workflow_execution(self, workflow_type: str, status: str, duration: float):
        """Record a workflow execution"""
        self.workflow_executions_total.labels(workflow_type=workflow_type, status=status).inc()
        self.workflow_execution_duration_seconds.labels(workflow_type=workflow_type).observe(duration)

    def record_error(self, error_type: str, component: str):
        """Record an error"""
        self.service_errors_total.labels(error_type=error_type, component=component).inc()

    def update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # CPU usage
            self.cpu_usage_percent.set(psutil.cpu_percent(interval=1))

            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage_percent.set(memory.percent)

            # Process count
            self.process_count.set(len(psutil.pids()))

        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def get_metrics_output(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

    def start_system_metrics_updater(self, interval: int = 30):
        """Start background thread to update system metrics"""
        def update_loop():
            while True:
                self.update_system_metrics()
                time.sleep(interval)

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        logger.info(f"Started system metrics updater (interval: {interval}s)")


# Global metrics instances for different services
_metrics_instances: Dict[str, AgentLightningMetrics] = {}

def get_metrics(service_name: str = "default") -> AgentLightningMetrics:
    """
    Get or create metrics instance for a service

    Args:
        service_name: Name of the service

    Returns:
        AgentLightningMetrics instance
    """
    if service_name not in _metrics_instances:
        _metrics_instances[service_name] = AgentLightningMetrics(service_name)

    return _metrics_instances[service_name]

def get_all_metrics_output() -> str:
    """Get metrics output from all registered services"""
    outputs = []
    for service_name, metrics in _metrics_instances.items():
        outputs.append(f"# Metrics for service: {service_name}")
        outputs.append(metrics.get_metrics_output())

    return "\n".join(outputs)

# Convenience functions for common use cases
def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float, service: str = "api"):
    """Record HTTP request metrics"""
    metrics = get_metrics(service)
    metrics.record_http_request(method, endpoint, status_code, duration)

def record_agent_metrics(agent_id: str, agent_type: str, status: str, duration: float, confidence: float = 0.0):
    """Record agent task metrics"""
    metrics = get_metrics("agent_executor")
    metrics.record_agent_task(agent_id, agent_type, status, duration, confidence)

def record_workflow_metrics(workflow_type: str, status: str, duration: float):
    """Record workflow execution metrics"""
    metrics = get_metrics("orchestration")
    metrics.record_workflow_execution(workflow_type, status, duration)

def record_error_metrics(error_type: str, component: str, service: str = "default"):
    """Record error metrics"""
    metrics = get_metrics(service)
    metrics.record_error(error_type, component)

# Initialize default metrics instance
_default_metrics = AgentLightningMetrics("agent_lightning_default")

if __name__ == "__main__":
    # Example usage
    print("🚀 Agent Lightning Metrics Module")
    print("=" * 50)

    # Get metrics for different services
    api_metrics = get_metrics("api")
    agent_metrics = get_metrics("agent_executor")
    db_metrics = get_metrics("database")

    # Record some sample metrics
    api_metrics.record_http_request("GET", "/health", 200, 0.05)
    agent_metrics.record_agent_task("agent_001", "full_stack_developer", "completed", 2.5, 0.85)
    db_metrics.record_database_query("SELECT", "users", 0.02)

    # Print metrics output
    print("\n📊 Sample Metrics Output:")
    print("-" * 30)
    print(api_metrics.get_metrics_output()[:500] + "...")

    print("\n✅ Metrics module ready for integration!")