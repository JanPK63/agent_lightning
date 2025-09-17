#!/usr/bin/env python3
"""
Performance Metrics Service
Collects, aggregates, and exposes real-time performance metrics
Provides data for dashboards and monitoring systems
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import time
import threading
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import aiohttp
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache
from shared.events import EventChannel, EventBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricData(BaseModel):
    """Individual metric data point"""
    metric_name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Optional[Dict[str, Any]] = None


class AgentMetrics(BaseModel):
    """Agent performance metrics"""
    agent_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_duration_seconds: float
    success_rate: float
    current_load: int
    last_active: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0


class SystemMetrics(BaseModel):
    """System-wide metrics"""
    total_agents: int
    active_agents: int
    total_tasks: int
    tasks_in_progress: int
    tasks_completed: int
    tasks_failed: int
    average_task_duration: float
    system_throughput: float  # tasks per minute
    error_rate: float
    cache_hit_rate: float
    database_connections: int
    api_response_time: float


class PerformanceMetricsService:
    """Service for collecting and exposing performance metrics"""
    
    def __init__(self):
        self.app = FastAPI(title="Performance Metrics Service", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("performance_metrics")
        self.cache = get_cache()
        self.event_bus = EventBus("performance_metrics")
        
        # Database connection
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agent_lightning"),
            "user": os.getenv("POSTGRES_USER", "agent_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "agent_password")
        }
        
        # Service URLs
        self.services = {
            "task_history": "http://localhost:8027",
            "rl_orchestrator": "http://localhost:8025",
            "monitoring": "http://localhost:8007",
            "auth_service": "http://localhost:8001"
        }
        
        # Metrics storage
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.aggregated_metrics = {}
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Create tables
        self._create_tables()
        
        # Start background workers
        self._start_workers()
        
        logger.info("✅ Performance Metrics Service initialized")
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_listeners()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        # Task metrics
        self.task_counter = Counter(
            'tasks_total', 
            'Total number of tasks',
            ['status', 'agent_id'],
            registry=self.registry
        )
        
        self.task_duration = Histogram(
            'task_duration_seconds',
            'Task execution duration in seconds',
            ['agent_id'],
            registry=self.registry
        )
        
        self.active_tasks = Gauge(
            'active_tasks',
            'Number of active tasks',
            ['agent_id'],
            registry=self.registry
        )
        
        # Agent metrics
        self.agent_success_rate = Gauge(
            'agent_success_rate',
            'Agent success rate percentage',
            ['agent_id'],
            registry=self.registry
        )
        
        self.agent_load = Gauge(
            'agent_current_load',
            'Current load on agent',
            ['agent_id'],
            registry=self.registry
        )
        
        # System metrics
        self.system_throughput = Gauge(
            'system_throughput',
            'System throughput (tasks per minute)',
            registry=self.registry
        )
        
        self.error_rate = Gauge(
            'system_error_rate',
            'System-wide error rate',
            registry=self.registry
        )
        
        self.api_latency = Histogram(
            'api_latency_seconds',
            'API endpoint latency',
            ['endpoint', 'method'],
            registry=self.registry
        )
    
    def _create_tables(self):
        """Create metrics storage tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Performance metrics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_type VARCHAR(20) NOT NULL,
                    value FLOAT NOT NULL,
                    labels JSONB,
                    metadata JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name);
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_metrics_labels ON performance_metrics USING GIN(labels);
            """)
            
            # Agent metrics summary table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_metrics_summary (
                    agent_id VARCHAR(100) PRIMARY KEY,
                    total_tasks INTEGER DEFAULT 0,
                    completed_tasks INTEGER DEFAULT 0,
                    failed_tasks INTEGER DEFAULT 0,
                    total_duration_seconds FLOAT DEFAULT 0,
                    last_active TIMESTAMP,
                    cpu_usage FLOAT DEFAULT 0,
                    memory_usage FLOAT DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Performance metrics tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
    
    def _start_workers(self):
        """Start background workers for metrics collection"""
        # Start metrics collector thread
        collector_thread = threading.Thread(target=self._metrics_collector_loop, daemon=True)
        collector_thread.start()
        
        # Start aggregator thread
        aggregator_thread = threading.Thread(target=self._metrics_aggregator_loop, daemon=True)
        aggregator_thread.start()
        
        logger.info("Background workers started")
    
    def _metrics_collector_loop(self):
        """Continuously collect metrics from various sources"""
        while True:
            try:
                # Collect from database
                self._collect_database_metrics()
                
                # Collect from services
                asyncio.run(self._collect_service_metrics())
                
                # Collect system metrics
                self._collect_system_metrics()
                
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                time.sleep(5)
    
    def _metrics_aggregator_loop(self):
        """Aggregate metrics periodically"""
        while True:
            try:
                self._aggregate_metrics()
                time.sleep(30)  # Aggregate every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics aggregator: {e}")
                time.sleep(5)
    
    def _collect_database_metrics(self):
        """Collect metrics from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get task counts by status
            cur.execute("""
                SELECT status, COUNT(*) as count
                FROM tasks
                WHERE created_at >= NOW() - INTERVAL '1 hour'
                GROUP BY status
            """)
            
            for row in cur.fetchall():
                self.metrics_buffer['task_status'].append({
                    'status': row['status'],
                    'count': row['count'],
                    'timestamp': datetime.utcnow()
                })
            
            # Get agent performance
            cur.execute("""
                SELECT 
                    agent_id,
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration
                FROM tasks
                WHERE agent_id IS NOT NULL
                GROUP BY agent_id
            """)
            
            for row in cur.fetchall():
                agent_id = row['agent_id']
                
                # Update Prometheus metrics
                self.agent_success_rate.labels(agent_id=agent_id).set(
                    (row['completed'] / row['total_tasks'] * 100) if row['total_tasks'] > 0 else 0
                )
                
                # Store in buffer
                self.metrics_buffer[f'agent_{agent_id}'].append({
                    'total_tasks': row['total_tasks'],
                    'completed': row['completed'],
                    'failed': row['failed'],
                    'avg_duration': row['avg_duration'] or 0,
                    'timestamp': datetime.utcnow()
                })
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
    
    async def _collect_service_metrics(self):
        """Collect metrics from other services"""
        try:
            async with aiohttp.ClientSession() as session:
                # Collect from task history service
                try:
                    async with session.get(
                        f"{self.services['task_history']}/analytics/summary?days=1"
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.metrics_buffer['task_history'].append({
                                'data': data,
                                'timestamp': datetime.utcnow()
                            })
                except:
                    pass
                
                # Collect from monitoring service
                try:
                    async with session.get(
                        f"{self.services['monitoring']}/metrics"
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.metrics_buffer['monitoring'].append({
                                'data': data,
                                'timestamp': datetime.utcnow()
                            })
                except:
                    pass
                
        except Exception as e:
            logger.error(f"Failed to collect service metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # Cache metrics
            cache_stats = self.cache.redis_client.info('stats')
            cache_hits = cache_stats.get('keyspace_hits', 0)
            cache_misses = cache_stats.get('keyspace_misses', 0)
            cache_hit_rate = (cache_hits / (cache_hits + cache_misses) * 100) if (cache_hits + cache_misses) > 0 else 0
            
            # Database connections
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM pg_stat_activity")
            db_connections = cur.fetchone()[0]
            conn.close()
            
            # Store system metrics
            self.metrics_buffer['system'].append({
                'cache_hit_rate': cache_hit_rate,
                'db_connections': db_connections,
                'timestamp': datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _aggregate_metrics(self):
        """Aggregate collected metrics"""
        try:
            now = datetime.utcnow()
            
            # Aggregate task metrics
            if 'task_status' in self.metrics_buffer:
                recent_tasks = [m for m in self.metrics_buffer['task_status'] 
                              if (now - m['timestamp']).seconds < 300]
                
                if recent_tasks:
                    total_tasks = sum(m['count'] for m in recent_tasks)
                    completed = sum(m['count'] for m in recent_tasks if m['status'] == 'completed')
                    failed = sum(m['count'] for m in recent_tasks if m['status'] == 'failed')
                    
                    # Calculate throughput (tasks per minute)
                    time_window = 5  # minutes
                    throughput = total_tasks / time_window
                    
                    self.system_throughput.set(throughput)
                    
                    # Calculate error rate
                    error_rate = (failed / total_tasks * 100) if total_tasks > 0 else 0
                    self.error_rate.set(error_rate)
                    
                    self.aggregated_metrics['tasks'] = {
                        'total': total_tasks,
                        'completed': completed,
                        'failed': failed,
                        'throughput': throughput,
                        'error_rate': error_rate
                    }
            
            # Aggregate agent metrics
            for key in list(self.metrics_buffer.keys()):
                if key.startswith('agent_'):
                    agent_id = key.replace('agent_', '')
                    recent = [m for m in self.metrics_buffer[key] 
                            if (now - m['timestamp']).seconds < 300]
                    
                    if recent:
                        latest = recent[-1]
                        self.aggregated_metrics[key] = {
                            'agent_id': agent_id,
                            'total_tasks': latest['total_tasks'],
                            'completed': latest['completed'],
                            'failed': latest['failed'],
                            'avg_duration': latest['avg_duration'],
                            'success_rate': (latest['completed'] / latest['total_tasks'] * 100) 
                                          if latest['total_tasks'] > 0 else 0
                        }
            
            # Store aggregated metrics in cache for quick access
            self.cache.set('metrics:aggregated', self.aggregated_metrics, ttl=60)
            
        except Exception as e:
            logger.error(f"Failed to aggregate metrics: {e}")
    
    async def record_metric(self, metric: MetricData):
        """Record a new metric"""
        try:
            # Store in database
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO performance_metrics 
                (metric_name, metric_type, value, labels, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                metric.metric_name,
                metric.metric_type.value,
                metric.value,
                json.dumps(metric.labels),
                json.dumps(metric.metadata) if metric.metadata else None
            ))
            
            conn.commit()
            conn.close()
            
            # Update Prometheus metrics
            if metric.metric_type == MetricType.COUNTER:
                # Find or create counter
                pass  # Dynamic metric creation would go here
            
            # Store in buffer
            self.metrics_buffer[metric.metric_name].append({
                'value': metric.value,
                'labels': metric.labels,
                'timestamp': datetime.fromisoformat(metric.timestamp)
            })
            
            # Emit event
            self.event_bus.emit(EventChannel.SYSTEM_METRICS, {
                'metric': metric.metric_name,
                'value': metric.value,
                'labels': metric.labels
            })
            
            return {"status": "recorded", "metric": metric.metric_name}
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """Get metrics for a specific agent"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get agent task stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration,
                    MAX(created_at) as last_active
                FROM tasks
                WHERE agent_id = %s
            """, (agent_id,))
            
            stats = cur.fetchone()
            conn.close()
            
            if not stats or stats['total_tasks'] == 0:
                return AgentMetrics(
                    agent_id=agent_id,
                    total_tasks=0,
                    completed_tasks=0,
                    failed_tasks=0,
                    average_duration_seconds=0,
                    success_rate=0,
                    current_load=0,
                    last_active=datetime.utcnow().isoformat()
                )
            
            success_rate = (stats['completed'] / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0
            
            return AgentMetrics(
                agent_id=agent_id,
                total_tasks=stats['total_tasks'],
                completed_tasks=stats['completed'] or 0,
                failed_tasks=stats['failed'] or 0,
                average_duration_seconds=stats['avg_duration'] or 0,
                success_rate=success_rate,
                current_load=stats['in_progress'] or 0,
                last_active=stats['last_active'].isoformat() if stats['last_active'] else datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to get agent metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get overall task stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration
                FROM tasks
                WHERE created_at >= NOW() - INTERVAL '1 hour'
            """)
            
            task_stats = cur.fetchone()
            
            # Get agent count
            cur.execute("""
                SELECT COUNT(DISTINCT agent_id) as total_agents,
                       COUNT(DISTINCT CASE WHEN created_at >= NOW() - INTERVAL '5 minutes' 
                             THEN agent_id END) as active_agents
                FROM tasks
                WHERE agent_id IS NOT NULL
            """)
            
            agent_stats = cur.fetchone()
            
            conn.close()
            
            # Get cached system metrics
            cached_metrics = self.aggregated_metrics.get('tasks', {})
            
            # Calculate metrics
            throughput = cached_metrics.get('throughput', 0)
            error_rate = cached_metrics.get('error_rate', 0)
            
            # Get cache hit rate
            cache_stats = self.cache.redis_client.info('stats')
            cache_hits = cache_stats.get('keyspace_hits', 0)
            cache_misses = cache_stats.get('keyspace_misses', 0)
            cache_hit_rate = (cache_hits / (cache_hits + cache_misses) * 100) if (cache_hits + cache_misses) > 0 else 0
            
            return SystemMetrics(
                total_agents=agent_stats['total_agents'] or 0,
                active_agents=agent_stats['active_agents'] or 0,
                total_tasks=task_stats['total_tasks'] or 0,
                tasks_in_progress=task_stats['in_progress'] or 0,
                tasks_completed=task_stats['completed'] or 0,
                tasks_failed=task_stats['failed'] or 0,
                average_task_duration=task_stats['avg_duration'] or 0,
                system_throughput=throughput,
                error_rate=error_rate,
                cache_hit_rate=cache_hit_rate,
                database_connections=20,  # Placeholder
                api_response_time=0.1  # Placeholder
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_event_listeners(self):
        """Setup event listeners for metric collection"""
        
        def on_task_event(event):
            """Handle task events for metrics"""
            try:
                if event.channel == EventChannel.TASK_COMPLETED.value:
                    self.task_counter.labels(
                        status='completed',
                        agent_id=event.data.get('agent_id', 'unknown')
                    ).inc()
                    
                elif event.channel == EventChannel.TASK_FAILED.value:
                    self.task_counter.labels(
                        status='failed',
                        agent_id=event.data.get('agent_id', 'unknown')
                    ).inc()
                    
            except Exception as e:
                logger.error(f"Failed to handle task event: {e}")
        
        # Subscribe to events
        self.event_bus.on(EventChannel.TASK_COMPLETED, on_task_event)
        self.event_bus.on(EventChannel.TASK_FAILED, on_task_event)
        self.event_bus.on(EventChannel.TASK_STARTED, on_task_event)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "performance_metrics",
                "status": "healthy",
                "metrics_collected": len(self.metrics_buffer),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/record")
        async def record_metric(metric: MetricData):
            """Record a new metric"""
            return await self.record_metric(metric)
        
        @self.app.get("/metrics/agent/{agent_id}")
        async def get_agent_metrics(agent_id: str):
            """Get metrics for a specific agent"""
            return await self.get_agent_metrics(agent_id)
        
        @self.app.get("/metrics/system")
        async def get_system_metrics():
            """Get system-wide metrics"""
            return await self.get_system_metrics()
        
        @self.app.get("/metrics/prometheus")
        async def prometheus_metrics():
            """Expose metrics in Prometheus format"""
            return generate_latest(self.registry)
        
        @self.app.get("/metrics/aggregated")
        async def get_aggregated_metrics():
            """Get aggregated metrics"""
            return self.aggregated_metrics
        
        @self.app.get("/metrics/realtime")
        async def get_realtime_metrics(
            metric_name: Optional[str] = Query(None),
            limit: int = Query(100, le=1000)
        ):
            """Get realtime metrics from buffer"""
            if metric_name:
                if metric_name in self.metrics_buffer:
                    data = list(self.metrics_buffer[metric_name])[-limit:]
                    return {
                        "metric": metric_name,
                        "data": [
                            {**m, 'timestamp': m['timestamp'].isoformat()} 
                            for m in data
                        ]
                    }
                else:
                    return {"metric": metric_name, "data": []}
            else:
                # Return all metrics
                result = {}
                for name, buffer in self.metrics_buffer.items():
                    data = list(buffer)[-limit:]
                    result[name] = [
                        {**m, 'timestamp': m['timestamp'].isoformat()} 
                        for m in data
                    ]
                return result
        
        @self.app.get("/metrics/grafana")
        async def grafana_metrics():
            """Grafana-compatible metrics endpoint"""
            # Format metrics for Grafana
            metrics = []
            
            # Add system metrics
            system = await self.get_system_metrics()
            metrics.append({
                "target": "system.throughput",
                "datapoints": [[system.system_throughput, int(time.time() * 1000)]]
            })
            metrics.append({
                "target": "system.error_rate",
                "datapoints": [[system.error_rate, int(time.time() * 1000)]]
            })
            metrics.append({
                "target": "system.active_agents",
                "datapoints": [[system.active_agents, int(time.time() * 1000)]]
            })
            
            return metrics
        
        @self.app.get("/metrics/export")
        async def export_metrics(
            start_time: Optional[str] = Query(None),
            end_time: Optional[str] = Query(None),
            format: str = Query("json", regex="^(json|csv|prometheus)$")
        ):
            """Export metrics for analysis"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                query = "SELECT * FROM performance_metrics"
                params = []
                
                if start_time:
                    query += " WHERE timestamp >= %s"
                    params.append(start_time)
                
                if end_time:
                    if start_time:
                        query += " AND timestamp <= %s"
                    else:
                        query += " WHERE timestamp <= %s"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC LIMIT 10000"
                
                cur.execute(query, params)
                metrics = cur.fetchall()
                conn.close()
                
                # Format timestamps
                for metric in metrics:
                    metric['timestamp'] = metric['timestamp'].isoformat()
                
                if format == "json":
                    return {"metrics": metrics, "count": len(metrics)}
                elif format == "csv":
                    # Convert to CSV format
                    import csv
                    import io
                    output = io.StringIO()
                    if metrics:
                        writer = csv.DictWriter(output, fieldnames=metrics[0].keys())
                        writer.writeheader()
                        writer.writerows(metrics)
                    return output.getvalue()
                else:
                    # Prometheus format
                    lines = []
                    for metric in metrics:
                        labels = json.loads(metric['labels']) if metric['labels'] else {}
                        label_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])
                        lines.append(f"{metric['metric_name']}{{{label_str}}} {metric['value']}")
                    return "\n".join(lines)
                
            except Exception as e:
                logger.error(f"Failed to export metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Performance Metrics Service starting up...")
        self.event_bus.start()
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Performance Metrics Service shutting down...")
        self.event_bus.stop()
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = PerformanceMetricsService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("METRICS_PORT", 8031))
    logger.info(f"Starting Performance Metrics Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()