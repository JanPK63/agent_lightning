# Minimal observability setup for containerized dashboard
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict, deque

class MetricsAggregator:
    """Simple metrics aggregator for dashboard"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.counters = defaultdict(int)
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        timestamp = datetime.now()
        self.metrics[name].append({
            'timestamp': timestamp,
            'value': value,
            'labels': labels or {}
        })
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        """Increment a counter"""
        key = f"{name}_{hash(str(sorted((labels or {}).items())))}"
        self.counters[key] += 1
    
    def get_metric_history(self, name: str, limit: int = 100):
        """Get metric history"""
        return list(self.metrics[name])[-limit:]

class AgentLightningObservability:
    """Simplified observability for containerized environment"""
    
    def __init__(self, prometheus_port: int = 8003, enable_console_export: bool = False):
        self.prometheus_port = prometheus_port
        self.enable_console_export = enable_console_export
        self.aggregator = MetricsAggregator()
        self.logger = logging.getLogger(__name__)
    
    def record_agent_execution(self, agent_id: str, task: str, duration: float, success: bool):
        """Record agent execution metrics"""
        self.aggregator.record_metric(
            'agent_execution_duration',
            duration,
            {'agent_id': agent_id, 'success': str(success)}
        )
        
        self.aggregator.increment_counter(
            'agent_executions_total',
            {'agent_id': agent_id, 'success': str(success)}
        )
    
    def record_training_metric(self, metric_name: str, value: float, step: int):
        """Record training metrics"""
        self.aggregator.record_metric(
            f'training_{metric_name}',
            value,
            {'step': str(step)}
        )
    
    def get_metrics(self):
        """Get all metrics"""
        return {
            'metrics': dict(self.aggregator.metrics),
            'counters': dict(self.aggregator.counters)
        }