#!/usr/bin/env python3
"""
InfluxDB Metrics Storage
Time-series database integration for performance metrics
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading
import queue
import logging
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import InfluxDB client
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    print("Warning: InfluxDB client not installed. Using mock client.")
    print("Install with: pip install influxdb-client")

from performance_monitor import (
    PerformanceMonitor, MetricSample, MetricType, 
    Alert, AlertLevel, PerformanceThresholds
)


@dataclass
class InfluxDBConfig:
    """InfluxDB connection configuration"""
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "default"
    bucket: str = "performance_metrics"
    timeout: int = 10000
    verify_ssl: bool = True
    batch_size: int = 100
    flush_interval: int = 10000  # milliseconds
    retry_interval: int = 5000
    max_retries: int = 3
    max_retry_delay: int = 30000
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls(
            url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
            token=os.getenv("INFLUXDB_TOKEN", ""),
            org=os.getenv("INFLUXDB_ORG", "default"),
            bucket=os.getenv("INFLUXDB_BUCKET", "performance_metrics")
        )


class MockInfluxDBClient:
    """Mock InfluxDB client for when the real client is not available"""
    
    def __init__(self, config: InfluxDBConfig):
        self.config = config
        self.data = []
        self.is_connected = True
        
    def write_api(self, write_options=None):
        """Return mock write API"""
        return self
    
    def write(self, bucket, org, record):
        """Mock write method"""
        self.data.append({
            "bucket": bucket,
            "org": org,
            "record": str(record)
        })
        print(f"[Mock] Would write to InfluxDB: {record}")
    
    def query_api(self):
        """Return mock query API"""
        return self
    
    def query(self, query, org):
        """Mock query method"""
        print(f"[Mock] Would query InfluxDB: {query}")
        return []
    
    def health(self):
        """Mock health check"""
        return type('obj', (object,), {'status': 'pass'})()
    
    def close(self):
        """Mock close method"""
        self.is_connected = False


class InfluxDBMetricsWriter:
    """Writes performance metrics to InfluxDB"""
    
    def __init__(self, config: InfluxDBConfig = None):
        self.config = config or InfluxDBConfig()
        self.client = None
        self.write_api = None
        self.query_api = None
        self.connected = False
        self.write_queue = queue.Queue(maxsize=10000)
        self.writer_thread = None
        self.running = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Connect to InfluxDB
        self._connect()
    
    def _connect(self):
        """Connect to InfluxDB"""
        try:
            if INFLUXDB_AVAILABLE:
                self.client = InfluxDBClient(
                    url=self.config.url,
                    token=self.config.token,
                    org=self.config.org,
                    timeout=self.config.timeout,
                    verify_ssl=self.config.verify_ssl
                )
                
                # Test connection
                health = self.client.health()
                if health.status == "pass":
                    self.connected = True
                    self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
                    self.query_api = self.client.query_api()
                    print(f"✅ Connected to InfluxDB at {self.config.url}")
                else:
                    print(f"❌ InfluxDB health check failed: {health.status}")
                    self._use_mock_client()
            else:
                self._use_mock_client()
                
        except Exception as e:
            print(f"❌ Failed to connect to InfluxDB: {e}")
            self._use_mock_client()
    
    def _use_mock_client(self):
        """Use mock client as fallback"""
        print("📝 Using mock InfluxDB client (data will not be persisted)")
        self.client = MockInfluxDBClient(self.config)
        self.write_api = self.client.write_api()
        self.query_api = self.client.query_api()
        self.connected = True
    
    def start(self):
        """Start the metrics writer"""
        self.running = True
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        print("🚀 InfluxDB metrics writer started")
    
    def stop(self):
        """Stop the metrics writer"""
        self.running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=5)
        
        # Flush remaining data
        self._flush_queue()
        
        # Close connection
        if self.client and hasattr(self.client, 'close'):
            self.client.close()
        
        print("✋ InfluxDB metrics writer stopped")
    
    def _writer_loop(self):
        """Main writer loop"""
        batch = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Get items from queue with timeout
                try:
                    item = self.write_queue.get(timeout=1.0)
                    batch.append(item)
                except queue.Empty:
                    pass
                
                # Flush if batch is full or interval exceeded
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (time.time() - last_flush) * 1000 >= self.config.flush_interval
                )
                
                if should_flush and batch:
                    self._write_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except Exception as e:
                self.logger.error(f"Error in writer loop: {e}")
                time.sleep(1)
    
    def _flush_queue(self):
        """Flush all remaining items in the queue"""
        batch = []
        while not self.write_queue.empty():
            try:
                batch.append(self.write_queue.get_nowait())
            except queue.Empty:
                break
        
        if batch:
            self._write_batch(batch)
    
    def _write_batch(self, batch: List[Any]):
        """Write a batch of points to InfluxDB"""
        if not self.connected or not self.write_api:
            return
        
        try:
            self.write_api.write(
                bucket=self.config.bucket,
                org=self.config.org,
                record=batch
            )
            self.logger.debug(f"Wrote {len(batch)} points to InfluxDB")
        except Exception as e:
            self.logger.error(f"Failed to write batch to InfluxDB: {e}")
    
    def write_metric_sample(self, sample: MetricSample):
        """Write a metric sample to InfluxDB"""
        try:
            point = Point("performance_metrics") \
                .tag("metric_type", sample.metric_type.value) \
                .field("value", float(sample.value)) \
                .time(sample.timestamp, WritePrecision.NS)
            
            # Add metadata as fields
            if sample.metadata:
                for key, value in sample.metadata.items():
                    if isinstance(value, (int, float, bool, str)):
                        point.field(key, value)
                    elif isinstance(value, dict):
                        # Flatten nested dict
                        for k, v in value.items():
                            if isinstance(v, (int, float, bool, str)):
                                point.field(f"{key}_{k}", v)
            
            # Add to write queue
            try:
                self.write_queue.put_nowait(point)
            except queue.Full:
                self.logger.warning("Write queue full, dropping metric")
                
        except Exception as e:
            self.logger.error(f"Error creating metric point: {e}")
    
    def write_alert(self, alert: Alert):
        """Write an alert to InfluxDB"""
        try:
            point = Point("alerts") \
                .tag("level", alert.level.value) \
                .tag("metric_type", alert.metric_type.value) \
                .field("message", alert.message) \
                .field("value", float(alert.value)) \
                .field("threshold", float(alert.threshold)) \
                .field("resolved", alert.resolved) \
                .time(alert.timestamp, WritePrecision.NS)
            
            if alert.duration:
                point.field("duration", float(alert.duration))
            
            # Add to write queue
            try:
                self.write_queue.put_nowait(point)
            except queue.Full:
                self.logger.warning("Write queue full, dropping alert")
                
        except Exception as e:
            self.logger.error(f"Error creating alert point: {e}")
    
    def query_metrics(
        self,
        metric_type: MetricType,
        start_time: str = "-1h",
        stop_time: str = "now()",
        aggregation: str = "mean"
    ) -> List[Dict]:
        """Query metrics from InfluxDB"""
        if not self.connected or not self.query_api:
            return []
        
        query = f'''
        from(bucket: "{self.config.bucket}")
            |> range(start: {start_time}, stop: {stop_time})
            |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
            |> filter(fn: (r) => r["metric_type"] == "{metric_type.value}")
            |> filter(fn: (r) => r["_field"] == "value")
            |> aggregateWindow(every: 1m, fn: {aggregation})
            |> yield(name: "{aggregation}")
        '''
        
        try:
            result = self.query_api.query(query=query, org=self.config.org)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        "time": record.get_time(),
                        "value": record.get_value(),
                        "metric_type": metric_type.value
                    })
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error querying metrics: {e}")
            return []
    
    def query_alerts(
        self,
        start_time: str = "-24h",
        stop_time: str = "now()",
        level: Optional[AlertLevel] = None
    ) -> List[Dict]:
        """Query alerts from InfluxDB"""
        if not self.connected or not self.query_api:
            return []
        
        query = f'''
        from(bucket: "{self.config.bucket}")
            |> range(start: {start_time}, stop: {stop_time})
            |> filter(fn: (r) => r["_measurement"] == "alerts")
        '''
        
        if level:
            query += f'''
            |> filter(fn: (r) => r["level"] == "{level.value}")
            '''
        
        query += '''
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> yield(name: "alerts")
        '''
        
        try:
            result = self.query_api.query(query=query, org=self.config.org)
            
            alerts = []
            for table in result:
                for record in table.records:
                    alerts.append({
                        "time": record.get_time(),
                        "level": record.values.get("level"),
                        "metric_type": record.values.get("metric_type"),
                        "message": record.values.get("message"),
                        "value": record.values.get("value"),
                        "threshold": record.values.get("threshold")
                    })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error querying alerts: {e}")
            return []
    
    def get_metrics_summary(
        self,
        start_time: str = "-1h"
    ) -> Dict[str, Dict]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for metric_type in MetricType:
            query = f'''
            from(bucket: "{self.config.bucket}")
                |> range(start: {start_time})
                |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
                |> filter(fn: (r) => r["metric_type"] == "{metric_type.value}")
                |> filter(fn: (r) => r["_field"] == "value")
                |> group()
            '''
            
            # Get statistics
            stats_queries = {
                "mean": f'{query} |> mean()',
                "min": f'{query} |> min()',
                "max": f'{query} |> max()',
                "count": f'{query} |> count()',
                "last": f'{query} |> last()'
            }
            
            metric_summary = {}
            
            for stat_name, stat_query in stats_queries.items():
                try:
                    result = self.query_api.query(query=stat_query, org=self.config.org)
                    if result:
                        for table in result:
                            for record in table.records:
                                metric_summary[stat_name] = record.get_value()
                except:
                    pass
            
            if metric_summary:
                summary[metric_type.value] = metric_summary
        
        return summary


class PerformanceMonitorWithInfluxDB:
    """Performance monitor with InfluxDB storage"""
    
    def __init__(
        self,
        influx_config: InfluxDBConfig = None,
        thresholds: PerformanceThresholds = None
    ):
        self.monitor = PerformanceMonitor(thresholds)
        self.influx_writer = InfluxDBMetricsWriter(influx_config)
        
        # Register callbacks
        self._register_callbacks()
    
    def _register_callbacks(self):
        """Register callbacks for metrics and alerts"""
        # Alert callback
        def alert_callback(alert: Alert):
            self.influx_writer.write_alert(alert)
        
        self.monitor.register_alert_callback(alert_callback)
    
    def start(self):
        """Start monitoring and storage"""
        self.monitor.start()
        self.influx_writer.start()
        
        # Start metric collection thread
        self.collection_thread = threading.Thread(
            target=self._collect_metrics,
            daemon=True
        )
        self.collection_thread.start()
        
        print("🚀 Performance monitoring with InfluxDB storage started")
    
    def stop(self):
        """Stop monitoring and storage"""
        self.monitor.stop()
        self.influx_writer.stop()
        print("✋ Performance monitoring with InfluxDB storage stopped")
    
    def _collect_metrics(self):
        """Collect and store metrics"""
        while self.monitor.running:
            try:
                # Collect samples from all collectors
                for collector in self.monitor.collectors.values():
                    recent_samples = collector.get_recent_samples(10)  # Last 10 seconds
                    for sample in recent_samples:
                        self.influx_writer.write_metric_sample(sample)
                
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                time.sleep(5)
    
    def get_dashboard_data(self, time_range: str = "-1h") -> Dict:
        """Get data for dashboard visualization"""
        data = {
            "current_stats": self.monitor.get_current_stats(),
            "system_info": self.monitor.get_system_info(),
            "recent_alerts": [a.to_dict() for a in list(self.monitor.alerts)[-10:]],
            "metrics_summary": self.influx_writer.get_metrics_summary(time_range)
        }
        
        # Query time series data for each metric
        for metric_type in MetricType:
            data[f"{metric_type.value}_series"] = self.influx_writer.query_metrics(
                metric_type,
                start_time=time_range
            )
        
        return data


# Example usage
def test_influxdb_metrics():
    """Test InfluxDB metrics integration"""
    print("\n" + "="*60)
    print("Testing InfluxDB Metrics Storage")
    print("="*60)
    
    # Create configuration
    config = InfluxDBConfig.from_env()
    
    # Create custom thresholds for testing
    thresholds = PerformanceThresholds(
        cpu_warning=50.0,
        cpu_critical=70.0,
        memory_warning=60.0,
        memory_critical=80.0
    )
    
    # Create monitor with InfluxDB storage
    monitor = PerformanceMonitorWithInfluxDB(config, thresholds)
    
    # Start monitoring
    monitor.start()
    
    print("\n📊 Collecting metrics for 15 seconds...")
    print("   Metrics are being written to InfluxDB")
    
    try:
        # Run for 15 seconds
        for i in range(15):
            time.sleep(1)
            
            # Get current stats
            stats = monitor.monitor.get_current_stats()
            cpu = stats.get(MetricType.CPU_USAGE.value)
            mem = stats.get(MetricType.MEMORY_USAGE.value)
            
            print(f"\r[{i+1}/15] CPU: {cpu.current:.1f}% | Memory: {mem.current:.1f}%", end="")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Get dashboard data
        print("\n\n📈 Dashboard Data:")
        dashboard_data = monitor.get_dashboard_data("-5m")
        
        # Display summary
        if "metrics_summary" in dashboard_data:
            print("\nMetrics Summary (last 5 minutes):")
            for metric, stats in dashboard_data["metrics_summary"].items():
                if stats:
                    print(f"  {metric}:")
                    for stat, value in stats.items():
                        if isinstance(value, float):
                            print(f"    {stat}: {value:.2f}")
                        else:
                            print(f"    {stat}: {value}")
        
        # Stop monitoring
        monitor.stop()
    
    return monitor


def setup_influxdb_docker():
    """Helper to set up InfluxDB using Docker"""
    print("\n📦 To set up InfluxDB with Docker:")
    print("="*60)
    print("""
# Pull and run InfluxDB 2.0
docker run -d \\
  --name influxdb \\
  -p 8086:8086 \\
  -v influxdb-storage:/var/lib/influxdb2 \\
  influxdb:2.7

# Access InfluxDB UI at http://localhost:8086
# Initial setup:
# 1. Create username and password
# 2. Create organization (e.g., 'default')
# 3. Create bucket (e.g., 'performance_metrics')
# 4. Generate API token

# Export environment variables:
export INFLUXDB_URL="http://localhost:8086"
export INFLUXDB_TOKEN="your-token-here"
export INFLUXDB_ORG="default"
export INFLUXDB_BUCKET="performance_metrics"
""")
    print("="*60)


if __name__ == "__main__":
    print("InfluxDB Metrics Storage")
    print("="*60)
    
    # Check if InfluxDB is available
    if not INFLUXDB_AVAILABLE:
        print("\n⚠️  InfluxDB client not installed!")
        print("   Install with: pip install influxdb-client")
        print("\n   Running with mock client for demonstration...")
        setup_influxdb_docker()
    
    # Test the integration
    monitor = test_influxdb_metrics()
    
    print("\n✅ InfluxDB metrics integration ready!")