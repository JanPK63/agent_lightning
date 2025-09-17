#!/usr/bin/env python3
"""
Start Performance Monitoring
Begins collecting and sending metrics to InfluxDB
"""

import os
import sys
import time
import threading
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from performance_monitor import PerformanceMonitor
from influxdb_metrics import InfluxDBMetricsWriter, MetricType

def start_monitoring():
    """Start the performance monitoring system"""
    print("🚀 Starting Performance Monitoring System")
    print("=" * 60)
    
    # Set environment variables if not set
    if not os.getenv("INFLUXDB_TOKEN"):
        os.environ["INFLUXDB_TOKEN"] = "my-super-secret-auth-token"
    if not os.getenv("INFLUXDB_ORG"):
        os.environ["INFLUXDB_ORG"] = "agent-system"
    if not os.getenv("INFLUXDB_BUCKET"):
        os.environ["INFLUXDB_BUCKET"] = "performance_metrics"
    
    # Initialize components
    monitor = PerformanceMonitor()
    writer = InfluxDBMetricsWriter()
    collection_interval = 5  # seconds
    
    print("📊 Components initialized:")
    print(f"   - Performance Monitor: ✅")
    print(f"   - InfluxDB Writer: ✅")
    print(f"   - Collection Interval: {collection_interval} seconds")
    
    # Start monitoring in a separate thread
    def monitor_loop():
        """Main monitoring loop"""
        print("\n🔄 Starting metric collection loop...")
        iteration = 0
        
        while True:
            try:
                iteration += 1
                
                # Collect metrics
                metrics = monitor.collect_metrics()
                
                # Display summary
                if iteration % 10 == 1:  # Show header every 10 iterations
                    print("\n" + "="*60)
                    print(f"{'Time':<20} {'CPU%':<10} {'Memory%':<10} {'Disk IO':<15} {'Network':<15}")
                    print("-"*60)
                
                # Extract key metrics for display
                cpu_percent = 0
                memory_percent = 0
                disk_io = 0
                network_io = 0
                
                for metric in metrics:
                    if metric.metric_type == MetricType.CPU_USAGE:
                        cpu_percent = metric.value.get('percent', 0)
                    elif metric.metric_type == MetricType.MEMORY_USAGE:
                        memory_percent = metric.value.get('percent', 0)
                    elif metric.metric_type == MetricType.DISK_IO:
                        disk_io = metric.value.get('read_bytes', 0) + metric.value.get('write_bytes', 0)
                    elif metric.metric_type == MetricType.NETWORK_IO:
                        network_io = metric.value.get('bytes_sent', 0) + metric.value.get('bytes_recv', 0)
                
                # Format values
                disk_io_str = f"{disk_io / 1024 / 1024:.1f} MB/s" if disk_io > 0 else "0 MB/s"
                network_io_str = f"{network_io / 1024 / 1024:.1f} MB/s" if network_io > 0 else "0 MB/s"
                
                print(f"{datetime.now().strftime('%H:%M:%S'):<20} "
                      f"{cpu_percent:<10.1f} "
                      f"{memory_percent:<10.1f} "
                      f"{disk_io_str:<15} "
                      f"{network_io_str:<15}")
                
                # Write to InfluxDB
                for metric in metrics:
                    try:
                        writer.write_metric(metric)
                    except Exception as e:
                        if iteration == 1:  # Only show error on first iteration
                            print(f"⚠️  InfluxDB write error: {e}")
                
                # Also write some test/agent metrics for demo purposes
                if iteration % 5 == 0:  # Every 5 iterations
                    # Simulate agent metrics
                    from metric_models import Metric
                    
                    agent_metric = Metric(
                        metric_type=MetricType.CUSTOM,
                        value={
                            "tasks_completed": iteration,
                            "agent": "code_generator",
                            "status": "active"
                        },
                        timestamp=datetime.now(),
                        tags={"agent": "code_generator"},
                        metadata={"measurement": "agent_metrics"}
                    )
                    writer.write_metric(agent_metric)
                    
                    # Simulate test metrics
                    test_metric = Metric(
                        metric_type=MetricType.CUSTOM,
                        value={
                            "coverage": 75.5 + (iteration % 10),
                            "project": "agent-lightning",
                            "tests_passed": iteration * 2,
                            "tests_failed": max(0, 5 - iteration % 6)
                        },
                        timestamp=datetime.now(),
                        tags={"project": "agent-lightning"},
                        metadata={"measurement": "test_metrics"}
                    )
                    writer.write_metric(test_metric)
                    
                    # Simulate alerts
                    if cpu_percent > 50:
                        alert_metric = Metric(
                            metric_type=MetricType.CUSTOM,
                            value={
                                "level": "warning" if cpu_percent < 70 else "critical",
                                "message": f"High CPU usage: {cpu_percent:.1f}%",
                                "metric": "cpu"
                            },
                            timestamp=datetime.now(),
                            tags={"level": "warning", "metric": "cpu"},
                            metadata={"measurement": "alerts"}
                        )
                        writer.write_metric(alert_metric)
                
                # Wait for next collection
                time.sleep(collection_interval)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    
    print("\n✅ Monitoring started successfully!")
    print("\n📊 View metrics in Grafana:")
    print("   http://localhost:3000")
    print("\n📈 View InfluxDB:")
    print("   http://localhost:8086")
    print("\nPress Ctrl+C to stop monitoring...")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down monitoring system...")
        sys.exit(0)


if __name__ == "__main__":
    start_monitoring()