#!/usr/bin/env python3
"""
Test InfluxDB Write
Simple script to test writing data to InfluxDB
"""

import time
import random
from datetime import datetime

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    
    # Configuration
    url = "http://localhost:8086"
    token = "my-super-secret-auth-token"
    org = "agent-system"
    bucket = "performance_metrics"
    
    # Create client
    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    print("📊 Sending test metrics to InfluxDB...")
    print("=" * 60)
    
    # Send test data in a loop
    for i in range(100):
        # Create points
        points = []
        
        # CPU metric
        cpu_point = Point("performance_metrics") \
            .tag("host", "agent-system") \
            .tag("metric_type", "cpu") \
            .field("value", random.uniform(20, 80)) \
            .time(datetime.utcnow())
        points.append(cpu_point)
        
        # Memory metric
        memory_point = Point("performance_metrics") \
            .tag("host", "agent-system") \
            .tag("metric_type", "memory") \
            .field("value", random.uniform(40, 90)) \
            .time(datetime.utcnow())
        points.append(memory_point)
        
        # Disk I/O metric
        disk_point = Point("performance_metrics") \
            .tag("host", "agent-system") \
            .tag("metric_type", "disk_io") \
            .field("read_bytes", random.uniform(0, 10000000)) \
            .field("write_bytes", random.uniform(0, 5000000)) \
            .time(datetime.utcnow())
        points.append(disk_point)
        
        # Network metric
        network_point = Point("performance_metrics") \
            .tag("host", "agent-system") \
            .tag("metric_type", "network") \
            .field("bytes_sent", random.uniform(0, 1000000)) \
            .field("bytes_recv", random.uniform(0, 2000000)) \
            .time(datetime.utcnow())
        points.append(network_point)
        
        # Agent metrics
        agent_point = Point("agent_metrics") \
            .tag("agent", "code_generator") \
            .field("tasks_completed", i) \
            .field("resource_usage", random.uniform(10, 50)) \
            .field("errors", random.randint(0, 2)) \
            .time(datetime.utcnow())
        points.append(agent_point)
        
        # Test metrics
        test_point = Point("test_metrics") \
            .tag("project", "agent-lightning") \
            .field("coverage", random.uniform(70, 95)) \
            .field("execution_time", random.uniform(0.5, 5.0)) \
            .field("tests_passed", random.randint(80, 100)) \
            .field("tests_failed", random.randint(0, 5)) \
            .time(datetime.utcnow())
        points.append(test_point)
        
        # Alert (occasionally)
        if random.random() > 0.8:
            alert_point = Point("alerts") \
                .tag("level", random.choice(["warning", "critical", "info"])) \
                .tag("metric", random.choice(["cpu", "memory", "disk"])) \
                .field("message", f"Test alert {i}") \
                .field("value", random.uniform(0, 100)) \
                .time(datetime.utcnow())
            points.append(alert_point)
        
        # Write points
        try:
            write_api.write(bucket=bucket, org=org, record=points)
            print(f"✅ Batch {i+1}/100 sent: {len(points)} metrics")
        except Exception as e:
            print(f"❌ Error writing batch {i+1}: {e}")
        
        # Wait before next batch
        time.sleep(2)
    
    print("\n✅ Test completed! Check Grafana dashboards for data.")
    client.close()
    
except ImportError:
    print("❌ InfluxDB client not installed.")
    print("Install with: pip install influxdb-client")
except Exception as e:
    print(f"❌ Error: {e}")