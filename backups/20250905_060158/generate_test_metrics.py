#!/usr/bin/env python3
"""
Generate Test Metrics for InfluxDB
Creates sample data to demonstrate compression and other features
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

# Load environment
load_dotenv('.env.influxdb')

def generate_test_data():
    """Generate various types of test data"""
    
    client = InfluxDBClient(
        url=os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
        token=os.getenv('INFLUXDB_TOKEN'),
        org=os.getenv('INFLUXDB_ORG', 'agent-system')
    )
    
    write_api = client.write_api(write_options=SYNCHRONOUS)
    org = os.getenv('INFLUXDB_ORG', 'agent-system')
    
    print("🔄 Generating test metrics...")
    print("=" * 60)
    
    # 1. Performance metrics (high frequency)
    print("📊 Writing performance metrics...")
    points = []
    base_time = datetime.utcnow() - timedelta(hours=2)
    
    for i in range(500):
        timestamp = base_time + timedelta(seconds=i*5)
        
        point = Point("system_metrics") \
            .tag("host", f"server-{i % 3 + 1}") \
            .tag("datacenter", f"dc-{i % 2 + 1}") \
            .field("cpu_usage", 20 + random.uniform(0, 60)) \
            .field("memory_usage", 40 + random.uniform(0, 40)) \
            .field("disk_io", random.uniform(100, 1000)) \
            .field("network_in", random.uniform(1000, 10000)) \
            .field("network_out", random.uniform(500, 5000)) \
            .field("temperature", 60 + random.uniform(0, 20)) \
            .time(timestamp, WritePrecision.NS)
        
        points.append(point)
    
    write_api.write(bucket="performance_metrics", org=org, record=points)
    print(f"  ✅ Written {len(points)} performance data points")
    
    # 2. Agent metrics
    print("🤖 Writing agent metrics...")
    points = []
    
    for i in range(200):
        timestamp = base_time + timedelta(minutes=i)
        
        point = Point("agent_activity") \
            .tag("agent", random.choice(["researcher", "writer", "reviewer", "optimizer"])) \
            .tag("task_type", random.choice(["analysis", "generation", "review", "optimization"])) \
            .field("tasks_completed", random.randint(1, 10)) \
            .field("response_time_ms", random.uniform(100, 2000)) \
            .field("tokens_used", random.randint(100, 5000)) \
            .field("success_rate", random.uniform(0.8, 1.0)) \
            .time(timestamp, WritePrecision.NS)
        
        points.append(point)
    
    write_api.write(bucket="agent_metrics", org=org, record=points)
    print(f"  ✅ Written {len(points)} agent metric points")
    
    # 3. Alert events
    print("🚨 Writing alert events...")
    points = []
    
    for i in range(50):
        timestamp = base_time + timedelta(minutes=i*10)
        
        severity = random.choice(["info", "warning", "error", "critical"])
        
        point = Point("alert") \
            .tag("severity", severity) \
            .tag("component", random.choice(["api", "database", "cache", "queue"])) \
            .tag("host", f"server-{i % 3 + 1}") \
            .field("message", f"Alert {i}: {severity} condition detected") \
            .field("metric_value", random.uniform(0, 100)) \
            .field("threshold", 75.0) \
            .time(timestamp, WritePrecision.NS)
        
        points.append(point)
    
    write_api.write(bucket="alerts", org=org, record=points)
    print(f"  ✅ Written {len(points)} alert points")
    
    # 4. Test execution metrics
    print("🧪 Writing test metrics...")
    points = []
    
    for i in range(100):
        timestamp = base_time + timedelta(minutes=i*5)
        
        point = Point("test_execution") \
            .tag("test_suite", random.choice(["unit", "integration", "e2e", "performance"])) \
            .tag("project", "agent-lightning") \
            .field("tests_run", random.randint(50, 200)) \
            .field("tests_passed", random.randint(40, 190)) \
            .field("duration_ms", random.uniform(1000, 30000)) \
            .field("coverage_percent", random.uniform(60, 95)) \
            .time(timestamp, WritePrecision.NS)
        
        points.append(point)
    
    write_api.write(bucket="test_metrics", org=org, record=points)
    print(f"  ✅ Written {len(points)} test metric points")
    
    # 5. Deployment metrics
    print("🚀 Writing deployment metrics...")
    points = []
    
    for i in range(20):
        timestamp = base_time - timedelta(days=i)
        
        point = Point("deployment") \
            .tag("environment", random.choice(["dev", "staging", "production"])) \
            .tag("service", random.choice(["api", "web", "worker", "scheduler"])) \
            .tag("version", f"v1.{i}.0") \
            .field("duration_seconds", random.uniform(30, 300)) \
            .field("status", random.choice([1, 1, 1, 0])) \
            .field("rollback", random.choice([0, 0, 0, 0, 1])) \
            .time(timestamp, WritePrecision.NS)
        
        points.append(point)
    
    write_api.write(bucket="deployment_metrics", org=org, record=points)
    print(f"  ✅ Written {len(points)} deployment points")
    
    # Close client
    client.close()
    
    print("\n" + "=" * 60)
    print("✅ Test data generation complete!")
    print("\nTotal data points written: 870")
    print("\nYou can now:")
    print("  1. View data in Grafana dashboards")
    print("  2. Test compression with: python influxdb_compression.py compress")
    print("  3. Run aggregation jobs")
    print("  4. Create backups")

if __name__ == "__main__":
    generate_test_data()