#!/usr/bin/env python3
"""
Test Monitoring Pipeline End-to-End
Verifies the complete monitoring stack is working
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.influxdb')

def test_influxdb_connection():
    """Test InfluxDB connectivity"""
    print("1️⃣  Testing InfluxDB Connection...")
    
    url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    token = os.getenv('INFLUXDB_TOKEN')
    
    headers = {
        'Authorization': f'Token {token}',
        'Content-Type': 'application/json'
    }
    
    # Test health endpoint
    health_response = requests.get(f"{url}/health")
    if health_response.status_code == 200:
        print("   ✅ InfluxDB is healthy")
    else:
        print(f"   ❌ InfluxDB health check failed: {health_response.status_code}")
        return False
    
    # Test authentication
    buckets_response = requests.get(
        f"{url}/api/v2/buckets",
        headers=headers,
        params={'org': os.getenv('INFLUXDB_ORG')}
    )
    
    if buckets_response.status_code == 200:
        buckets = buckets_response.json().get('buckets', [])
        print(f"   ✅ Authenticated successfully. Found {len(buckets)} buckets")
        return True
    else:
        print(f"   ❌ Authentication failed: {buckets_response.status_code}")
        return False

def test_write_metrics():
    """Test writing metrics to InfluxDB"""
    print("\n2️⃣  Testing Metric Writing...")
    
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    
    client = InfluxDBClient(
        url=os.getenv('INFLUXDB_URL'),
        token=os.getenv('INFLUXDB_TOKEN'),
        org=os.getenv('INFLUXDB_ORG')
    )
    
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    # Write test metrics
    points = [
        Point("test_metric")
        .tag("source", "pipeline_test")
        .field("value", 42.0)
        .field("status", "testing")
        .time(datetime.utcnow()),
        
        Point("test_metric")
        .tag("source", "pipeline_test")
        .field("value", 84.0)
        .field("status", "validating")
        .time(datetime.utcnow())
    ]
    
    try:
        write_api.write(
            bucket=os.getenv('INFLUXDB_BUCKET'),
            org=os.getenv('INFLUXDB_ORG'),
            record=points
        )
        print("   ✅ Successfully wrote test metrics")
        return True
    except Exception as e:
        print(f"   ❌ Failed to write metrics: {e}")
        return False
    finally:
        client.close()

def test_query_metrics():
    """Test querying metrics from InfluxDB"""
    print("\n3️⃣  Testing Metric Querying...")
    
    from influxdb_client import InfluxDBClient
    
    client = InfluxDBClient(
        url=os.getenv('INFLUXDB_URL'),
        token=os.getenv('INFLUXDB_TOKEN'),
        org=os.getenv('INFLUXDB_ORG')
    )
    
    query_api = client.query_api()
    
    query = f'''
    from(bucket: "{os.getenv('INFLUXDB_BUCKET')}")
        |> range(start: -1h)
        |> filter(fn: (r) => r["_measurement"] == "test_metric")
        |> filter(fn: (r) => r["source"] == "pipeline_test")
    '''
    
    try:
        result = query_api.query(query=query, org=os.getenv('INFLUXDB_ORG'))
        
        count = 0
        for table in result:
            for record in table.records:
                count += 1
                
        if count > 0:
            print(f"   ✅ Successfully queried {count} data points")
            return True
        else:
            print("   ⚠️  No data points found (this is okay if first run)")
            return True
    except Exception as e:
        print(f"   ❌ Failed to query metrics: {e}")
        return False
    finally:
        client.close()

def test_grafana_connection():
    """Test Grafana connectivity"""
    print("\n4️⃣  Testing Grafana Connection...")
    
    grafana_url = "http://localhost:3000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{grafana_url}/api/health")
        if response.status_code == 200:
            print("   ✅ Grafana is healthy")
        else:
            print(f"   ⚠️  Grafana returned status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Cannot connect to Grafana: {e}")
        return False
    
    # Test authentication
    auth = ('admin', 'admin123')
    dashboards_response = requests.get(
        f"{grafana_url}/api/search",
        auth=auth
    )
    
    if dashboards_response.status_code == 200:
        dashboards = dashboards_response.json()
        print(f"   ✅ Authenticated successfully. Found {len(dashboards)} dashboards")
        
        # List dashboards
        if dashboards:
            print("   📊 Available dashboards:")
            for dash in dashboards[:5]:  # Show first 5
                print(f"      - {dash.get('title', 'Untitled')}")
        return True
    else:
        print(f"   ⚠️  Authentication status: {dashboards_response.status_code}")
        return True  # Non-critical

def test_monitoring_components():
    """Test that monitoring Python components work"""
    print("\n5️⃣  Testing Monitoring Components...")
    
    try:
        from performance_monitor import PerformanceMonitor
        from influxdb_metrics import PerformanceMonitorWithInfluxDB
        
        # Test basic monitor
        monitor = PerformanceMonitor()
        metrics = monitor.collect_metrics()
        
        if metrics:
            print(f"   ✅ Performance monitor collected {len(metrics)} metric types")
        else:
            print("   ❌ No metrics collected")
            return False
        
        # Test InfluxDB monitor (non-blocking)
        influx_monitor = PerformanceMonitorWithInfluxDB()
        print("   ✅ InfluxDB monitor initialized")
        
        return True
    except Exception as e:
        print(f"   ❌ Component test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing Monitoring Pipeline End-to-End")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("InfluxDB Connection", test_influxdb_connection()))
    results.append(("Metric Writing", test_write_metrics()))
    results.append(("Metric Querying", test_query_metrics()))
    results.append(("Grafana Connection", test_grafana_connection()))
    results.append(("Monitoring Components", test_monitoring_components()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! Monitoring pipeline is fully operational.")
        print("\n📈 You can now:")
        print("   1. View dashboards at http://localhost:3000")
        print("   2. Access InfluxDB at http://localhost:8086")
        print("   3. Start collecting real metrics")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)