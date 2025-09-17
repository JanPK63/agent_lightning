#!/usr/bin/env python3
"""
Create InfluxDB Aggregation Tasks via API
Simple script to create data aggregation tasks
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment
load_dotenv('.env.influxdb')

# Configuration
INFLUXDB_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', 'agent-system-token-supersecret-12345678')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'agent-system')

headers = {
    'Authorization': f'Token {INFLUXDB_TOKEN}',
    'Content-Type': 'application/json'
}

# Define aggregation tasks
tasks = [
    {
        "name": "performance_5min_avg",
        "description": "Aggregate performance metrics to 5-minute averages",
        "status": "active",
        "every": "5m",
        "flux": '''
option task = {name: "performance_5min_avg", every: 5m}

from(bucket: "performance_metrics")
    |> range(start: -5m)
    |> filter(fn: (r) => r["_measurement"] == "performance")
    |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
    |> set(key: "aggregation", value: "5min")
    |> to(bucket: "performance_metrics_long", org: "agent-system")
'''
    },
    {
        "name": "performance_hourly",
        "description": "Create hourly performance summaries",
        "status": "active", 
        "every": "1h",
        "flux": '''
option task = {name: "performance_hourly", every: 1h}

from(bucket: "performance_metrics")
    |> range(start: -1h)
    |> filter(fn: (r) => r["_measurement"] == "performance")
    |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    |> set(key: "aggregation", value: "hourly")
    |> to(bucket: "performance_metrics_long", org: "agent-system")
'''
    },
    {
        "name": "agent_daily_summary",
        "description": "Daily summary of agent activities",
        "status": "active",
        "every": "24h",
        "flux": '''
option task = {name: "agent_daily_summary", every: 24h}

from(bucket: "agent_metrics")
    |> range(start: -24h)
    |> filter(fn: (r) => r["_measurement"] == "agent_activity")
    |> aggregateWindow(every: 24h, fn: sum, createEmpty: false)
    |> set(key: "aggregation", value: "daily")
    |> to(bucket: "agent_metrics", org: "agent-system")
'''
    },
    {
        "name": "alerts_hourly_summary",
        "description": "Hourly alert summary by severity",
        "status": "active",
        "every": "1h",
        "flux": '''
option task = {name: "alerts_hourly_summary", every: 1h}

from(bucket: "alerts")
    |> range(start: -1h)
    |> filter(fn: (r) => r["_measurement"] == "alert")
    |> group(columns: ["severity"])
    |> count()
    |> set(key: "aggregation", value: "hourly_count")
    |> to(bucket: "alerts", org: "agent-system")
'''
    },
    {
        "name": "test_metrics_daily",
        "description": "Daily test execution summary",
        "status": "active",
        "every": "24h",
        "flux": '''
option task = {name: "test_metrics_daily", every: 24h}

from(bucket: "test_metrics")
    |> range(start: -24h)
    |> filter(fn: (r) => r["_measurement"] == "test_execution")
    |> aggregateWindow(every: 24h, fn: mean, createEmpty: false)
    |> set(key: "aggregation", value: "daily")
    |> to(bucket: "test_metrics", org: "agent-system")
'''
    }
]

def get_org_id():
    """Get organization ID"""
    response = requests.get(
        f"{INFLUXDB_URL}/api/v2/orgs",
        headers=headers,
        params={"org": INFLUXDB_ORG}
    )
    
    if response.status_code == 200:
        orgs = response.json().get('orgs', [])
        if orgs:
            return orgs[0]['id']
    return None

def list_tasks():
    """List existing tasks"""
    response = requests.get(
        f"{INFLUXDB_URL}/api/v2/tasks",
        headers=headers
    )
    
    if response.status_code == 200:
        existing_tasks = response.json().get('tasks', [])
        print(f"\nExisting tasks: {len(existing_tasks)}")
        for task in existing_tasks:
            print(f"  - {task['name']}: {task['status']}")
        return [t['name'] for t in existing_tasks]
    return []

def create_task(task_config, org_id):
    """Create a single task"""
    task_config['orgID'] = org_id
    
    response = requests.post(
        f"{INFLUXDB_URL}/api/v2/tasks",
        headers=headers,
        json=task_config
    )
    
    if response.status_code == 201:
        print(f"✅ Created task: {task_config['name']}")
        return True
    else:
        print(f"❌ Failed to create task '{task_config['name']}': {response.status_code}")
        if response.text:
            print(f"   Error: {response.text}")
        return False

def main():
    print("🔄 Creating InfluxDB Aggregation Tasks")
    print("=" * 60)
    
    # Get organization ID
    org_id = get_org_id()
    if not org_id:
        print("❌ Failed to get organization ID")
        return
    
    print(f"Organization ID: {org_id}")
    
    # List existing tasks
    existing = list_tasks()
    
    # Create tasks
    print("\nCreating aggregation tasks...")
    created = 0
    skipped = 0
    failed = 0
    
    for task in tasks:
        if task['name'] in existing:
            print(f"⏭️  Skipping existing task: {task['name']}")
            skipped += 1
        else:
            if create_task(task, org_id):
                created += 1
            else:
                failed += 1
    
    print("\n" + "=" * 60)
    print(f"Summary: {created} created, {skipped} skipped, {failed} failed")
    
    if created > 0:
        print("\n✅ Data aggregation jobs are now active!")
        print("\nThese tasks will:")
        print("  • Downsample high-resolution metrics for long-term storage")
        print("  • Create hourly and daily summaries")
        print("  • Reduce storage requirements while preserving trends")
        print("  • Run automatically at scheduled intervals")

if __name__ == "__main__":
    main()