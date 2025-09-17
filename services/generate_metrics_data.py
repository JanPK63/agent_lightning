#!/usr/bin/env python3
"""
Generate test data for metrics visualization
Populates the system with sample tasks and metrics
"""

import requests
import json
import time
import random
import uuid
from datetime import datetime

def generate_test_data():
    """Generate test data for metrics"""
    
    print("=" * 60)
    print("Generating Test Data for Metrics")
    print("=" * 60)
    
    # Service URLs
    coordination_url = "http://localhost:8030"
    metrics_url = "http://localhost:8031"
    
    # Sample task descriptions for different agents
    task_templates = {
        "web_developer": [
            "Create a responsive landing page",
            "Build a REST API endpoint",
            "Implement user authentication",
            "Create a dashboard component",
            "Add form validation"
        ],
        "security_expert": [
            "Perform security audit",
            "Review authentication implementation",
            "Check for SQL injection vulnerabilities",
            "Analyze encryption methods",
            "Test API security"
        ],
        "tester": [
            "Write unit tests for user service",
            "Create integration tests",
            "Perform load testing",
            "Test error handling",
            "Validate API responses"
        ],
        "database_specialist": [
            "Optimize database queries",
            "Create database indexes",
            "Design database schema",
            "Implement data migration",
            "Setup database replication"
        ],
        "devops_engineer": [
            "Setup CI/CD pipeline",
            "Configure Docker containers",
            "Deploy to production",
            "Setup monitoring alerts",
            "Configure load balancer"
        ]
    }
    
    print("\n1. Generating tasks to populate system...")
    
    tasks_created = []
    
    # Generate 50 tasks
    for i in range(50):
        agent_type = random.choice(list(task_templates.keys()))
        task_desc = random.choice(task_templates[agent_type])
        
        task_data = {
            "description": f"{task_desc} - Test #{i+1}",
            "priority": random.randint(1, 10),
            "user_id": f"test_user_{random.randint(1, 5)}",
            "skip_validation": True,  # Skip to speed up
            "skip_governance": True   # Skip to speed up
        }
        
        try:
            response = requests.post(f"{coordination_url}/execute", json=task_data)
            if response.status_code == 200:
                result = response.json()
                tasks_created.append(result['task_id'])
                print(f"   Task {i+1}/50: {result['agent_id']} - {result['status']}")
            
            # Small delay to not overwhelm the system
            time.sleep(0.2)
            
        except Exception as e:
            print(f"   Error creating task {i+1}: {e}")
    
    print(f"\n   Created {len(tasks_created)} tasks")
    
    # 2. Generate custom metrics
    print("\n2. Generating custom metrics...")
    
    metric_types = [
        ("api_latency", "histogram", lambda: random.uniform(0.01, 0.5)),
        ("cache_hits", "counter", lambda: random.randint(1, 100)),
        ("active_connections", "gauge", lambda: random.randint(10, 50)),
        ("error_count", "counter", lambda: random.randint(0, 5)),
        ("cpu_usage", "gauge", lambda: random.uniform(10, 90)),
        ("memory_usage", "gauge", lambda: random.uniform(20, 80))
    ]
    
    for i in range(20):
        for metric_name, metric_type, value_generator in metric_types:
            metric_data = {
                "metric_name": metric_name,
                "metric_type": metric_type,
                "value": value_generator(),
                "labels": {
                    "service": random.choice(["auth", "api", "database", "cache"]),
                    "environment": "production"
                }
            }
            
            try:
                response = requests.post(f"{metrics_url}/record", json=metric_data)
                if response.status_code == 200:
                    print(f"   Recorded {metric_name}: {metric_data['value']:.2f}")
            except:
                pass
            
            time.sleep(0.1)
    
    # 3. Check system metrics
    print("\n3. Checking system metrics...")
    
    try:
        response = requests.get(f"{metrics_url}/metrics/system")
        if response.status_code == 200:
            metrics = response.json()
            print(f"   Total agents: {metrics['total_agents']}")
            print(f"   Active agents: {metrics['active_agents']}")
            print(f"   Total tasks: {metrics['total_tasks']}")
            print(f"   Tasks completed: {metrics['tasks_completed']}")
            print(f"   Tasks failed: {metrics['tasks_failed']}")
            print(f"   System throughput: {metrics['system_throughput']:.2f} tasks/min")
            print(f"   Error rate: {metrics['error_rate']:.2f}%")
            print(f"   Cache hit rate: {metrics['cache_hit_rate']:.2f}%")
    except Exception as e:
        print(f"   Error getting system metrics: {e}")
    
    # 4. Check agent performance
    print("\n4. Checking agent performance...")
    
    try:
        response = requests.get(f"{coordination_url}/agents/performance")
        if response.status_code == 200:
            performance = response.json()
            print(f"   Total agents tracked: {performance['total_agents']}")
            
            for agent_id, perf in list(performance['agents'].items())[:5]:
                if perf:
                    print(f"\n   {agent_id}:")
                    print(f"     Total tasks: {perf.get('total_tasks', 0)}")
                    print(f"     Success rate: {perf.get('success_rate', 0):.2%}")
    except Exception as e:
        print(f"   Error getting agent performance: {e}")
    
    # 5. Check Prometheus metrics
    print("\n5. Checking Prometheus metrics endpoint...")
    
    try:
        response = requests.get(f"{metrics_url}/metrics/prometheus")
        if response.status_code == 200:
            prometheus_data = response.text[:500]  # First 500 chars
            print(f"   Prometheus metrics available: {len(response.text)} bytes")
            print(f"   Sample:\n{prometheus_data}...")
    except Exception as e:
        print(f"   Error getting Prometheus metrics: {e}")
    
    # 6. Check Grafana endpoint
    print("\n6. Checking Grafana-compatible endpoint...")
    
    try:
        response = requests.get(f"{metrics_url}/metrics/grafana")
        if response.status_code == 200:
            grafana_data = response.json()
            print(f"   Grafana metrics available: {len(grafana_data)} series")
            for metric in grafana_data[:3]:
                print(f"   - {metric['target']}: {metric['datapoints'][0][0]:.2f}")
    except Exception as e:
        print(f"   Error getting Grafana metrics: {e}")
    
    print("\n" + "=" * 60)
    print("Test Data Generation Complete!")
    print("=" * 60)
    print("\n✅ Metrics are now available at:")
    print("  • Streamlit Dashboard: http://localhost:8051")
    print("  • Grafana: http://localhost:3000")
    print("  • Prometheus metrics: http://localhost:8031/metrics/prometheus")
    print("  • System metrics API: http://localhost:8031/metrics/system")
    print("  • Realtime metrics: http://localhost:8031/metrics/realtime")
    
    print("\n📊 To view metrics in Grafana:")
    print("  1. Open http://localhost:3000")
    print("  2. Login with admin/admin")
    print("  3. Go to Dashboards")
    print("  4. Data sources are already configured")
    print("  5. Create a new dashboard or import one")

if __name__ == "__main__":
    generate_test_data()