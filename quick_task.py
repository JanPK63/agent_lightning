#!/usr/bin/env python3
"""
Quick Task Assignment for Agent Lightning
Simple script to quickly assign tasks to agents
"""

import requests
import sys
import json

def quick_task(task_description: str):
    """Quickly assign a task to the best agent"""
    
    # API endpoint
    base_url = "http://localhost:8001"
    
    # First authenticate
    auth_response = requests.post(
        f"{base_url}/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    
    if auth_response.status_code != 200:
        print("❌ Authentication failed. Is the API running on port 8001?")
        return
    
    token = auth_response.json()["access_token"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Create task
    print(f"\n📝 Task: {task_description}")
    print("-" * 50)
    
    task_response = requests.post(
        f"{base_url}/tasks",
        json={
            "description": task_description,
            "agent_type": "auto",  # Let system choose best agent
            "priority": "normal"
        },
        headers=headers
    )
    
    if task_response.status_code != 200:
        print(f"❌ Failed to create task: {task_response.text}")
        return
    
    task = task_response.json()
    task_id = task["task_id"]
    assigned_agent = task.get("assigned_agent", "auto")
    
    print(f"✅ Task created (ID: {task_id})")
    print(f"🤖 Assigned to: {assigned_agent}")
    print("\n⏳ Processing...")
    
    # Execute task
    exec_response = requests.post(
        f"{base_url}/tasks/{task_id}/execute",
        headers=headers
    )
    
    if exec_response.status_code != 200:
        print(f"❌ Failed to execute: {exec_response.text}")
        return
    
    result = exec_response.json()
    
    print("\n📊 RESULT:")
    print("-" * 50)
    
    if "result" in result:
        print(result["result"])
    elif "response" in result:
        print(result["response"])
    else:
        print(json.dumps(result, indent=2))
    
    print("\n✅ Task completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n⚡ Agent Lightning - Quick Task Assignment")
        print("-" * 40)
        print("\nUsage: python quick_task.py \"your task description\"")
        print("\nExamples:")
        print('  python quick_task.py "Research the benefits of solar energy"')
        print('  python quick_task.py "Write a haiku about coding"')
        print('  python quick_task.py "Review this Python function for bugs: def add(a,b): return a+b"')
        print('  python quick_task.py "Optimize database query performance"')
        sys.exit(1)
    
    task = " ".join(sys.argv[1:])
    quick_task(task)