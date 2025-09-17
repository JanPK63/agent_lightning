#!/usr/bin/env python3
"""
Agent Lightning Client - Interact with your AI agents
This script shows how to assign tasks and get responses from agents
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class AgentLightningClient:
    """Client for interacting with Agent Lightning API"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.token = None
        self.headers = {"Content-Type": "application/json"}
    
    def authenticate(self, username: str = "admin", password: str = "admin123"):
        """Authenticate with the API"""
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.headers["Authorization"] = f"Bearer {self.token}"
            print("✅ Authenticated successfully!")
            return True
        else:
            print(f"❌ Authentication failed: {response.text}")
            return False
    
    def create_task(self, task_description: str, agent_type: str = "auto", context: Dict = None):
        """Create a new task for an agent"""
        task_data = {
            "description": task_description,
            "agent_type": agent_type,
            "context": context or {},
            "priority": "normal"
        }
        
        response = requests.post(
            f"{self.base_url}/tasks",
            json=task_data,
            headers=self.headers
        )
        
        if response.status_code == 200:
            task = response.json()
            print(f"✅ Task created: {task['task_id']}")
            return task
        else:
            print(f"❌ Failed to create task: {response.text}")
            return None
    
    def execute_task(self, task_id: str):
        """Execute a task and get the result"""
        response = requests.post(
            f"{self.base_url}/tasks/{task_id}/execute",
            headers=self.headers
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Task executed successfully!")
            return result
        else:
            print(f"❌ Failed to execute task: {response.text}")
            return None
    
    def chat_with_agent(self, message: str, agent_id: str = "researcher"):
        """Have a conversation with a specific agent"""
        chat_data = {
            "message": message,
            "agent_id": agent_id,
            "context": {}
        }
        
        response = requests.post(
            f"{self.base_url}/agents/{agent_id}/chat",
            json=chat_data,
            headers=self.headers
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            print(f"❌ Chat failed: {response.text}")
            return None
    
    def get_agent_status(self, agent_id: str):
        """Get the status of a specific agent"""
        response = requests.get(
            f"{self.base_url}/agents/{agent_id}/status",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def list_available_agents(self):
        """List all available agents"""
        response = requests.get(
            f"{self.base_url}/agents",
            headers=self.headers
        )
        
        if response.status_code == 200:
            agents = response.json()["agents"]
            print("\n📋 Available Agents:")
            print("-" * 40)
            for agent in agents:
                print(f"  • {agent['id']}: {agent['name']} ({agent['model']})")
                print(f"    Specialization: {agent['specialization']}")
                print(f"    Status: {agent['status']}")
                print()
            return agents
        return []


def main():
    """Example usage of Agent Lightning Client"""
    
    # Initialize client
    client = AgentLightningClient()
    
    print("=" * 60)
    print("⚡ AGENT LIGHTNING CLIENT")
    print("=" * 60)
    
    # Authenticate
    if not client.authenticate():
        return
    
    # List available agents
    client.list_available_agents()
    
    print("\n" + "=" * 60)
    print("EXAMPLE TASKS YOU CAN ASSIGN TO AGENTS:")
    print("=" * 60)
    
    examples = [
        {
            "description": "Research the latest developments in quantum computing",
            "agent": "researcher",
            "type": "Research Task"
        },
        {
            "description": "Write a blog post about AI safety best practices",
            "agent": "writer",
            "type": "Content Creation"
        },
        {
            "description": "Review this code for security vulnerabilities: def login(user, pass): return user == 'admin'",
            "agent": "reviewer",
            "type": "Code Review"
        },
        {
            "description": "Optimize the performance of our database queries",
            "agent": "optimizer",
            "type": "Performance Optimization"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['type']}:")
        print(f"   Task: {example['description']}")
        print(f"   Best Agent: {example['agent']}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Assign Tasks to Agents")
    print("=" * 60)
    print("\nType 'help' for commands, 'quit' to exit\n")
    
    while True:
        command = input("🤖 > ").strip()
        
        if command.lower() == 'quit':
            print("Goodbye! 👋")
            break
        
        elif command.lower() == 'help':
            print("\nAvailable commands:")
            print("  task <description>  - Create and execute a task")
            print("  chat <agent> <msg>  - Chat with a specific agent")
            print("  status <agent>      - Get agent status")
            print("  agents              - List all agents")
            print("  example <1-4>       - Run an example task")
            print("  quit                - Exit the client\n")
        
        elif command.startswith('task '):
            task_desc = command[5:]
            print(f"\n📝 Creating task: {task_desc}")
            task = client.create_task(task_desc)
            if task:
                print("⏳ Executing task...")
                result = client.execute_task(task['task_id'])
                if result:
                    print(f"\n📊 Result:\n{json.dumps(result, indent=2)}")
        
        elif command.startswith('chat '):
            parts = command[5:].split(' ', 1)
            if len(parts) == 2:
                agent_id, message = parts
                print(f"\n💬 Chatting with {agent_id}...")
                response = client.chat_with_agent(message, agent_id)
                if response:
                    print(f"\n{agent_id}: {response}")
        
        elif command.startswith('status '):
            agent_id = command[7:]
            status = client.get_agent_status(agent_id)
            if status:
                print(f"\n📊 Status for {agent_id}:")
                print(json.dumps(status, indent=2))
        
        elif command == 'agents':
            client.list_available_agents()
        
        elif command.startswith('example '):
            try:
                idx = int(command[8:]) - 1
                if 0 <= idx < len(examples):
                    example = examples[idx]
                    print(f"\n🚀 Running example: {example['type']}")
                    task = client.create_task(example['description'], example['agent'])
                    if task:
                        result = client.execute_task(task['task_id'])
                        if result:
                            print(f"\n📊 Result:\n{json.dumps(result, indent=2)}")
            except (ValueError, IndexError):
                print("Invalid example number. Use 1-4.")
        
        else:
            print("Unknown command. Type 'help' for available commands.")


if __name__ == "__main__":
    main()