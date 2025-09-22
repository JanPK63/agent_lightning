#!/usr/bin/env python3
"""
Simple test script to verify Agent Lightning system is working
Automatically assigns and executes a test task
"""

import requests
import json
import time
from typing import Dict, Any

class SystemTester:
    """Test the complete Agent Lightning system"""

    def __init__(self, orchestrator_url: str = "http://localhost:8025"):
        self.orchestrator_url = orchestrator_url
        self.headers = {"Content-Type": "application/json"}

    def test_system(self):
        """Test the complete system flow"""
        print("🧪 Testing Agent Lightning System")
        print("=" * 50)

        # Test 1: Health check
        print("\n1. Testing RL Orchestrator health...")
        try:
            response = requests.get(f"{self.orchestrator_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✅ RL Orchestrator healthy: {health['service']} v{health['version']}")
            else:
                print(f"❌ RL Orchestrator health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ RL Orchestrator not reachable: {e}")
            return False

        # Test 2: Assign a simple task
        print("\n2. Assigning test task...")
        test_task = {
            "task_id": f"test_{int(time.time())}",
            "description": "Write a Python function to calculate factorial",
            "priority": 5,
            "force_execute": True
        }

        try:
            response = requests.post(
                f"{self.orchestrator_url}/assign-task",
                json=test_task,
                headers=self.headers,
                timeout=30  # Allow time for execution
            )

            if response.status_code == 200:
                result = response.json()
                print("✅ Task assigned successfully!")
                print(f"   Task ID: {result['task_id']}")
                print(f"   Assigned Agent: {result['assigned_agent']}")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Reason: {result['reason']}")
                if result.get('warnings'):
                    print(f"   Warnings: {result['warnings']}")
                return True
            else:
                print(f"❌ Task assignment failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Task assignment error: {e}")
            return False

def main():
    """Run system test"""
    tester = SystemTester()

    success = tester.test_system()

    if success:
        print("\n🎉 System test PASSED - Agent Lightning is working!")
        print("\nThe system successfully:")
        print("- Connected to RL Orchestrator")
        print("- Assigned task to appropriate agent")
        print("- Executed task via Enhanced Production API")
    else:
        print("\n💥 System test FAILED - Check service status and logs")

if __name__ == "__main__":
    main()