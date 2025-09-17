#!/usr/bin/env python3
"""
Test Fixed Agents - Demonstrate that agents now actually work
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any

class AgentTester:
    """Test the fixed agent system"""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url
    
    def test_agent_list(self):
        """Test listing available agents"""
        print("\\n🤖 Testing Agent List...")
        try:
            response = requests.get(f"{self.base_url}/agents")
            if response.status_code == 200:
                agents = response.json()
                print(f"✅ Found {agents['count']} agents:")
                for agent in agents['agents']:
                    print(f"   • {agent['id']}: {agent['name']} ({agent['model']})")
                return True
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_task_execution(self, task: str, agent_id: str = "auto"):
        """Test actual task execution"""
        print(f"\\n⚡ Testing Task Execution with {agent_id}...")
        print(f"Task: {task}")
        
        try:
            payload = {
                "task": task,
                "agent_id": agent_id,
                "context": {"test": True}
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/execute", json=payload)
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Task completed in {execution_time:.2f}s")
                print(f"Agent: {result.get('agent_name', 'Unknown')}")
                print(f"Status: {result.get('status', 'Unknown')}")
                
                # Show result preview
                result_text = result.get('result', '')
                if len(result_text) > 200:
                    print(f"Result: {result_text[:200]}...")
                else:
                    print(f"Result: {result_text}")
                
                return True
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_chat(self, agent_id: str, message: str):
        """Test chat functionality"""
        print(f"\\n💬 Testing Chat with {agent_id}...")
        print(f"Message: {message}")
        
        try:
            payload = {"message": message}
            response = requests.post(f"{self.base_url}/chat/{agent_id}", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Chat successful")
                print(f"Agent Response: {result.get('agent_response', '')[:200]}...")
                return True
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("\\n" + "="*60)
        print("🧪 COMPREHENSIVE AGENT TEST SUITE")
        print("="*60)
        
        # Test 1: List agents
        success1 = self.test_agent_list()
        
        # Test 2: Execute various tasks
        test_tasks = [
            ("Create a simple Python function to calculate fibonacci numbers", "full_stack_developer"),
            ("Analyze this dataset and provide insights", "data_scientist"),
            ("Review this code for security vulnerabilities: def login(user, pwd): return user == 'admin'", "security_expert"),
            ("Design a microservices architecture for an e-commerce platform", "system_architect"),
            ("Create a Docker deployment configuration", "devops_engineer")
        ]
        
        success2 = True
        for task, agent in test_tasks:
            if not self.test_task_execution(task, agent):
                success2 = False
        
        # Test 3: Auto agent selection
        success3 = self.test_task_execution("Build a REST API for user management", "auto")
        
        # Test 4: Chat functionality
        success4 = self.test_chat("full_stack_developer", "What's the best way to handle authentication in a web app?")
        
        # Summary
        print("\\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Agent List: {'✅ PASS' if success1 else '❌ FAIL'}")
        print(f"Task Execution: {'✅ PASS' if success2 else '❌ FAIL'}")
        print(f"Auto Selection: {'✅ PASS' if success3 else '❌ FAIL'}")
        print(f"Chat Function: {'✅ PASS' if success4 else '❌ FAIL'}")
        
        overall_success = success1 and success2 and success3 and success4
        print(f"\\nOverall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if overall_success:
            print("\\n🎉 Congratulations! Your agents are now working properly!")
            print("They will actually execute tasks instead of just describing them.")
        else:
            print("\\n⚠️ Some issues remain. Check the error messages above.")
        
        return overall_success


def main():
    """Main test function"""
    print("\\n" + "="*60)
    print("⚡ AGENT LIGHTNING - FIXED SYSTEM TEST")
    print("="*60)
    print("\\nThis will test if your agents are now actually working...")
    print("\\n🔧 Make sure the fixed API is running on port 8888:")
    print("   python fixed_agent_api.py")
    print("\\n" + "-"*60)
    
    # Wait for user confirmation
    input("Press Enter when the API is running...")
    
    # Run tests
    tester = AgentTester()
    tester.run_comprehensive_test()
    
    print("\\n" + "="*60)
    print("🚀 NEXT STEPS")
    print("="*60)
    print("\\n1. If tests passed, your agents are now working!")
    print("2. Use the /execute endpoint to assign real tasks")
    print("3. Agents will provide actual implementations, not descriptions")
    print("4. You can integrate this with your existing system")
    print("\\n💡 Example usage:")
    print('   curl -X POST http://localhost:8888/execute \\\\')
    print('        -H "Content-Type: application/json" \\\\')
    print('        -d \'{"task": "Create a login system", "agent_id": "full_stack_developer"}\'')


if __name__ == "__main__":
    main()