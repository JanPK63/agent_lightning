#!/usr/bin/env python3
"""
Test the Improved RL Orchestrator to verify it prevents wrong agent assignments
This tests the exact scenario that failed: security_expert getting a web task
"""

import requests
import json
import sys

def test_task_assignment():
    """Test task assignment with the problematic case"""
    
    # The exact task that was wrongly assigned to security_expert
    test_task = {
        "task_id": "330d0577-f8e2-46f1-87d9-86916f2872e3",
        "description": "create a hello world website in this directory /Users/jankootstra/Identity_blockchain/agent-backbone-architecture/docs",
        "priority": 5
    }
    
    print("=" * 60)
    print("Testing Improved RL Orchestrator - Agent Assignment")
    print("=" * 60)
    
    # Test improved orchestrator
    improved_url = "http://localhost:8025/assign-task"
    
    print(f"\nTask: {test_task['description']}")
    print("-" * 60)
    
    try:
        response = requests.post(improved_url, json=test_task)
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ IMPROVED ORCHESTRATOR RESULT:")
            print(f"  Assigned Agent: {result['assigned_agent']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Reason: {result['reason']}")
            print(f"  Validation Passed: {result['validation_passed']}")
            
            if result['warnings']:
                print(f"\n⚠️  Warnings:")
                for warning in result['warnings']:
                    print(f"    - {warning}")
            
            # Check if the bug was prevented
            if result['assigned_agent'] != "security_expert":
                print(f"\n🎉 SUCCESS: Bug prevented! Task correctly assigned to {result['assigned_agent']}")
                print("   (Previously this was wrongly assigned to security_expert)")
            else:
                print("\n❌ FAILURE: Task still assigned to security_expert!")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        
    # Test more cases
    print("\n" + "=" * 60)
    print("Testing Additional Cases")
    print("=" * 60)
    
    test_cases = [
        {
            "task_id": "test-001",
            "description": "perform security audit on the authentication system",
            "expected": "security_expert"
        },
        {
            "task_id": "test-002", 
            "description": "analyze user engagement data for the last quarter",
            "expected": "data_analyst"
        },
        {
            "task_id": "test-003",
            "description": "write unit tests for the payment module",
            "expected": "tester-agent"
        },
        {
            "task_id": "test-004",
            "description": "deploy the application to kubernetes cluster",
            "expected": "devops_engineer"
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['description'][:50]}...")
        print(f"Expected: {test['expected']}")
        
        try:
            response = requests.post(improved_url, json={
                "task_id": test["task_id"],
                "description": test["description"],
                "priority": 5
            })
            
            if response.status_code == 200:
                result = response.json()
                assigned = result['assigned_agent']
                confidence = result['confidence']
                
                if assigned == test['expected']:
                    print(f"✅ Correct: {assigned} (confidence: {confidence:.2f})")
                else:
                    print(f"⚠️  Different: {assigned} (expected {test['expected']}, confidence: {confidence:.2f})")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Check Q-table status
    print("\n" + "=" * 60)
    print("Q-Table Status")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:8025/q-table")
        if response.status_code == 200:
            q_data = response.json()
            print(f"States learned: {len(q_data['states'])}")
            print(f"Total Q-values: {q_data['total_entries']}")
            
            if q_data['states']:
                print("\nQ-Table entries:")
                for state, actions in q_data['q_table'].items():
                    print(f"  {state}:")
                    for agent, q_value in actions.items():
                        print(f"    - {agent}: {q_value:.3f}")
        
    except Exception as e:
        print(f"Could not get Q-table: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_task_assignment()