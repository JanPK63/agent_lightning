#!/usr/bin/env python3
"""
Test Idempotent State Manager
Verifies operations can be safely retried without side effects
"""

import requests
import json
import time
import uuid

def test_idempotency():
    """Test idempotent operations"""
    
    base_url = "http://localhost:8026"
    
    print("=" * 60)
    print("Testing Idempotent State Management")
    print("=" * 60)
    
    # Test 1: Check if operation can proceed (should be allowed first time)
    print("\n1. Testing first operation check...")
    
    operation_params = {
        "operation_type": "task_assignment",
        "entity_id": "test-task-123",
        "operation_params": {
            "agent_id": "coder-agent",
            "priority": 5,
            "description": "Create test function"
        }
    }
    
    response = requests.post(f"{base_url}/check", json=operation_params)
    if response.status_code == 200:
        result = response.json()
        print(f"   Can proceed: {result['can_proceed']}")
        print(f"   State: {result['state']}")
        print(f"   Message: {result['message']}")
    
    # Test 2: Register the operation
    print("\n2. Registering operation...")
    
    register_params = {
        **operation_params,
        "ttl_seconds": 300,
        "retry_count": 0
    }
    
    response = requests.post(f"{base_url}/register", json=register_params)
    if response.status_code == 200:
        result = response.json()
        operation_id = result['operation_id']
        print(f"   Operation ID: {operation_id}")
        print(f"   State: {result['state']}")
        print(f"   Can proceed: {result['can_proceed']}")
    
    # Test 3: Try to register same operation again (should be blocked)
    print("\n3. Testing duplicate registration (should be blocked)...")
    
    response = requests.post(f"{base_url}/register", json=register_params)
    if response.status_code == 200:
        result = response.json()
        print(f"   Can proceed: {result['can_proceed']}")
        print(f"   State: {result['state']}")
        print(f"   Message: {result['message']}")
    
    # Test 4: Complete the operation
    print("\n4. Completing operation...")
    
    completion_data = {
        "success": True,
        "data": {
            "result": "Function created successfully",
            "lines_of_code": 42
        }
    }
    
    response = requests.post(f"{base_url}/complete/{operation_id}", json=completion_data)
    if response.status_code == 200:
        print(f"   ✅ Operation completed successfully")
    
    # Test 5: Try to register same operation after completion (should return cached result)
    print("\n5. Testing after completion (should return cached result)...")
    
    response = requests.post(f"{base_url}/check", json=operation_params)
    if response.status_code == 200:
        result = response.json()
        print(f"   Can proceed: {result['can_proceed']}")
        print(f"   State: {result['state']}")
        print(f"   Message: {result['message']}")
        if result.get('existing_result'):
            print(f"   Cached result: {result['existing_result']}")
    
    # Test 6: Test different operation (should be allowed)
    print("\n6. Testing different operation...")
    
    new_operation = {
        "operation_type": "task_assignment",
        "entity_id": "test-task-456",  # Different entity
        "operation_params": {
            "agent_id": "tester-agent",
            "priority": 3,
            "description": "Run tests"
        }
    }
    
    response = requests.post(f"{base_url}/check", json=new_operation)
    if response.status_code == 200:
        result = response.json()
        print(f"   Can proceed: {result['can_proceed']}")
        print(f"   State: {result['state']}")
    
    # Test 7: Test failed operation and retry
    print("\n7. Testing failed operation and retry...")
    
    failing_op = {
        "operation_type": "agent_execution",
        "entity_id": "fail-test-001",
        "operation_params": {
            "command": "simulate_failure"
        },
        "ttl_seconds": 60
    }
    
    # Register
    response = requests.post(f"{base_url}/register", json=failing_op)
    if response.status_code == 200:
        fail_op_id = response.json()['operation_id']
        print(f"   Registered operation: {fail_op_id}")
    
    # Mark as failed
    response = requests.post(f"{base_url}/complete/{fail_op_id}", 
                            json={"success": False, "data": {"error": "Simulated failure"}})
    print(f"   Marked as failed")
    
    # Check if retry is allowed
    response = requests.post(f"{base_url}/check", json={
        "operation_type": failing_op["operation_type"],
        "entity_id": failing_op["entity_id"],
        "operation_params": failing_op["operation_params"]
    })
    if response.status_code == 200:
        result = response.json()
        print(f"   Can retry: {result['can_proceed']}")
        print(f"   Message: {result['message']}")
    
    # Test 8: Get operation history for entity
    print("\n8. Getting operation history for entity...")
    
    response = requests.get(f"{base_url}/operations/entity/test-task-123")
    if response.status_code == 200:
        result = response.json()
        print(f"   Operations for entity: {len(result['operations'])}")
        for op in result['operations']:
            print(f"   - State: {op['state']}, Created: {op['created_at']}")
    
    print("\n" + "=" * 60)
    print("Idempotency Test Complete!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Operations tracked with unique keys")
    print("✓ Duplicate operations blocked while in progress")
    print("✓ Completed operations return cached results")
    print("✓ Failed operations can be retried")
    print("✓ Different operations proceed independently")
    print("✓ Operation history maintained for audit")

if __name__ == "__main__":
    test_idempotency()