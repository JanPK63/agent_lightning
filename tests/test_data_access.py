#!/usr/bin/env python3
"""
Test script for Data Access Layer
Verifies database, cache, and event integration
"""

import sys
import os
import time
import uuid
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set Redis password
os.environ['REDIS_PASSWORD'] = 'redis_secure_password_123'

from shared.data_access import DataAccessLayer
from shared.events import EventChannel

def test_agent_operations(dal: DataAccessLayer):
    """Test agent CRUD operations"""
    print("\n🧪 Testing Agent Operations...")
    
    # Create test agent
    agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
    agent_data = {
        "id": agent_id,
        "name": "Test Agent",
        "model": "test-model",
        "specialization": "testing",
        "status": "idle"
    }
    
    # Create agent
    created_agent = dal.create_agent(agent_data)
    assert created_agent['id'] == agent_id
    print(f"✅ Agent created: {agent_id}")
    
    # Get agent (should hit database)
    retrieved = dal.get_agent(agent_id)
    assert retrieved['name'] == "Test Agent"
    print("✅ Agent retrieved from database")
    
    # Get agent again (should hit cache)
    cached = dal.get_agent(agent_id)
    assert cached['id'] == agent_id
    print("✅ Agent retrieved from cache")
    
    # List agents
    agents = dal.list_agents()
    assert any(a['id'] == agent_id for a in agents)
    print(f"✅ Listed {len(agents)} agents")
    
    # Update agent
    updated = dal.update_agent(agent_id, {"status": "busy"})
    assert updated['status'] == "busy"
    print("✅ Agent updated")
    
    # Delete agent
    deleted = dal.delete_agent(agent_id)
    assert deleted == True
    print("✅ Agent deleted")
    
    # Verify deletion
    not_found = dal.get_agent(agent_id)
    assert not_found is None
    print("✅ Deletion verified")

def test_task_operations(dal: DataAccessLayer):
    """Test task operations"""
    print("\n🧪 Testing Task Operations...")
    
    # Create task
    task_data = {
        "agent_id": "test_agent",
        "description": "Test task",
        "priority": "high",
        "context": {"test": True}
    }
    
    task = dal.create_task(task_data)
    task_id = task['id']
    assert task['status'] == "pending"
    print(f"✅ Task created: {task_id}")
    
    # Get task
    retrieved = dal.get_task(task_id)
    assert retrieved['description'] == "Test task"
    print("✅ Task retrieved")
    
    # Update task status - started
    updated = dal.update_task_status(task_id, "started")
    assert updated['status'] == "started"
    assert updated['started_at'] is not None
    print("✅ Task started")
    
    # Update task status - completed
    result = {"output": "Task completed successfully"}
    completed = dal.update_task_status(task_id, "completed", result=result)
    assert completed['status'] == "completed"
    assert completed['result'] == result
    print("✅ Task completed")
    
    # List tasks
    tasks = dal.list_tasks(agent_id="test_agent")
    assert any(t['id'] == task_id for t in tasks)
    print(f"✅ Listed {len(tasks)} tasks")

def test_knowledge_operations(dal: DataAccessLayer):
    """Test knowledge management"""
    print("\n🧪 Testing Knowledge Operations...")
    
    agent_id = "test_agent"
    
    # Add knowledge
    knowledge_data = {
        "category": "test_category",
        "content": "Important test knowledge",
        "source": "test_source",
        "metadata": {"importance": "high"}
    }
    
    knowledge = dal.add_knowledge(agent_id, knowledge_data)
    knowledge_id = knowledge['id']
    print(f"✅ Knowledge added: {knowledge_id}")
    
    # Get agent knowledge
    all_knowledge = dal.get_agent_knowledge(agent_id)
    assert any(k['id'] == knowledge_id for k in all_knowledge)
    print(f"✅ Retrieved {len(all_knowledge)} knowledge items")
    
    # Get filtered knowledge
    filtered = dal.get_agent_knowledge(agent_id, category="test_category")
    assert any(k['id'] == knowledge_id for k in filtered)
    print("✅ Filtered knowledge retrieved")
    
    # Update usage
    updated = dal.update_knowledge_usage(knowledge_id)
    assert updated['usage_count'] > 0
    print("✅ Knowledge usage updated")

def test_workflow_operations(dal: DataAccessLayer):
    """Test workflow operations"""
    print("\n🧪 Testing Workflow Operations...")
    
    # Create workflow
    workflow_data = {
        "name": "Test Workflow",
        "description": "Test workflow description",
        "steps": [
            {"step": 1, "action": "initialize"},
            {"step": 2, "action": "process"},
            {"step": 3, "action": "complete"}
        ],
        "status": "pending",
        "created_by": "test_user"
    }
    
    workflow = dal.create_workflow(workflow_data)
    workflow_id = workflow['id']
    print(f"✅ Workflow created: {workflow_id}")
    
    # Update workflow status
    updated = dal.update_workflow_status(workflow_id, "running", step=1)
    assert updated['status'] == "running"
    print("✅ Workflow started")
    
    # Progress through steps
    for step in [2, 3]:
        updated = dal.update_workflow_status(workflow_id, "running", step=step)
        assert updated['context']['current_step'] == step
        print(f"✅ Workflow step {step} completed")
    
    # Complete workflow
    completed = dal.update_workflow_status(workflow_id, "completed")
    assert completed['status'] == "completed"
    print("✅ Workflow completed")

def test_session_operations(dal: DataAccessLayer):
    """Test session management"""
    print("\n🧪 Testing Session Operations...")
    
    user_id = "test_user"
    token = f"test_token_{uuid.uuid4().hex}"
    
    # Create session
    session = dal.create_session(user_id, token, {"role": "admin"})
    assert session['token'] == token
    print("✅ Session created")
    
    # Get session
    retrieved = dal.get_session(token)
    assert retrieved['user_id'] == user_id
    print("✅ Session retrieved")
    
    # Delete session
    deleted = dal.delete_session(token)
    assert deleted == True
    print("✅ Session deleted")
    
    # Verify deletion
    not_found = dal.get_session(token)
    assert not_found is None
    print("✅ Session deletion verified")

def test_metrics_operations(dal: DataAccessLayer):
    """Test metrics recording"""
    print("\n🧪 Testing Metrics Operations...")
    
    # Record metrics
    dal.record_metric("test_latency", 42.5, {"endpoint": "/test"})
    dal.record_metric("test_throughput", 1000, {"endpoint": "/test"})
    dal.record_metric("test_errors", 2, {"type": "validation"})
    print("✅ Metrics recorded")
    
    # Get recent metrics
    metrics = dal.get_metrics(minutes=5)
    assert len(metrics) >= 3
    print(f"✅ Retrieved {len(metrics)} recent metrics")
    
    # Get specific metric
    latency_metrics = dal.get_metrics(metric_name="test_latency", minutes=5)
    assert any(m['value'] == 42.5 for m in latency_metrics)
    print("✅ Filtered metrics retrieved")

def test_event_propagation(dal: DataAccessLayer):
    """Test event emission and handling"""
    print("\n🧪 Testing Event Propagation...")
    
    events_received = []
    
    def event_handler(event):
        events_received.append(event)
        print(f"  📨 Received event: {event.channel}")
    
    # Register event handlers
    dal.event_bus.on(EventChannel.AGENT_CREATED, event_handler)
    dal.event_bus.on(EventChannel.TASK_COMPLETED, event_handler)
    
    # Create agent (should emit event)
    agent_id = f"event_test_{uuid.uuid4().hex[:8]}"
    dal.create_agent({
        "id": agent_id,
        "name": "Event Test Agent",
        "model": "test-model"
    })
    
    # Create and complete task (should emit events)
    task = dal.create_task({
        "agent_id": agent_id,
        "description": "Event test task"
    })
    dal.update_task_status(task['id'], "completed", result={"done": True})
    
    # Wait for events
    time.sleep(2)
    
    # Verify events received
    assert len(events_received) >= 2
    print(f"✅ Received {len(events_received)} events")
    
    # Cleanup
    dal.delete_agent(agent_id)

def test_distributed_transaction(dal: DataAccessLayer):
    """Test distributed transaction management"""
    print("\n🧪 Testing Distributed Transactions...")
    
    try:
        with dal.distributed_transaction() as tx_id:
            print(f"  🔒 Transaction {tx_id} started")
            
            # Perform operations within transaction
            agent_id = f"tx_test_{uuid.uuid4().hex[:8]}"
            dal.create_agent({
                "id": agent_id,
                "name": "Transaction Test",
                "model": "test-model"
            })
            
            # Simulate some work
            time.sleep(0.5)
            
            print(f"  ✅ Transaction {tx_id} completed")
            
            # Cleanup
            dal.delete_agent(agent_id)
            
    except Exception as e:
        print(f"  ❌ Transaction failed: {e}")
        raise

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 Data Access Layer Test Suite")
    print("=" * 60)
    
    try:
        # Initialize DAL
        dal = DataAccessLayer("test_service")
        
        # Check health
        health = dal.health_check()
        print(f"\n📊 Health Check: {health}")
        
        if not health['database'] or not health['cache']:
            print("❌ Required services not available")
            print("Please ensure PostgreSQL and Redis are running")
            return 1
        
        # Run tests
        test_agent_operations(dal)
        test_task_operations(dal)
        test_knowledge_operations(dal)
        test_workflow_operations(dal)
        test_session_operations(dal)
        test_metrics_operations(dal)
        test_event_propagation(dal)
        test_distributed_transaction(dal)
        
        print("\n" + "=" * 60)
        print("✨ All tests passed successfully!")
        print("=" * 60)
        
        # Cleanup
        dal.cleanup()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())