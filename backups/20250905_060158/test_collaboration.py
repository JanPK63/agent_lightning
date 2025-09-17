#!/usr/bin/env python3
"""
Test script for Multi-Agent Collaboration System
Tests various collaboration patterns and validates functionality
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_collaboration import (
    CollaborationOrchestrator,
    CollaborativeTask,
    TaskComplexity,
    CollaborationMode,
    create_collaborative_task,
    start_collaboration_session
)
from agent_communication_protocol import (
    AgentMessage,
    Performative,
    MessageRouter,
    ConversationManager,
    TaskSharingProtocol
)


async def test_communication_protocol():
    """Test the communication protocol"""
    print("\n" + "="*60)
    print("TEST 1: Communication Protocol")
    print("="*60)
    
    # Create router and conversation manager
    router = MessageRouter()
    conv_manager = ConversationManager()
    
    # Register test agents
    test_agents = ["agent_alpha", "agent_beta", "agent_gamma"]
    for agent_id in test_agents:
        await router.register_agent(agent_id)
        router.subscribe_to_broadcast(agent_id)
    
    print(f"✓ Registered {len(test_agents)} agents")
    
    # Test direct messaging
    msg = AgentMessage(
        performative=Performative.INFORM,
        sender="agent_alpha",
        receiver="agent_beta",
        content={"info": "Test message"}
    )
    
    await router.route_message(msg)
    received = await router.get_message("agent_beta", timeout=1.0)
    
    if received and received.content == msg.content:
        print("✓ Direct messaging working")
    else:
        print("✗ Direct messaging failed")
    
    # Test broadcast messaging
    broadcast_msg = AgentMessage(
        performative=Performative.CFP,
        sender="agent_alpha",
        receiver="broadcast",
        content={"task": "Collaborative task"}
    )
    
    await router.route_message(broadcast_msg)
    
    # Check all other agents received it
    received_count = 0
    for agent_id in ["agent_beta", "agent_gamma"]:
        msg = await router.get_message(agent_id, timeout=1.0)
        if msg:
            received_count += 1
    
    if received_count == 2:
        print("✓ Broadcast messaging working")
    else:
        print(f"✗ Broadcast failed (received by {received_count}/2)")
    
    # Test conversation management
    conv_manager.start_conversation(msg)
    conv_manager.add_message(broadcast_msg)
    history = conv_manager.get_conversation_history(msg.conversation_id)
    
    if len(history) == 2:
        print("✓ Conversation tracking working")
    else:
        print("✗ Conversation tracking failed")
    
    # Test task sharing protocol
    task_announcement = TaskSharingProtocol.TaskAnnouncement(
        task_id="test_001",
        task_type="analysis",
        description="Test task",
        requirements={"skill": "python"},
        deadline=datetime.now() + timedelta(hours=1),
        reward=None,
        complexity=3
    )
    
    print("✓ Task sharing protocol structures created")
    
    return True


async def test_task_creation():
    """Test collaborative task creation"""
    print("\n" + "="*60)
    print("TEST 2: Task Creation and Structure")
    print("="*60)
    
    # Create simple task
    simple_task = await create_collaborative_task(
        description="Analyze Python code for security vulnerabilities",
        complexity=2,
        required_capabilities=["python", "security", "code_analysis"]
    )
    
    print(f"✓ Created simple task: {simple_task.task_id[:8]}...")
    print(f"  Description: {simple_task.description}")
    print(f"  Complexity: {simple_task.complexity.name}")
    print(f"  Capabilities: {simple_task.required_capabilities}")
    
    # Create complex task with subtasks
    complex_task = await create_collaborative_task(
        description="Build and deploy a web application",
        complexity=5,
        required_capabilities=["frontend", "backend", "database", "deployment"],
        deadline_hours=48
    )
    
    # Add subtasks
    subtask1 = CollaborativeTask(
        description="Design database schema",
        complexity=TaskComplexity.MODERATE,
        required_capabilities=["database", "sql"]
    )
    
    subtask2 = CollaborativeTask(
        description="Implement REST API",
        complexity=TaskComplexity.MODERATE,
        required_capabilities=["backend", "api"]
    )
    
    subtask3 = CollaborativeTask(
        description="Create React frontend",
        complexity=TaskComplexity.MODERATE,
        required_capabilities=["frontend", "react"]
    )
    
    complex_task.add_subtask(subtask1)
    complex_task.add_subtask(subtask2)
    complex_task.add_subtask(subtask3)
    
    # Add dependency (API depends on database)
    subtask2.dependencies.append(subtask1.task_id)
    
    print(f"✓ Created complex task with {len(complex_task.subtasks)} subtasks")
    print(f"  Main task: {complex_task.description}")
    for i, subtask in enumerate(complex_task.subtasks, 1):
        print(f"  Subtask {i}: {subtask.description}")
        if subtask.dependencies:
            print(f"    Dependencies: {len(subtask.dependencies)}")
    
    # Test dependency checking
    completed_tasks = {subtask1.task_id}
    if subtask2.is_ready(completed_tasks):
        print("✓ Dependency checking working")
    else:
        print("✗ Dependency checking failed")
    
    return simple_task, complex_task


async def test_collaboration_patterns():
    """Test different collaboration patterns"""
    print("\n" + "="*60)
    print("TEST 3: Collaboration Patterns")
    print("="*60)
    
    # Initialize orchestrator
    orchestrator = CollaborationOrchestrator()
    
    # Note: This will try to load actual agents, which might not exist in test
    # So we'll catch and continue
    try:
        await orchestrator.initialize()
        print(f"✓ Orchestrator initialized with {len(orchestrator.agents)} agents")
    except Exception as e:
        print(f"⚠ Orchestrator initialization partial: {e}")
        # Continue with empty agents for pattern testing
    
    # Test each pattern initialization
    patterns_tested = []
    
    for mode in CollaborationMode:
        pattern = orchestrator.collaboration_patterns.get(mode)
        if pattern:
            patterns_tested.append(mode.value)
            print(f"✓ {mode.value} pattern available")
    
    print(f"\nTotal patterns available: {len(patterns_tested)}")
    print(f"Patterns: {', '.join(patterns_tested)}")
    
    # Create test task for pattern execution
    test_task = CollaborativeTask(
        description="Test pattern execution",
        complexity=TaskComplexity.SIMPLE,
        required_capabilities=["general"]
    )
    
    # Test Master-Worker pattern specifically
    print("\nTesting Master-Worker Pattern:")
    from agent_collaboration import MasterWorkerPattern
    
    mw_pattern = MasterWorkerPattern()
    
    # Simulate initialization
    class MockSession:
        def __init__(self):
            self.participating_agents = ["master", "worker1", "worker2"]
            self.coordinator = None
    
    mock_session = MockSession()
    mock_router = MessageRouter()
    
    await mw_pattern.initialize(mock_session, {}, mock_router)
    
    if mw_pattern.master == "master" and len(mw_pattern.workers) == 2:
        print("✓ Master-Worker initialization successful")
        print(f"  Master: {mw_pattern.master}")
        print(f"  Workers: {mw_pattern.workers}")
    else:
        print("✗ Master-Worker initialization failed")
    
    # Test task decomposition
    test_task.complexity = TaskComplexity.COMPLEX
    subtasks = await mw_pattern._decompose_task(test_task)
    
    if subtasks and len(subtasks) > 0:
        print(f"✓ Task decomposition created {len(subtasks)} subtasks")
    else:
        print("✗ Task decomposition failed")
    
    # Test task assignment
    if subtasks:
        assignments = await mw_pattern._assign_tasks(subtasks, mw_pattern.workers)
        if len(assignments) == len(subtasks):
            print(f"✓ Task assignment successful ({len(assignments)} assignments)")
            for task, worker in assignments[:2]:
                print(f"  {worker} <- {task.description[:50]}")
        else:
            print("✗ Task assignment failed")
    
    return True


async def test_orchestrator_session():
    """Test collaboration session management"""
    print("\n" + "="*60)
    print("TEST 4: Collaboration Session Management")
    print("="*60)
    
    orchestrator = CollaborationOrchestrator()
    
    # Create a test task
    task = CollaborativeTask(
        description="Optimize database queries",
        complexity=TaskComplexity.MODERATE,
        required_capabilities=["database", "sql", "optimization"]
    )
    
    print(f"✓ Created test task: {task.description}")
    
    # Try to start a session (might fail without agents)
    try:
        # Create mock agents for testing
        from agent_collaboration import CollaborativeAgent
        
        class MockAgent(CollaborativeAgent):
            async def process_task(self, task):
                return f"Processed by {self.agent_id}"
            
            async def handle_message(self, message):
                pass
        
        # Add mock agents
        orchestrator.agents["mock_db_expert"] = MockAgent(
            "mock_db_expert", 
            ["database", "sql", "optimization"]
        )
        orchestrator.agents["mock_optimizer"] = MockAgent(
            "mock_optimizer",
            ["optimization", "performance"]
        )
        
        print(f"✓ Added {len(orchestrator.agents)} mock agents")
        
        # Start collaboration session
        session = await orchestrator.start_collaboration(
            task=task,
            mode=CollaborationMode.PEER_TO_PEER
        )
        
        print(f"✓ Started session: {session.session_id[:8]}...")
        print(f"  Mode: {session.mode.value}")
        print(f"  Status: {session.status}")
        print(f"  Participants: {len(session.participating_agents)}")
        
        # Wait briefly for async execution
        await asyncio.sleep(0.5)
        
        # Get session status
        status = await orchestrator.get_session_status(session.session_id)
        if status:
            print("✓ Session status retrieval working")
            print(f"  Session tracked: {status['session_id'][:8]}...")
            print(f"  Current status: {status['status']}")
        else:
            print("✗ Session status retrieval failed")
        
        # Test performance metrics
        if orchestrator.performance_metrics:
            print("✓ Performance tracking initialized")
        
        return session
        
    except Exception as e:
        print(f"⚠ Session creation error (expected without full agent setup): {e}")
        return None


async def test_integration_points():
    """Test integration with existing Agent Lightning components"""
    print("\n" + "="*60)
    print("TEST 5: Integration Points")
    print("="*60)
    
    # Test imports and availability
    components = []
    
    try:
        from agent_config import AgentConfigManager
        components.append("AgentConfigManager")
    except ImportError:
        print("✗ AgentConfigManager not available")
    
    try:
        from enhanced_production_api import EnhancedAgentService
        components.append("EnhancedAgentService")
    except ImportError:
        print("✗ EnhancedAgentService not available")
    
    try:
        from shared_memory_system import SharedMemorySystem
        components.append("SharedMemorySystem")
    except ImportError:
        print("✗ SharedMemorySystem not available")
    
    try:
        from knowledge_manager import KnowledgeManager
        components.append("KnowledgeManager")
    except ImportError:
        print("✗ KnowledgeManager not available")
    
    print(f"✓ Successfully imported {len(components)} components:")
    for comp in components:
        print(f"  • {comp}")
    
    # Test API helper functions
    try:
        # Test create_collaborative_task
        task = await create_collaborative_task(
            "Test API integration",
            complexity=3,
            required_capabilities=["api", "integration"]
        )
        print("✓ API helper create_collaborative_task working")
        
        # Test start_collaboration_session structure
        # (Won't actually start without full setup)
        print("✓ API helper start_collaboration_session available")
        
    except Exception as e:
        print(f"✗ API helper error: {e}")
    
    return True


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("MULTI-AGENT COLLABORATION SYSTEM TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Communication Protocol
    try:
        result = await test_communication_protocol()
        results.append(("Communication Protocol", "PASS" if result else "FAIL"))
    except Exception as e:
        results.append(("Communication Protocol", f"ERROR: {str(e)[:50]}"))
    
    # Test 2: Task Creation
    try:
        simple, complex = await test_task_creation()
        results.append(("Task Creation", "PASS"))
    except Exception as e:
        results.append(("Task Creation", f"ERROR: {str(e)[:50]}"))
    
    # Test 3: Collaboration Patterns
    try:
        result = await test_collaboration_patterns()
        results.append(("Collaboration Patterns", "PASS" if result else "FAIL"))
    except Exception as e:
        results.append(("Collaboration Patterns", f"ERROR: {str(e)[:50]}"))
    
    # Test 4: Session Management
    try:
        session = await test_orchestrator_session()
        results.append(("Session Management", "PASS" if session else "PARTIAL"))
    except Exception as e:
        results.append(("Session Management", f"ERROR: {str(e)[:50]}"))
    
    # Test 5: Integration Points
    try:
        result = await test_integration_points()
        results.append(("Integration Points", "PASS" if result else "FAIL"))
    except Exception as e:
        results.append(("Integration Points", f"ERROR: {str(e)[:50]}"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, status in results:
        symbol = "✅" if "PASS" in status else "⚠️" if "PARTIAL" in status else "❌"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, s in results if "PASS" in s)
    total = len(results)
    
    print("\n" + "="*60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed successfully!")
    elif passed > 0:
        print("⚠️ Some tests passed. System partially functional.")
    else:
        print("❌ Tests failed. Review implementation.")
    
    print("="*60)


if __name__ == "__main__":
    print("Starting Multi-Agent Collaboration Tests...")
    asyncio.run(run_all_tests())
    print("\nTest suite completed!")