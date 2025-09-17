#!/usr/bin/env python3
"""
Comprehensive Test for Multi-Agent Collaboration System
Tests the complete workflow of task creation, decomposition, role assignment,
execution, and result aggregation.
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_collaboration import (
    CollaborationOrchestrator,
    CollaborativeTask,
    TaskComplexity,
    CollaborationMode,
    create_collaborative_task,
    AgentRole
)
from agent_communication_protocol import AgentMessage, Performative
from agent_role_assignment import RoleAssigner, RoleAssignmentStrategy
from agent_message_queue import MessageQueueManager, QueueType
from task_decomposition import TaskDecomposer, DecompositionStrategy
from agent_coordination_state_machine import CoordinationStateMachine, StateContext
from collaboration_result_aggregator import (
    CollaborationResultAggregator,
    AgentResult,
    ResultType,
    AggregationStrategy
)


class MultiAgentCollaborationTest:
    """Test suite for multi-agent collaboration"""
    
    def __init__(self):
        self.orchestrator = CollaborationOrchestrator()
        self.role_assigner = RoleAssigner()
        self.queue_manager = MessageQueueManager()
        self.task_decomposer = TaskDecomposer()
        self.result_aggregator = CollaborationResultAggregator()
        self.test_results = []
    
    async def setup(self):
        """Initialize the test environment"""
        print("\n🔧 Setting up test environment...")
        
        # Initialize orchestrator
        await self.orchestrator.initialize()
        
        # Create message queues
        self.queue_manager.create_queue("test_coordination", QueueType.PRIORITY)
        self.queue_manager.create_queue("test_tasks", QueueType.WORK_STEALING)
        self.queue_manager.create_queue("test_results", QueueType.FIFO)
        
        print("✅ Test environment ready")
    
    async def test_simple_task_collaboration(self):
        """Test 1: Simple task with 2 agents"""
        print("\n" + "="*60)
        print("Test 1: Simple Task Collaboration")
        print("="*60)
        
        # Create a simple task
        task = await create_collaborative_task(
            "Write a Python function to calculate Fibonacci numbers",
            complexity=2,  # Simple
            required_capabilities=["can_write_code", "can_test"]
        )
        
        print(f"\n📋 Task created: {task.task_id}")
        print(f"   Description: {task.description}")
        print(f"   Complexity: {task.complexity.name}")
        
        # Start collaboration
        session = await self.orchestrator.start_collaboration(
            task=task,
            mode=CollaborationMode.PEER_TO_PEER
        )
        
        print(f"\n🚀 Session started: {session.session_id}")
        print(f"   Mode: {session.mode}")
        print(f"   Agents: {session.participating_agents}")
        
        # Wait for completion
        await asyncio.sleep(2)
        
        # Get status
        status = await self.orchestrator.get_session_status(session.session_id)
        print(f"\n📊 Session status: {status['status']}")
        
        self.test_results.append({
            "test": "simple_task",
            "success": status['status'] == 'completed',
            "duration": 2
        })
        
        return session
    
    async def test_complex_task_decomposition(self):
        """Test 2: Complex task with decomposition"""
        print("\n" + "="*60)
        print("Test 2: Complex Task with Decomposition")
        print("="*60)
        
        # Create a complex task
        task = await create_collaborative_task(
            "Build a complete REST API with authentication, database models, and testing",
            complexity=4,  # Complex
            required_capabilities=[
                "can_write_code",
                "can_design_architecture",
                "can_test",
                "can_write_documentation"
            ]
        )
        
        print(f"\n📋 Complex task created: {task.task_id}")
        
        # Decompose the task
        decomposed = self.task_decomposer.decompose_task(
            task,
            strategy=DecompositionStrategy.FUNCTIONAL,
            max_depth=2,
            max_subtasks=5
        )
        
        print(f"\n🔍 Task decomposed into {len(decomposed.subtasks)} subtasks:")
        for i, subtask in enumerate(decomposed.subtasks, 1):
            print(f"   {i}. {subtask.description}")
        
        # Start collaboration with Master-Worker mode
        session = await self.orchestrator.start_collaboration(
            task=decomposed,
            mode=CollaborationMode.MASTER_WORKER
        )
        
        print(f"\n🚀 Session started with Master-Worker pattern")
        print(f"   Session ID: {session.session_id}")
        print(f"   Participating agents: {session.participating_agents}")
        
        # Simulate work
        await asyncio.sleep(3)
        
        self.test_results.append({
            "test": "complex_decomposition",
            "success": len(decomposed.subtasks) > 0,
            "subtasks": len(decomposed.subtasks)
        })
        
        return session, decomposed
    
    async def test_role_assignment(self):
        """Test 3: Role assignment with different strategies"""
        print("\n" + "="*60)
        print("Test 3: Role Assignment Strategies")
        print("="*60)
        
        # Create task requiring specific roles
        task = await create_collaborative_task(
            "Review and optimize database queries for performance",
            complexity=3,
            required_capabilities=["can_optimize", "can_review_code", "can_analyze_data"]
        )
        
        # Get available agents
        from agent_collaboration import SpecializedCollaborativeAgent
        from agent_config import AgentConfigManager
        from enhanced_production_api import EnhancedAgentService
        
        config_manager = AgentConfigManager()
        agent_service = EnhancedAgentService()
        
        agents = []
        agent_names = ["database_specialist", "system_architect", "full_stack_developer"]
        
        for name in agent_names:
            config = config_manager.get_agent(name)
            if config:
                agent = SpecializedCollaborativeAgent(
                    agent_id=name,
                    config=config,
                    agent_service=agent_service
                )
                agents.append(agent)
        
        # Test different assignment strategies
        strategies = [
            RoleAssignmentStrategy.CAPABILITY_BASED,
            RoleAssignmentStrategy.LOAD_BALANCED,
            RoleAssignmentStrategy.SPECIALIZATION
        ]
        
        for strategy in strategies:
            print(f"\n🎯 Testing {strategy.name} strategy:")
            assignments = self.role_assigner.assign_roles(task, agents, strategy)
            
            for agent_id, role in assignments.items():
                print(f"   {agent_id}: {role.value}")
        
        self.test_results.append({
            "test": "role_assignment",
            "success": len(assignments) > 0,
            "agents_assigned": len(assignments)
        })
        
        return assignments
    
    async def test_message_queue_communication(self):
        """Test 4: Inter-agent message queue communication"""
        print("\n" + "="*60)
        print("Test 4: Message Queue Communication")
        print("="*60)
        
        # Subscribe agents to queues
        agents = ["agent_1", "agent_2", "agent_3"]
        for agent in agents:
            self.queue_manager.subscribe(agent, "test_coordination")
        
        print(f"\n📬 Subscribed {len(agents)} agents to coordination queue")
        
        # Send broadcast message
        broadcast_msg = AgentMessage(
            performative=Performative.INFORM,
            sender="coordinator",
            receiver="broadcast",
            content={"message": "Task starting", "priority": "high"}
        )
        
        success = await self.queue_manager.send_message(broadcast_msg, "test_coordination")
        print(f"\n📤 Broadcast message sent: {success}")
        
        # Send targeted messages
        messages_sent = 0
        for i, agent in enumerate(agents):
            msg = AgentMessage(
                performative=Performative.REQUEST,
                sender="coordinator",
                receiver=agent,
                content={"task_id": f"task_{i+1}", "action": "execute"}
            )
            
            if await self.queue_manager.send_message(msg, "test_coordination"):
                messages_sent += 1
        
        print(f"📤 Sent {messages_sent} targeted messages")
        
        # Test message retrieval
        messages_received = 0
        for agent in agents:
            msg = await self.queue_manager.receive_message(agent, "test_coordination", timeout=0.5)
            if msg:
                messages_received += 1
                print(f"📥 {agent} received: {msg.content}")
        
        print(f"\n✉️ Message delivery rate: {messages_received}/{len(agents)}")
        
        self.test_results.append({
            "test": "message_queue",
            "success": messages_received == len(agents),
            "delivery_rate": f"{messages_received}/{len(agents)}"
        })
    
    async def test_state_machine_workflow(self):
        """Test 5: Complete state machine workflow"""
        print("\n" + "="*60)
        print("Test 5: State Machine Workflow")
        print("="*60)
        
        # Create task
        task = CollaborativeTask(
            description="Implement caching system with Redis",
            complexity=TaskComplexity.MODERATE,
            required_capabilities=["can_write_code", "can_design_architecture"]
        )
        
        # Create state context
        context = StateContext(
            session_id="test_state_session",
            task=task,
            agents=["system_architect", "database_specialist"],
            mode=CollaborationMode.PEER_TO_PEER
        )
        
        # Create and start state machine
        state_machine = CoordinationStateMachine()
        
        print(f"\n🎯 Starting state machine")
        print(f"   Initial state: {context.current_state.name}")
        
        # Run state machine with timeout
        try:
            await asyncio.wait_for(
                state_machine.start(context),
                timeout=10
            )
        except asyncio.TimeoutError:
            print("   State machine running (timeout expected for demo)")
        
        print(f"\n📊 State transitions:")
        for state, timestamp in context.state_history[:10]:  # Show first 10
            print(f"   {state.name} at {timestamp.strftime('%H:%M:%S')}")
        
        self.test_results.append({
            "test": "state_machine",
            "success": len(context.state_history) > 5,
            "transitions": len(context.state_history)
        })
    
    async def test_result_aggregation(self):
        """Test 6: Result aggregation from multiple agents"""
        print("\n" + "="*60)
        print("Test 6: Result Aggregation")
        print("="*60)
        
        # Create sample results from different agents
        results = [
            AgentResult(
                agent_id="code_writer",
                role=AgentRole.WORKER,
                task_id="task_001",
                result_type=ResultType.CODE,
                content={
                    "code": "def cache_get(key):\n    return redis_client.get(key)",
                    "language": "python"
                },
                confidence=0.9
            ),
            AgentResult(
                agent_id="architect",
                role=AgentRole.SPECIALIST,
                task_id="task_001",
                result_type=ResultType.ANALYSIS,
                content="Cache implementation should use connection pooling for efficiency",
                confidence=0.95
            ),
            AgentResult(
                agent_id="tester",
                role=AgentRole.REVIEWER,
                task_id="task_001",
                result_type=ResultType.CODE,
                content={
                    "code": "def cache_get(key):\n    # Added error handling\n    try:\n        return redis_client.get(key)\n    except Exception as e:\n        logger.error(f'Cache error: {e}')\n        return None",
                    "language": "python"
                },
                confidence=0.85
            )
        ]
        
        print(f"\n📊 Aggregating {len(results)} results from agents:")
        for r in results:
            print(f"   {r.agent_id} ({r.role.value}): {r.result_type.value}")
        
        # Aggregate results
        aggregated = await self.result_aggregator.aggregate_results(
            results,
            strategy=AggregationStrategy.WEIGHTED
        )
        
        if aggregated:
            print(f"\n✅ Aggregation complete:")
            print(f"   Strategy: {aggregated.aggregation_strategy.name}")
            print(f"   Confidence: {aggregated.confidence_score:.2%}")
            print(f"   Consensus: {aggregated.consensus_level:.2%}")
            print(f"   Conflicts resolved: {aggregated.conflicts_resolved}")
            
            # Generate consensus report
            report = self.result_aggregator.get_consensus_report(aggregated)
            print(f"\n📈 Consensus Report:")
            print(f"   {report['consensus_analysis']['interpretation']}")
            print(f"   {report['confidence_analysis']['interpretation']}")
        
        self.test_results.append({
            "test": "result_aggregation",
            "success": aggregated is not None,
            "consensus": aggregated.consensus_level if aggregated else 0
        })
    
    async def test_end_to_end_collaboration(self):
        """Test 7: Complete end-to-end collaboration scenario"""
        print("\n" + "="*60)
        print("Test 7: End-to-End Collaboration Scenario")
        print("="*60)
        
        # Create a real-world task
        task_description = """
        Create a user authentication system with the following requirements:
        1. User registration with email validation
        2. Secure password hashing
        3. JWT token generation
        4. Rate limiting for login attempts
        5. Unit tests for all components
        """
        
        print(f"\n📋 Creating real-world task:")
        print(task_description)
        
        # Create task
        task = await create_collaborative_task(
            task_description,
            complexity=4,
            required_capabilities=[
                "can_write_code",
                "can_design_architecture", 
                "can_test",
                "can_review_code",
                "can_optimize"
            ],
            deadline_hours=2
        )
        
        # Decompose task
        decomposed = self.task_decomposer.decompose_task(
            task,
            strategy=DecompositionStrategy.FUNCTIONAL,
            max_subtasks=5
        )
        
        print(f"\n🔍 Decomposed into {len(decomposed.subtasks)} subtasks")
        
        # Start collaboration
        session = await self.orchestrator.start_collaboration(
            task=decomposed,
            mode=CollaborationMode.MASTER_WORKER
        )
        
        print(f"\n🚀 Collaboration session started")
        print(f"   Session ID: {session.session_id}")
        print(f"   Participating agents: {len(session.participating_agents)}")
        
        # Create state machine for coordination
        context = StateContext(
            session_id=session.session_id,
            task=decomposed,
            agents=session.participating_agents,
            mode=CollaborationMode.MASTER_WORKER
        )
        
        state_machine = CoordinationStateMachine()
        
        # Run with timeout
        try:
            await asyncio.wait_for(
                state_machine.start(context),
                timeout=5
            )
        except asyncio.TimeoutError:
            pass
        
        # Simulate agent results
        agent_results = []
        for agent_id in session.participating_agents[:3]:
            result = AgentResult(
                agent_id=agent_id,
                role=AgentRole.WORKER,
                task_id=task.task_id,
                result_type=ResultType.CODE if "developer" in agent_id else ResultType.ANALYSIS,
                content=f"Implementation from {agent_id}",
                confidence=0.8 + (0.1 if "specialist" in agent_id else 0)
            )
            agent_results.append(result)
        
        # Aggregate results
        final_result = await self.result_aggregator.aggregate_results(agent_results)
        
        print(f"\n📊 Final Results:")
        print(f"   State transitions: {len(context.state_history)}")
        print(f"   Results collected: {len(agent_results)}")
        if final_result:
            print(f"   Final confidence: {final_result.confidence_score:.2%}")
            print(f"   Consensus level: {final_result.consensus_level:.2%}")
        
        self.test_results.append({
            "test": "end_to_end",
            "success": len(context.state_history) > 0 and final_result is not None,
            "agents": len(session.participating_agents)
        })
        
        return session, final_result
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get("success", False))
        
        print(f"\nTests Run: {total_tests}")
        print(f"Tests Passed: {passed_tests}")
        print(f"Tests Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for i, result in enumerate(self.test_results, 1):
            status = "✅" if result.get("success", False) else "❌"
            test_name = result.get("test", f"Test {i}")
            print(f"  {status} {test_name}")
            
            # Print additional details
            for key, value in result.items():
                if key not in ["test", "success"]:
                    print(f"      {key}: {value}")
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "🚀"*30)
        print("STARTING MULTI-AGENT COLLABORATION TESTS")
        print("🚀"*30)
        
        start_time = time.time()
        
        try:
            # Setup
            await self.setup()
            
            # Run tests
            await self.test_simple_task_collaboration()
            await self.test_complex_task_decomposition()
            await self.test_role_assignment()
            await self.test_message_queue_communication()
            await self.test_state_machine_workflow()
            await self.test_result_aggregation()
            await self.test_end_to_end_collaboration()
            
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            import traceback
            traceback.print_exc()
        
        # Print summary
        self.print_summary()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Total test time: {elapsed_time:.2f} seconds")
        
        return self.test_results


async def main():
    """Main test runner"""
    tester = MultiAgentCollaborationTest()
    results = await tester.run_all_tests()
    
    # Return exit code based on test results
    all_passed = all(r.get("success", False) for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    print("Multi-Agent Collaboration System Test Suite")
    print("="*60)
    
    exit_code = asyncio.run(main())
    
    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n⚠️ Some tests failed. Please review the results above.")
    
    sys.exit(exit_code)