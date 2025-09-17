#!/usr/bin/env python3
"""
Agent Coordination State Machine for Multi-Agent Collaboration
Manages state transitions and coordination workflows between agents
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
import logging
import networkx as nx
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_collaboration import (
    CollaborativeTask,
    CollaborationSession,
    CollaborationMode,
    AgentRole,
    TaskComplexity
)
from agent_communication_protocol import AgentMessage, Performative
from agent_role_assignment import RoleAssigner, AgentProfile
from agent_message_queue import MessageQueueManager, QueueType, DeliveryMode
from task_decomposition import TaskDecomposer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinationState(Enum):
    """States in agent coordination workflow"""
    # Initialization states
    IDLE = auto()
    INITIALIZING = auto()
    
    # Planning states
    ANALYZING_TASK = auto()
    DECOMPOSING_TASK = auto()
    ASSIGNING_ROLES = auto()
    PLANNING_EXECUTION = auto()
    
    # Execution states
    DISTRIBUTING_WORK = auto()
    EXECUTING = auto()
    MONITORING = auto()
    SYNCHRONIZING = auto()
    
    # Completion states
    COLLECTING_RESULTS = auto()
    AGGREGATING = auto()
    REVIEWING = auto()
    FINALIZING = auto()
    
    # Terminal states
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    
    # Recovery states
    RECOVERING = auto()
    REASSIGNING = auto()
    ROLLBACK = auto()


class StateTransition:
    """Represents a valid state transition"""
    
    def __init__(
        self,
        from_state: CoordinationState,
        to_state: CoordinationState,
        condition: Optional[Callable] = None,
        action: Optional[Callable] = None,
        name: str = ""
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition or (lambda ctx: True)
        self.action = action
        self.name = name or f"{from_state.name} -> {to_state.name}"
    
    def can_transition(self, context: Dict[str, Any]) -> bool:
        """Check if transition is allowed"""
        return self.condition(context)
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transition action"""
        if self.action:
            return await self.action(context)
        return context


@dataclass
class StateContext:
    """Context for state machine execution"""
    session_id: str
    task: CollaborativeTask
    agents: List[str]
    mode: CollaborationMode
    current_state: CoordinationState = CoordinationState.IDLE
    previous_state: Optional[CoordinationState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    max_errors: int = 3
    start_time: datetime = field(default_factory=datetime.now)
    state_history: List[Tuple[CoordinationState, datetime]] = field(default_factory=list)
    
    def update_state(self, new_state: CoordinationState):
        """Update current state and maintain history"""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_history.append((new_state, datetime.now()))
        logger.info(f"State transition: {self.previous_state.name} -> {new_state.name}")
    
    def get_state_duration(self) -> timedelta:
        """Get duration in current state"""
        if self.state_history:
            return datetime.now() - self.state_history[-1][1]
        return datetime.now() - self.start_time
    
    def increment_error(self) -> bool:
        """Increment error count and check if max reached"""
        self.error_count += 1
        return self.error_count >= self.max_errors


class CoordinationStateMachine:
    """State machine for agent coordination"""
    
    def __init__(self):
        self.transitions: List[StateTransition] = []
        self.state_handlers: Dict[CoordinationState, Callable] = {}
        self.context: Optional[StateContext] = None
        self.running = False
        
        # Components
        self.task_decomposer = TaskDecomposer()
        self.role_assigner = RoleAssigner()
        self.queue_manager = MessageQueueManager()
        
        # Initialize state machine
        self._initialize_transitions()
        self._initialize_handlers()
        
        # Create queues
        self.queue_manager.create_queue("coordination", QueueType.PRIORITY)
        self.queue_manager.create_queue("tasks", QueueType.WORK_STEALING)
        self.queue_manager.create_queue("results", QueueType.FIFO)
    
    def _initialize_transitions(self):
        """Define valid state transitions"""
        
        # Initialization transitions
        self.add_transition(
            CoordinationState.IDLE,
            CoordinationState.INITIALIZING,
            name="Start Coordination"
        )
        
        self.add_transition(
            CoordinationState.INITIALIZING,
            CoordinationState.ANALYZING_TASK,
            condition=lambda ctx: ctx["agents"] and ctx["task"],
            name="Begin Analysis"
        )
        
        # Planning transitions
        self.add_transition(
            CoordinationState.ANALYZING_TASK,
            CoordinationState.DECOMPOSING_TASK,
            condition=lambda ctx: ctx["task"].complexity.value >= 3,
            name="Complex Task Decomposition"
        )
        
        self.add_transition(
            CoordinationState.ANALYZING_TASK,
            CoordinationState.ASSIGNING_ROLES,
            condition=lambda ctx: ctx["task"].complexity.value < 3,
            name="Simple Task Assignment"
        )
        
        self.add_transition(
            CoordinationState.DECOMPOSING_TASK,
            CoordinationState.ASSIGNING_ROLES,
            name="Proceed to Role Assignment"
        )
        
        self.add_transition(
            CoordinationState.ASSIGNING_ROLES,
            CoordinationState.PLANNING_EXECUTION,
            condition=lambda ctx: "role_assignments" in ctx["metadata"],
            name="Plan Execution"
        )
        
        # Execution transitions
        self.add_transition(
            CoordinationState.PLANNING_EXECUTION,
            CoordinationState.DISTRIBUTING_WORK,
            name="Distribute Work"
        )
        
        self.add_transition(
            CoordinationState.DISTRIBUTING_WORK,
            CoordinationState.EXECUTING,
            name="Begin Execution"
        )
        
        self.add_transition(
            CoordinationState.EXECUTING,
            CoordinationState.MONITORING,
            name="Monitor Progress"
        )
        
        self.add_transition(
            CoordinationState.MONITORING,
            CoordinationState.SYNCHRONIZING,
            condition=lambda ctx: ctx["metadata"].get("sync_needed", False),
            name="Synchronize Agents"
        )
        
        self.add_transition(
            CoordinationState.MONITORING,
            CoordinationState.EXECUTING,
            condition=lambda ctx: not ctx["metadata"].get("tasks_complete", False),
            name="Continue Execution"
        )
        
        self.add_transition(
            CoordinationState.SYNCHRONIZING,
            CoordinationState.EXECUTING,
            name="Resume Execution"
        )
        
        # Completion transitions
        self.add_transition(
            CoordinationState.MONITORING,
            CoordinationState.COLLECTING_RESULTS,
            condition=lambda ctx: ctx["metadata"].get("tasks_complete", False),
            name="Collect Results"
        )
        
        self.add_transition(
            CoordinationState.COLLECTING_RESULTS,
            CoordinationState.AGGREGATING,
            name="Aggregate Results"
        )
        
        self.add_transition(
            CoordinationState.AGGREGATING,
            CoordinationState.REVIEWING,
            condition=lambda ctx: ctx["mode"] != CollaborationMode.PEER_TO_PEER,
            name="Review Results"
        )
        
        self.add_transition(
            CoordinationState.AGGREGATING,
            CoordinationState.FINALIZING,
            condition=lambda ctx: ctx["mode"] == CollaborationMode.PEER_TO_PEER,
            name="Skip Review"
        )
        
        self.add_transition(
            CoordinationState.REVIEWING,
            CoordinationState.FINALIZING,
            name="Finalize"
        )
        
        self.add_transition(
            CoordinationState.FINALIZING,
            CoordinationState.COMPLETED,
            name="Complete"
        )
        
        # Error recovery transitions
        self.add_transition(
            CoordinationState.EXECUTING,
            CoordinationState.RECOVERING,
            condition=lambda ctx: ctx["error_count"] > 0,
            name="Error Recovery"
        )
        
        self.add_transition(
            CoordinationState.RECOVERING,
            CoordinationState.REASSIGNING,
            condition=lambda ctx: ctx["metadata"].get("agent_failure", False),
            name="Reassign Failed Agent"
        )
        
        self.add_transition(
            CoordinationState.RECOVERING,
            CoordinationState.EXECUTING,
            condition=lambda ctx: ctx["error_count"] < ctx["max_errors"],
            name="Retry Execution"
        )
        
        self.add_transition(
            CoordinationState.REASSIGNING,
            CoordinationState.DISTRIBUTING_WORK,
            name="Redistribute Work"
        )
        
        self.add_transition(
            CoordinationState.RECOVERING,
            CoordinationState.ROLLBACK,
            condition=lambda ctx: ctx["error_count"] >= ctx["max_errors"],
            name="Rollback"
        )
        
        self.add_transition(
            CoordinationState.ROLLBACK,
            CoordinationState.FAILED,
            name="Fail"
        )
        
        # Cancellation transitions (from any state)
        for state in CoordinationState:
            if state not in [CoordinationState.COMPLETED, CoordinationState.FAILED, CoordinationState.CANCELLED]:
                self.add_transition(
                    state,
                    CoordinationState.CANCELLED,
                    condition=lambda ctx: ctx["metadata"].get("cancel_requested", False),
                    name="Cancel"
                )
    
    def _initialize_handlers(self):
        """Define state handlers"""
        
        self.state_handlers = {
            CoordinationState.IDLE: self._handle_idle,
            CoordinationState.INITIALIZING: self._handle_initializing,
            CoordinationState.ANALYZING_TASK: self._handle_analyzing_task,
            CoordinationState.DECOMPOSING_TASK: self._handle_decomposing_task,
            CoordinationState.ASSIGNING_ROLES: self._handle_assigning_roles,
            CoordinationState.PLANNING_EXECUTION: self._handle_planning_execution,
            CoordinationState.DISTRIBUTING_WORK: self._handle_distributing_work,
            CoordinationState.EXECUTING: self._handle_executing,
            CoordinationState.MONITORING: self._handle_monitoring,
            CoordinationState.SYNCHRONIZING: self._handle_synchronizing,
            CoordinationState.COLLECTING_RESULTS: self._handle_collecting_results,
            CoordinationState.AGGREGATING: self._handle_aggregating,
            CoordinationState.REVIEWING: self._handle_reviewing,
            CoordinationState.FINALIZING: self._handle_finalizing,
            CoordinationState.COMPLETED: self._handle_completed,
            CoordinationState.FAILED: self._handle_failed,
            CoordinationState.CANCELLED: self._handle_cancelled,
            CoordinationState.RECOVERING: self._handle_recovering,
            CoordinationState.REASSIGNING: self._handle_reassigning,
            CoordinationState.ROLLBACK: self._handle_rollback
        }
    
    def add_transition(
        self,
        from_state: CoordinationState,
        to_state: CoordinationState,
        condition: Optional[Callable] = None,
        action: Optional[Callable] = None,
        name: str = ""
    ):
        """Add a state transition"""
        transition = StateTransition(from_state, to_state, condition, action, name)
        self.transitions.append(transition)
    
    def get_valid_transitions(self, state: CoordinationState) -> List[StateTransition]:
        """Get all valid transitions from a state"""
        return [t for t in self.transitions if t.from_state == state]
    
    async def start(self, context: StateContext):
        """Start the state machine"""
        self.context = context
        self.running = True
        
        logger.info(f"Starting coordination state machine for session {context.session_id}")
        
        while self.running and not self._is_terminal_state(self.context.current_state):
            try:
                # Execute current state handler
                await self._execute_state_handler(self.context.current_state)
                
                # Find and execute valid transition
                await self._process_transition()
                
                # Check for timeout
                if datetime.now() - self.context.start_time > timedelta(hours=1):
                    logger.error("Coordination timeout reached")
                    self.context.metadata["cancel_requested"] = True
                
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
                
            except Exception as e:
                logger.error(f"Error in state machine: {e}")
                self.context.increment_error()
                if self.context.error_count >= self.context.max_errors:
                    self.context.update_state(CoordinationState.FAILED)
                    break
        
        logger.info(f"State machine completed with state: {self.context.current_state.name}")
    
    async def _execute_state_handler(self, state: CoordinationState):
        """Execute the handler for current state"""
        if state in self.state_handlers:
            handler = self.state_handlers[state]
            await handler()
    
    async def _process_transition(self):
        """Process state transition"""
        valid_transitions = self.get_valid_transitions(self.context.current_state)
        
        for transition in valid_transitions:
            context_dict = {
                "session_id": self.context.session_id,
                "task": self.context.task,
                "agents": self.context.agents,
                "mode": self.context.mode,
                "metadata": self.context.metadata,
                "error_count": self.context.error_count,
                "max_errors": self.context.max_errors
            }
            
            if transition.can_transition(context_dict):
                logger.info(f"Executing transition: {transition.name}")
                
                # Execute transition action if any
                if transition.action:
                    updated_context = await transition.execute(context_dict)
                    self.context.metadata.update(updated_context.get("metadata", {}))
                
                # Update state
                self.context.update_state(transition.to_state)
                break
    
    def _is_terminal_state(self, state: CoordinationState) -> bool:
        """Check if state is terminal"""
        return state in [
            CoordinationState.COMPLETED,
            CoordinationState.FAILED,
            CoordinationState.CANCELLED
        ]
    
    # State Handlers
    
    async def _handle_idle(self):
        """Handle idle state"""
        logger.debug("In IDLE state, waiting to start")
    
    async def _handle_initializing(self):
        """Handle initialization"""
        logger.info("Initializing coordination session")
        
        # Register agents with queues
        for agent_id in self.context.agents:
            self.queue_manager.subscribe(agent_id, "coordination")
            self.queue_manager.subscribe(agent_id, "tasks")
            self.queue_manager.subscribe(agent_id, "results")
        
        self.context.metadata["initialized"] = True
    
    async def _handle_analyzing_task(self):
        """Analyze the task"""
        logger.info("Analyzing task requirements")
        
        # Analyze task using decomposer
        analysis = self.task_decomposer.analyze_task(self.context.task.description)
        
        self.context.metadata["task_analysis"] = analysis
        self.context.metadata["task_type"] = analysis["task_type"].value
        self.context.metadata["complexity"] = analysis["estimated_complexity"].value
        self.context.metadata["capabilities_needed"] = analysis["detected_capabilities"]
    
    async def _handle_decomposing_task(self):
        """Decompose complex task"""
        logger.info("Decomposing complex task into subtasks")
        
        # Decompose task
        decomposed = self.task_decomposer.decompose_task(
            self.context.task,
            max_depth=2,
            min_subtasks=2,
            max_subtasks=len(self.context.agents)
        )
        
        # Optimize dependencies
        optimized = self.task_decomposer.optimize_dependencies(decomposed)
        
        self.context.task = optimized
        self.context.metadata["subtask_count"] = len(optimized.subtasks)
        self.context.metadata["decomposition_complete"] = True
        
        logger.info(f"Task decomposed into {len(optimized.subtasks)} subtasks")
    
    async def _handle_assigning_roles(self):
        """Assign roles to agents"""
        logger.info("Assigning roles to agents")
        
        # Register agents with role assigner
        from agent_collaboration import SpecializedCollaborativeAgent
        from agent_config import AgentConfigManager
        from enhanced_production_api import EnhancedAgentService
        
        config_manager = AgentConfigManager()
        agent_service = EnhancedAgentService()
        
        agents = []
        for agent_id in self.context.agents:
            config = config_manager.get_agent(agent_id)
            if config:
                agent = SpecializedCollaborativeAgent(
                    agent_id=agent_id,
                    config=config,
                    agent_service=agent_service
                )
                agents.append(agent)
                
                # Create profile
                capabilities = [k for k, v in config.capabilities.__dict__.items() if v]
                profile = AgentProfile(
                    agent_id=agent_id,
                    capabilities=capabilities,
                    experience_level=5
                )
                self.role_assigner.register_agent(agent, profile)
        
        # Assign roles
        assignments = self.role_assigner.assign_roles(
            self.context.task,
            agents
        )
        
        self.context.metadata["role_assignments"] = assignments
        logger.info(f"Assigned roles: {assignments}")
    
    async def _handle_planning_execution(self):
        """Plan execution strategy"""
        logger.info("Planning execution strategy")
        
        execution_plan = {
            "mode": self.context.mode.value,
            "parallelizable_tasks": [],
            "sequential_tasks": [],
            "dependencies": {}
        }
        
        # Analyze task dependencies
        if self.context.task.subtasks:
            for subtask in self.context.task.subtasks:
                if not subtask.dependencies:
                    execution_plan["parallelizable_tasks"].append(subtask.task_id)
                else:
                    execution_plan["sequential_tasks"].append(subtask.task_id)
                    execution_plan["dependencies"][subtask.task_id] = subtask.dependencies
        else:
            execution_plan["parallelizable_tasks"].append(self.context.task.task_id)
        
        self.context.metadata["execution_plan"] = execution_plan
    
    async def _handle_distributing_work(self):
        """Distribute work to agents"""
        logger.info("Distributing work to agents")
        
        role_assignments = self.context.metadata.get("role_assignments", {})
        
        # Create task messages for each agent
        for agent_id, role in role_assignments.items():
            # Find appropriate subtasks for this agent's role
            tasks_for_agent = []
            
            if self.context.task.subtasks:
                for subtask in self.context.task.subtasks:
                    # Assign based on role and capabilities
                    if self._match_task_to_role(subtask, role):
                        tasks_for_agent.append(subtask)
            else:
                tasks_for_agent.append(self.context.task)
            
            # Send task messages
            for task in tasks_for_agent:
                msg = AgentMessage(
                    performative=Performative.REQUEST,
                    sender="coordinator",
                    receiver=agent_id,
                    content={
                        "action": "execute_task",
                        "task": asdict(task),
                        "role": role.value
                    }
                )
                
                await self.queue_manager.send_message(msg, "tasks")
        
        self.context.metadata["work_distributed"] = True
    
    def _match_task_to_role(self, task: CollaborativeTask, role: AgentRole) -> bool:
        """Match task to agent role"""
        # Simple matching logic - can be enhanced
        role_task_map = {
            AgentRole.WORKER: True,  # Workers can do any task
            AgentRole.SPECIALIST: task.complexity.value >= 3,
            AgentRole.REVIEWER: "review" in task.description.lower(),
            AgentRole.COORDINATOR: "coordinate" in task.description.lower(),
            AgentRole.AGGREGATOR: "aggregate" in task.description.lower(),
            AgentRole.MONITOR: "monitor" in task.description.lower()
        }
        
        return role_task_map.get(role, False)
    
    async def _handle_executing(self):
        """Handle execution state"""
        logger.info("Agents executing tasks")
        
        # Track execution progress
        if "execution_start" not in self.context.metadata:
            self.context.metadata["execution_start"] = datetime.now()
        
        # Simulate execution progress (in real system, would monitor actual agent work)
        await asyncio.sleep(1)
    
    async def _handle_monitoring(self):
        """Monitor execution progress"""
        logger.info("Monitoring execution progress")
        
        # Check if tasks are complete (simplified)
        execution_time = datetime.now() - self.context.metadata.get(
            "execution_start", datetime.now()
        )
        
        # For demo, mark complete after 3 seconds
        if execution_time > timedelta(seconds=3):
            self.context.metadata["tasks_complete"] = True
            logger.info("Tasks marked as complete")
        
        # Check if synchronization needed
        if execution_time > timedelta(seconds=2) and not self.context.metadata.get("synced", False):
            self.context.metadata["sync_needed"] = True
    
    async def _handle_synchronizing(self):
        """Synchronize agent states"""
        logger.info("Synchronizing agents")
        
        # Send sync message to all agents
        sync_msg = AgentMessage(
            performative=Performative.SYNC,
            sender="coordinator",
            receiver="broadcast",
            content={
                "action": "synchronize",
                "checkpoint": datetime.now().isoformat()
            }
        )
        
        for agent_id in self.context.agents:
            await self.queue_manager.send_message(sync_msg, "coordination")
        
        self.context.metadata["sync_needed"] = False
        self.context.metadata["synced"] = True
    
    async def _handle_collecting_results(self):
        """Collect results from agents"""
        logger.info("Collecting results from agents")
        
        collected_results = []
        
        # Collect results from queue (simplified)
        for agent_id in self.context.agents:
            # In real system, would actually collect from results queue
            result = {
                "agent": agent_id,
                "status": "completed",
                "output": f"Result from {agent_id}"
            }
            collected_results.append(result)
        
        self.context.metadata["collected_results"] = collected_results
        logger.info(f"Collected {len(collected_results)} results")
    
    async def _handle_aggregating(self):
        """Aggregate results"""
        logger.info("Aggregating results")
        
        results = self.context.metadata.get("collected_results", [])
        
        # Simple aggregation
        aggregated = {
            "total_agents": len(results),
            "completed": sum(1 for r in results if r["status"] == "completed"),
            "outputs": [r["output"] for r in results],
            "aggregation_time": datetime.now().isoformat()
        }
        
        self.context.metadata["aggregated_result"] = aggregated
    
    async def _handle_reviewing(self):
        """Review aggregated results"""
        logger.info("Reviewing results")
        
        # In real system, would have review logic
        self.context.metadata["review_complete"] = True
        self.context.metadata["review_status"] = "approved"
    
    async def _handle_finalizing(self):
        """Finalize coordination"""
        logger.info("Finalizing coordination")
        
        # Clean up resources
        for agent_id in self.context.agents:
            self.queue_manager.unsubscribe(agent_id, "coordination")
            self.queue_manager.unsubscribe(agent_id, "tasks")
            self.queue_manager.unsubscribe(agent_id, "results")
        
        self.context.metadata["finalized"] = True
    
    async def _handle_completed(self):
        """Handle completion"""
        logger.info("Coordination completed successfully")
        self.running = False
    
    async def _handle_failed(self):
        """Handle failure"""
        logger.error("Coordination failed")
        self.running = False
    
    async def _handle_cancelled(self):
        """Handle cancellation"""
        logger.warning("Coordination cancelled")
        self.running = False
    
    async def _handle_recovering(self):
        """Handle error recovery"""
        logger.warning("Attempting to recover from error")
        
        # Check type of error and determine recovery strategy
        if self.context.metadata.get("agent_failure"):
            self.context.metadata["needs_reassignment"] = True
        else:
            # Try simple retry
            await asyncio.sleep(1)
    
    async def _handle_reassigning(self):
        """Reassign work from failed agent"""
        logger.info("Reassigning work from failed agent")
        
        # Find replacement agent and reassign
        # In real system, would implement actual reassignment logic
        self.context.metadata["agent_failure"] = False
        self.context.metadata["reassignment_complete"] = True
    
    async def _handle_rollback(self):
        """Handle rollback"""
        logger.warning("Rolling back coordination")
        
        # Rollback logic here
        self.context.metadata["rollback_complete"] = True
    
    def visualize_state_machine(self) -> str:
        """Generate visualization of state machine"""
        graph = nx.DiGraph()
        
        # Add states as nodes
        for state in CoordinationState:
            graph.add_node(state.name)
        
        # Add transitions as edges
        for transition in self.transitions:
            graph.add_edge(
                transition.from_state.name,
                transition.to_state.name,
                label=transition.name
            )
        
        # Generate DOT format for visualization
        dot = ["digraph CoordinationStateMachine {"]
        dot.append("  rankdir=TB;")
        dot.append("  node [shape=ellipse];")
        
        # Mark terminal states
        for state in [CoordinationState.COMPLETED, CoordinationState.FAILED, CoordinationState.CANCELLED]:
            dot.append(f'  {state.name} [shape=doublecircle, style=filled, fillcolor=lightgray];')
        
        # Add edges
        for transition in self.transitions:
            label = transition.name.replace(" ", "_")
            dot.append(f'  {transition.from_state.name} -> {transition.to_state.name} [label="{label}"];')
        
        dot.append("}")
        
        return "\n".join(dot)


# Example usage and testing
async def test_coordination_state_machine():
    """Test the coordination state machine"""
    print("\n" + "="*60)
    print("Testing Agent Coordination State Machine")
    print("="*60)
    
    from agent_collaboration import CollaborativeTask
    
    # Create a test task
    task = CollaborativeTask(
        description="Analyze and optimize Python codebase",
        complexity=TaskComplexity.COMPLEX,
        required_capabilities=["code_analysis", "optimization"]
    )
    
    # Create context
    context = StateContext(
        session_id="test_session_001",
        task=task,
        agents=["agent_1", "agent_2", "agent_3"],
        mode=CollaborationMode.MASTER_WORKER
    )
    
    # Create state machine
    state_machine = CoordinationStateMachine()
    
    print(f"\nStarting state machine for task: {task.description}")
    print(f"Initial state: {context.current_state.name}")
    print(f"Agents: {context.agents}")
    print(f"Mode: {context.mode.value}")
    
    # Start state machine
    await state_machine.start(context)
    
    print(f"\nFinal state: {context.current_state.name}")
    print(f"State history ({len(context.state_history)} transitions):")
    for state, timestamp in context.state_history:
        print(f"  - {state.name} at {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
    
    # Print execution metadata
    print("\nExecution metadata:")
    for key, value in context.metadata.items():
        if not isinstance(value, (dict, list)):
            print(f"  {key}: {value}")
    
    # Generate visualization
    dot_graph = state_machine.visualize_state_machine()
    print("\nState machine visualization (DOT format):")
    print("First 5 transitions:")
    for line in dot_graph.split('\n')[4:9]:
        print(line)
    
    return state_machine, context


if __name__ == "__main__":
    print("Agent Coordination State Machine")
    print("="*60)
    
    # Run test
    state_machine, context = asyncio.run(test_coordination_state_machine())
    
    print("\n✅ Agent Coordination State Machine ready!")