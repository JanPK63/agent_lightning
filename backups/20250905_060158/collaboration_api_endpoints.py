#!/usr/bin/env python3
"""
Collaboration API Endpoints for Multi-Agent System
Exposes collaboration functionality through REST API
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from enum import Enum
import logging
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_collaboration import (
    CollaborationOrchestrator,
    CollaborativeTask,
    TaskComplexity,
    CollaborationMode,
    create_collaborative_task,
    start_collaboration_session
)
from agent_communication_protocol import AgentMessage, Performative
from agent_role_assignment import RoleAssigner, RoleAssignmentStrategy
from agent_message_queue import MessageQueueManager, QueueType
from task_decomposition import TaskDecomposer, DecompositionStrategy
from agent_coordination_state_machine import CoordinationStateMachine, StateContext, CoordinationState
from collaboration_result_aggregator import CollaborationResultAggregator, AgentResult, ResultType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
collaboration_app = FastAPI(
    title="Agent Lightning Collaboration API",
    description="Multi-Agent Collaboration System API",
    version="1.0.0"
)

# Global instances
orchestrator = CollaborationOrchestrator()
role_assigner = RoleAssigner()
queue_manager = MessageQueueManager()
task_decomposer = TaskDecomposer()
result_aggregator = CollaborationResultAggregator()
active_sessions: Dict[str, Dict[str, Any]] = {}
active_state_machines: Dict[str, CoordinationStateMachine] = {}


# Pydantic models for API

class TaskComplexityEnum(str, Enum):
    """Task complexity levels for API"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class CollaborationModeEnum(str, Enum):
    """Collaboration modes for API"""
    MASTER_WORKER = "master_worker"
    PEER_TO_PEER = "peer_to_peer"
    BLACKBOARD = "blackboard"
    CONTRACT_NET = "contract_net"
    PIPELINE = "pipeline"
    ENSEMBLE = "ensemble"


class CreateTaskRequest(BaseModel):
    """Request model for creating a collaborative task"""
    description: str = Field(..., description="Task description")
    complexity: TaskComplexityEnum = Field(TaskComplexityEnum.MODERATE, description="Task complexity level")
    required_capabilities: List[str] = Field(default_factory=list, description="Required agent capabilities")
    deadline_hours: Optional[int] = Field(None, description="Deadline in hours from now")
    subtasks: Optional[List[Dict[str, Any]]] = Field(None, description="Optional subtasks")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class StartCollaborationRequest(BaseModel):
    """Request model for starting a collaboration session"""
    task_id: str = Field(..., description="ID of the task to collaborate on")
    mode: CollaborationModeEnum = Field(CollaborationModeEnum.MASTER_WORKER, description="Collaboration mode")
    agents: Optional[List[str]] = Field(None, description="Specific agents to use (auto-select if not provided)")
    auto_execute: bool = Field(True, description="Automatically start execution")


class AssignRolesRequest(BaseModel):
    """Request model for role assignment"""
    task_id: str = Field(..., description="Task ID")
    agents: List[str] = Field(..., description="Available agents")
    strategy: str = Field("hungarian", description="Assignment strategy")


class SendMessageRequest(BaseModel):
    """Request model for sending a message"""
    sender: str = Field(..., description="Sender agent ID")
    receiver: str = Field(..., description="Receiver agent ID or 'broadcast'")
    content: Dict[str, Any] = Field(..., description="Message content")
    performative: str = Field("inform", description="Message performative")
    queue: str = Field("default", description="Queue name")


class AggregateResultsRequest(BaseModel):
    """Request model for aggregating results"""
    results: List[Dict[str, Any]] = Field(..., description="Results from agents")
    strategy: Optional[str] = Field(None, description="Aggregation strategy")
    conflict_resolution: str = Field("majority_vote", description="Conflict resolution strategy")


# Initialize on startup
@collaboration_app.on_event("startup")
async def startup_event():
    """Initialize collaboration system on startup"""
    try:
        await orchestrator.initialize()
        logger.info("Collaboration system initialized")
        
        # Create default queues
        queue_manager.create_queue("default", QueueType.FIFO)
        queue_manager.create_queue("priority", QueueType.PRIORITY)
        queue_manager.create_queue("collaboration", QueueType.TOPIC)
        logger.info("Message queues created")
        
    except Exception as e:
        logger.error(f"Failed to initialize collaboration system: {e}")


# Health check endpoint
@collaboration_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "registered_agents": len(orchestrator.agents)
    }


# Task management endpoints

@collaboration_app.post("/tasks/create")
async def create_task(request: CreateTaskRequest):
    """Create a new collaborative task"""
    try:
        # Map complexity enum
        complexity_map = {
            TaskComplexityEnum.TRIVIAL: 1,
            TaskComplexityEnum.SIMPLE: 2,
            TaskComplexityEnum.MODERATE: 3,
            TaskComplexityEnum.COMPLEX: 4,
            TaskComplexityEnum.VERY_COMPLEX: 5
        }
        
        # Create task
        task = await create_collaborative_task(
            description=request.description,
            complexity=complexity_map[request.complexity],
            required_capabilities=request.required_capabilities,
            deadline_hours=request.deadline_hours
        )
        
        # Add subtasks if provided
        if request.subtasks:
            for subtask_data in request.subtasks:
                subtask = CollaborativeTask(**subtask_data)
                task.add_subtask(subtask)
        
        # Add metadata
        task.metadata.update(request.metadata)
        
        # Store task
        active_sessions[task.task_id] = {
            "task": task,
            "created": datetime.now(),
            "status": "created"
        }
        
        logger.info(f"Created task: {task.task_id}")
        
        return {
            "task_id": task.task_id,
            "description": task.description,
            "complexity": task.complexity.name,
            "capabilities": task.required_capabilities,
            "subtasks": len(task.subtasks),
            "deadline": task.deadline.isoformat() if task.deadline else None
        }
        
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@collaboration_app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task details"""
    if task_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Task not found")
    
    session = active_sessions[task_id]
    task = session["task"]
    
    return {
        "task_id": task.task_id,
        "description": task.description,
        "complexity": task.complexity.name,
        "status": session["status"],
        "created": session["created"].isoformat(),
        "subtasks": [
            {
                "task_id": st.task_id,
                "description": st.description,
                "status": st.status
            }
            for st in task.subtasks
        ]
    }


@collaboration_app.post("/tasks/{task_id}/decompose")
async def decompose_task(
    task_id: str,
    strategy: Optional[str] = Query(None, description="Decomposition strategy"),
    max_depth: int = Query(3, description="Maximum decomposition depth"),
    max_subtasks: int = Query(7, description="Maximum number of subtasks")
):
    """Decompose a complex task into subtasks"""
    if task_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        task = active_sessions[task_id]["task"]
        
        # Map strategy string to enum
        strategy_enum = None
        if strategy:
            strategy_map = {
                "functional": DecompositionStrategy.FUNCTIONAL,
                "temporal": DecompositionStrategy.TEMPORAL,
                "data_parallel": DecompositionStrategy.DATA_PARALLEL,
                "hierarchical": DecompositionStrategy.HIERARCHICAL,
                "domain": DecompositionStrategy.DOMAIN,
                "hybrid": DecompositionStrategy.HYBRID
            }
            strategy_enum = strategy_map.get(strategy)
        
        # Decompose task
        decomposed = task_decomposer.decompose_task(
            task,
            strategy=strategy_enum,
            max_depth=max_depth,
            max_subtasks=max_subtasks
        )
        
        # Optimize dependencies
        optimized = task_decomposer.optimize_dependencies(decomposed)
        
        # Update stored task
        active_sessions[task_id]["task"] = optimized
        
        return {
            "task_id": task_id,
            "subtasks": len(optimized.subtasks),
            "strategy": strategy or "auto",
            "execution_levels": max(
                st.metadata.get("execution_level", 0)
                for st in optimized.subtasks
            ) + 1 if optimized.subtasks else 0
        }
        
    except Exception as e:
        logger.error(f"Error decomposing task: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Collaboration session endpoints

@collaboration_app.post("/collaboration/start")
async def start_collaboration(request: StartCollaborationRequest, background_tasks: BackgroundTasks):
    """Start a collaboration session"""
    if request.task_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        task = active_sessions[request.task_id]["task"]
        
        # Map mode string to enum
        mode_map = {
            CollaborationModeEnum.MASTER_WORKER: CollaborationMode.MASTER_WORKER,
            CollaborationModeEnum.PEER_TO_PEER: CollaborationMode.PEER_TO_PEER,
            CollaborationModeEnum.BLACKBOARD: CollaborationMode.BLACKBOARD,
            CollaborationModeEnum.CONTRACT_NET: CollaborationMode.CONTRACT_NET,
            CollaborationModeEnum.PIPELINE: CollaborationMode.PIPELINE,
            CollaborationModeEnum.ENSEMBLE: CollaborationMode.ENSEMBLE
        }
        mode = mode_map[request.mode]
        
        # Start collaboration
        session = await orchestrator.start_collaboration(
            task=task,
            mode=mode,
            agents=request.agents
        )
        
        # Update session info
        active_sessions[request.task_id]["session_id"] = session.session_id
        active_sessions[request.task_id]["status"] = "collaborating"
        active_sessions[request.task_id]["mode"] = mode.value
        active_sessions[request.task_id]["agents"] = session.participating_agents
        
        # Start state machine if auto_execute
        if request.auto_execute:
            context = StateContext(
                session_id=session.session_id,
                task=task,
                agents=session.participating_agents,
                mode=mode
            )
            
            state_machine = CoordinationStateMachine()
            active_state_machines[session.session_id] = state_machine
            
            # Run in background
            background_tasks.add_task(state_machine.start, context)
        
        logger.info(f"Started collaboration session: {session.session_id}")
        
        return {
            "session_id": session.session_id,
            "task_id": request.task_id,
            "mode": request.mode,
            "agents": session.participating_agents,
            "status": session.status,
            "auto_executing": request.auto_execute
        }
        
    except Exception as e:
        logger.error(f"Error starting collaboration: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@collaboration_app.get("/collaboration/{session_id}/status")
async def get_collaboration_status(session_id: str):
    """Get collaboration session status"""
    try:
        status = await orchestrator.get_session_status(session_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add state machine status if available
        if session_id in active_state_machines:
            state_machine = active_state_machines[session_id]
            if state_machine.context:
                status["current_state"] = state_machine.context.current_state.name
                status["state_duration"] = str(state_machine.context.get_state_duration())
                status["error_count"] = state_machine.context.error_count
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@collaboration_app.post("/collaboration/{session_id}/cancel")
async def cancel_collaboration(session_id: str):
    """Cancel a collaboration session"""
    try:
        # Cancel state machine if running
        if session_id in active_state_machines:
            state_machine = active_state_machines[session_id]
            if state_machine.context:
                state_machine.context.metadata["cancel_requested"] = True
        
        # Update session status
        for task_id, session_data in active_sessions.items():
            if session_data.get("session_id") == session_id:
                session_data["status"] = "cancelled"
                break
        
        return {
            "session_id": session_id,
            "status": "cancelled",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cancelling session: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Role assignment endpoints

@collaboration_app.post("/roles/assign")
async def assign_roles(request: AssignRolesRequest):
    """Assign roles to agents for a task"""
    if request.task_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        task = active_sessions[request.task_id]["task"]
        
        # Register agents with role assigner
        from agent_collaboration import SpecializedCollaborativeAgent
        from agent_config import AgentConfigManager
        from enhanced_production_api import EnhancedAgentService
        
        config_manager = AgentConfigManager()
        agent_service = EnhancedAgentService()
        
        agents = []
        for agent_id in request.agents:
            config = config_manager.get_agent(agent_id)
            if config:
                agent = SpecializedCollaborativeAgent(
                    agent_id=agent_id,
                    config=config,
                    agent_service=agent_service
                )
                agents.append(agent)
        
        # Map strategy
        strategy_map = {
            "hungarian": RoleAssignmentStrategy.HUNGARIAN,
            "capability_based": RoleAssignmentStrategy.CAPABILITY_BASED,
            "experience_based": RoleAssignmentStrategy.EXPERIENCE_BASED,
            "load_balanced": RoleAssignmentStrategy.LOAD_BALANCED,
            "specialization": RoleAssignmentStrategy.SPECIALIZATION,
            "cost_optimized": RoleAssignmentStrategy.COST_OPTIMIZED
        }
        strategy = strategy_map.get(request.strategy, RoleAssignmentStrategy.HUNGARIAN)
        
        # Assign roles
        assignments = role_assigner.assign_roles(task, agents, strategy)
        
        return {
            "task_id": request.task_id,
            "assignments": {
                agent_id: role.value
                for agent_id, role in assignments.items()
            },
            "strategy": request.strategy,
            "total_assigned": len(assignments)
        }
        
    except Exception as e:
        logger.error(f"Error assigning roles: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@collaboration_app.get("/roles/requirements/{task_id}")
async def get_role_requirements(task_id: str):
    """Get role requirements for a task"""
    if task_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        task = active_sessions[task_id]["task"]
        requirements = role_assigner.analyze_task_requirements(task)
        
        return {
            "task_id": task_id,
            "required_roles": {
                role.value: count
                for role, count in requirements.items()
            },
            "total_roles": sum(requirements.values())
        }
        
    except Exception as e:
        logger.error(f"Error getting role requirements: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Message queue endpoints

@collaboration_app.post("/messages/send")
async def send_message(request: SendMessageRequest):
    """Send a message through the queue system"""
    try:
        # Create message
        msg = AgentMessage(
            performative=Performative[request.performative.upper()],
            sender=request.sender,
            receiver=request.receiver,
            content=request.content
        )
        
        # Send through queue
        success = await queue_manager.send_message(msg, request.queue)
        
        return {
            "message_id": msg.message_id,
            "sender": request.sender,
            "receiver": request.receiver,
            "queue": request.queue,
            "sent": success,
            "timestamp": msg.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@collaboration_app.get("/messages/receive")
async def receive_message(
    agent_id: str = Query(..., description="Agent ID"),
    queue: str = Query("default", description="Queue name"),
    timeout: float = Query(1.0, description="Timeout in seconds")
):
    """Receive a message from the queue"""
    try:
        msg = await queue_manager.receive_message(agent_id, queue, timeout=timeout)
        
        if msg:
            return {
                "message_id": msg.message_id,
                "sender": msg.sender,
                "receiver": msg.receiver,
                "content": msg.content,
                "performative": msg.performative.value,
                "timestamp": msg.timestamp.isoformat()
            }
        else:
            return {"message": "No messages available"}
        
    except Exception as e:
        logger.error(f"Error receiving message: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@collaboration_app.get("/messages/queues")
async def get_queue_status():
    """Get status of all message queues"""
    try:
        metrics = queue_manager.get_all_metrics()
        
        return {
            "total_queues": metrics["total_queues"],
            "total_subscribers": metrics["total_subscribers"],
            "queues": metrics["queues"]
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Result aggregation endpoints

@collaboration_app.post("/results/aggregate")
async def aggregate_results(request: AggregateResultsRequest):
    """Aggregate results from multiple agents"""
    try:
        # Convert dicts to AgentResult objects
        results = []
        for result_data in request.results:
            result = AgentResult(
                agent_id=result_data["agent_id"],
                role=result_data.get("role", "worker"),
                task_id=result_data["task_id"],
                result_type=ResultType[result_data.get("result_type", "TEXT").upper()],
                content=result_data["content"],
                confidence=result_data.get("confidence", 1.0)
            )
            results.append(result)
        
        # Aggregate
        aggregated = await result_aggregator.aggregate_results(results)
        
        if aggregated:
            return {
                "task_id": aggregated.task_id,
                "strategy": aggregated.aggregation_strategy.name,
                "final_result": aggregated.final_result,
                "confidence": aggregated.confidence_score,
                "consensus": aggregated.consensus_level,
                "conflicts_resolved": aggregated.conflicts_resolved,
                "contributing_agents": aggregated.contributing_agents
            }
        else:
            return {"message": "No results to aggregate"}
        
    except Exception as e:
        logger.error(f"Error aggregating results: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@collaboration_app.get("/results/consensus/{task_id}")
async def get_consensus_report(task_id: str):
    """Get consensus report for aggregated results"""
    try:
        # Find most recent aggregation for this task
        for aggregated in reversed(result_aggregator.aggregation_history):
            if aggregated.task_id == task_id:
                report = result_aggregator.get_consensus_report(aggregated)
                return report
        
        raise HTTPException(status_code=404, detail="No aggregation found for task")
        
    except Exception as e:
        logger.error(f"Error getting consensus report: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Metrics and monitoring endpoints

@collaboration_app.get("/metrics")
async def get_system_metrics():
    """Get system-wide metrics"""
    try:
        return {
            "orchestrator": {
                "total_sessions": len(orchestrator.sessions),
                "active_agents": len(orchestrator.agents),
                "performance_metrics": dict(orchestrator.performance_metrics)
            },
            "role_assigner": role_assigner.get_assignment_report(),
            "queues": queue_manager.get_all_metrics(),
            "aggregator": result_aggregator.get_aggregation_metrics(),
            "active_sessions": len(active_sessions),
            "active_state_machines": len(active_state_machines)
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@collaboration_app.get("/agents")
async def list_agents():
    """List all available agents"""
    try:
        from agent_config import AgentConfigManager
        
        config_manager = AgentConfigManager()
        agent_names = config_manager.list_agents()
        
        agents = []
        for name in agent_names:
            config = config_manager.get_agent(name)
            if config:
                capabilities = [k for k, v in config.capabilities.__dict__.items() if v]
                agents.append({
                    "agent_id": name,
                    "type": config.agent_type,
                    "model": config.model_name,
                    "capabilities": capabilities
                })
        
        return {
            "total_agents": len(agents),
            "agents": agents
        }
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Testing endpoint for demonstration
@collaboration_app.post("/demo/simple-collaboration")
async def demo_simple_collaboration(
    task_description: str = Body(..., description="Task description"),
    background_tasks: BackgroundTasks = None
):
    """Demo endpoint for simple collaboration"""
    try:
        # Create task
        task = await create_collaborative_task(
            task_description,
            complexity=3,
            required_capabilities=["can_write_code", "can_review_code"]
        )
        
        # Store task
        active_sessions[task.task_id] = {
            "task": task,
            "created": datetime.now(),
            "status": "created"
        }
        
        # Start collaboration
        session = await orchestrator.start_collaboration(
            task=task,
            mode=CollaborationMode.MASTER_WORKER
        )
        
        # Update session
        active_sessions[task.task_id]["session_id"] = session.session_id
        active_sessions[task.task_id]["status"] = "collaborating"
        
        # Start execution in background
        context = StateContext(
            session_id=session.session_id,
            task=task,
            agents=session.participating_agents,
            mode=CollaborationMode.MASTER_WORKER
        )
        
        state_machine = CoordinationStateMachine()
        active_state_machines[session.session_id] = state_machine
        
        if background_tasks:
            background_tasks.add_task(state_machine.start, context)
        
        return {
            "task_id": task.task_id,
            "session_id": session.session_id,
            "description": task_description,
            "agents": session.participating_agents,
            "status": "started",
            "message": "Collaboration started. Check status endpoint for progress."
        }
        
    except Exception as e:
        logger.error(f"Error in demo collaboration: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Collaboration API Server")
    print("="*60)
    
    # Run the API server
    uvicorn.run(
        collaboration_app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )