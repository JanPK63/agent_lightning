#!/usr/bin/env python3
"""
Agent Designer Microservice - Integrated with Shared Database
Handles all agent design, creation, and management operations using shared PostgreSQL and Redis
"""

import os
import sys
import json
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared data access layer
from shared.data_access import DataAccessLayer
from shared.events import EventChannel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class AgentType(str, Enum):
    """Types of agents"""
    CONVERSATIONAL = "conversational"
    TASK_EXECUTOR = "task_executor"
    DATA_PROCESSOR = "data_processor"
    INTEGRATION = "integration"
    MONITORING = "monitoring"


# Pydantic models for API
class AgentCreate(BaseModel):
    """Agent creation model"""
    id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Agent name")
    model: str = Field(default="claude-3-haiku", description="AI model to use")
    specialization: str = Field(description="Agent specialization")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")


class AgentUpdate(BaseModel):
    """Agent update model"""
    name: Optional[str] = None
    model: Optional[str] = None
    specialization: Optional[str] = None
    status: Optional[str] = None
    capabilities: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None


class TaskAssignment(BaseModel):
    """Task assignment model"""
    description: str = Field(description="Task description")
    priority: str = Field(default="normal", description="Task priority")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")


class AgentDesignerService:
    """Main Agent Designer Service class - Integrated with shared database"""
    
    def __init__(self):
        self.app = FastAPI(title="Agent Designer Service (Integrated)", version="2.0.0")
        
        # Initialize Data Access Layer
        self.dal = DataAccessLayer("agent_designer")
        logger.info("✅ Connected to shared database and cache")
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
        
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            health_status = self.dal.health_check()
            return {
                "service": "agent_designer",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/agents")
        async def list_agents():
            """List all agents from shared database"""
            try:
                agents = self.dal.list_agents()
                return {
                    "agents": agents,
                    "count": len(agents),
                    "source": "shared_database"
                }
            except Exception as e:
                logger.error(f"Failed to list agents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str):
            """Get specific agent from shared database"""
            agent = self.dal.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            return agent
        
        @self.app.post("/agents")
        async def create_agent(agent_data: AgentCreate):
            """Create new agent in shared database"""
            try:
                # Check if agent already exists
                existing = self.dal.get_agent(agent_data.id)
                if existing:
                    raise HTTPException(status_code=409, detail=f"Agent {agent_data.id} already exists")
                
                # Create agent
                agent = self.dal.create_agent(agent_data.dict())
                logger.info(f"Created agent {agent['id']} in shared database")
                return agent
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to create agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/agents/{agent_id}")
        async def update_agent(agent_id: str, updates: AgentUpdate):
            """Update agent in shared database"""
            try:
                # Get update data, excluding None values
                update_data = {k: v for k, v in updates.dict().items() if v is not None}
                
                if not update_data:
                    raise HTTPException(status_code=400, detail="No updates provided")
                
                agent = self.dal.update_agent(agent_id, update_data)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                
                logger.info(f"Updated agent {agent_id} in shared database")
                return agent
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/agents/{agent_id}")
        async def delete_agent(agent_id: str):
            """Delete agent from shared database"""
            try:
                success = self.dal.delete_agent(agent_id)
                if not success:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                
                logger.info(f"Deleted agent {agent_id} from shared database")
                return {"message": f"Agent {agent_id} deleted successfully"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to delete agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/agents/{agent_id}/status")
        async def update_agent_status(agent_id: str, status: str):
            """Update agent status"""
            try:
                agent = self.dal.update_agent_status(agent_id, status)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                
                logger.info(f"Updated agent {agent_id} status to {status}")
                return {"agent_id": agent_id, "status": status}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update agent status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/agents/{agent_id}/tasks")
        async def assign_task(agent_id: str, task: TaskAssignment):
            """Assign task to agent"""
            try:
                # Verify agent exists
                agent = self.dal.get_agent(agent_id)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                
                # Create task
                task_data = task.dict()
                task_data['agent_id'] = agent_id
                created_task = self.dal.create_task(task_data)
                
                # Update agent status to busy
                self.dal.update_agent_status(agent_id, "busy")
                
                logger.info(f"Assigned task {created_task['id']} to agent {agent_id}")
                return created_task
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to assign task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents/{agent_id}/tasks")
        async def get_agent_tasks(agent_id: str):
            """Get tasks for specific agent"""
            try:
                tasks = self.dal.list_tasks(agent_id=agent_id)
                return {
                    "agent_id": agent_id,
                    "tasks": tasks,
                    "count": len(tasks)
                }
            except Exception as e:
                logger.error(f"Failed to get agent tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents/{agent_id}/knowledge")
        async def get_agent_knowledge(agent_id: str, category: Optional[str] = None):
            """Get knowledge base for agent"""
            try:
                knowledge = self.dal.get_agent_knowledge(agent_id, category)
                return {
                    "agent_id": agent_id,
                    "knowledge": knowledge,
                    "count": len(knowledge),
                    "category": category
                }
            except Exception as e:
                logger.error(f"Failed to get agent knowledge: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks")
        async def list_tasks(status: Optional[str] = None, agent_id: Optional[str] = None):
            """List all tasks with optional filtering"""
            try:
                tasks = self.dal.list_tasks(agent_id=agent_id, status=status)
                return {
                    "tasks": tasks,
                    "count": len(tasks)
                }
            except Exception as e:
                logger.error(f"Failed to list tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            """Get task details"""
            try:
                task = self.dal.get_task(task_id)
                if not task:
                    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
                return task
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents/{agent_id}/performance")
        async def get_agent_performance(agent_id: str):
            """Get agent performance metrics"""
            try:
                # Get completed tasks for this agent
                tasks = self.dal.list_tasks(agent_id=agent_id, status="completed")
                
                # Calculate metrics
                total_tasks = len(tasks)
                if total_tasks > 0:
                    avg_time = sum(
                        (datetime.fromisoformat(t['completed_at']) - datetime.fromisoformat(t['created_at'])).total_seconds()
                        for t in tasks if t.get('completed_at')
                    ) / total_tasks
                else:
                    avg_time = 0
                
                return {
                    "agent_id": agent_id,
                    "total_tasks": total_tasks,
                    "average_completion_time": avg_time,
                    "success_rate": 100.0 if total_tasks > 0 else 0.0
                }
            except Exception as e:
                logger.error(f"Failed to get agent performance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/agents/{agent_id}/knowledge")
        async def add_knowledge(agent_id: str, knowledge_data: Dict[str, Any]):
            """Add knowledge to agent"""
            try:
                # Verify agent exists
                agent = self.dal.get_agent(agent_id)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                
                # Add knowledge
                knowledge = self.dal.add_knowledge(agent_id, knowledge_data)
                logger.info(f"Added knowledge to agent {agent_id}")
                return knowledge
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to add knowledge: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/agents/execute")
        async def execute_task(request: Dict[str, Any]):
            """Execute a task with an agent"""
            try:
                task_description = request.get("task")
                agent_id = request.get("agent_id")
                context = request.get("context", {})
                timeout = request.get("timeout", 60)
                
                if not task_description:
                    raise HTTPException(status_code=400, detail="Task description is required")
                
                # Auto-select agent if not specified
                if not agent_id or agent_id == "auto":
                    # Simple auto-selection based on task keywords
                    agents = self.dal.list_agents()
                    available_agents = [a for a in agents if a['status'] == 'idle']
                    if available_agents:
                        agent_id = available_agents[0]['id']
                    else:
                        raise HTTPException(status_code=503, detail="No available agents")
                
                # Verify agent exists and is available
                agent = self.dal.get_agent(agent_id)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                
                if agent['status'] == 'busy':
                    raise HTTPException(status_code=409, detail=f"Agent {agent_id} is busy")
                
                # Create task
                task_data = {
                    "agent_id": agent_id,
                    "description": task_description,
                    "status": "pending",
                    "context": context
                }
                
                task = self.dal.create_task(task_data)
                
                # Update agent status to busy
                self.dal.update_agent_status(agent_id, "busy")
                
                # Execute task with real AI
                self.dal.update_task_status(task['id'], "started")
                
                # Import and use AI task executor
                from ai_task_executor import AITaskExecutor
                
                async def execute_with_ai():
                    try:
                        executor = AITaskExecutor()
                        result = await executor.execute_task(task['id'])
                        self.dal.update_agent_status(agent_id, "idle")
                    except Exception as e:
                        logger.error(f"Task execution failed: {e}")
                        self.dal.update_task_status(task['id'], "failed", {"error": str(e)})
                        self.dal.update_agent_status(agent_id, "idle")
                
                asyncio.create_task(execute_with_ai())
                
                logger.info(f"Task {task['id']} assigned to agent {agent_id}")
                
                return {
                    "task_id": task['id'],
                    "agent_id": agent_id,
                    "agent_name": agent['name'],
                    "status": "accepted",
                    "message": f"Task assigned to {agent['name']}"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to execute task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""
        
        def on_task_completed(event):
            """Handle task completion events"""
            try:
                task_id = event.data.get('task_id')
                # Get task details to find agent
                task = self.dal.get_task(task_id)
                if task and task['agent_id']:
                    # Update agent status back to idle
                    self.dal.update_agent_status(task['agent_id'], "idle")
                    logger.info(f"Agent {task['agent_id']} marked as idle after task completion")
            except Exception as e:
                logger.error(f"Error handling task completion: {e}")
        
        def on_task_failed(event):
            """Handle task failure events"""
            try:
                task_id = event.data.get('task_id')
                error = event.data.get('error')
                # Get task details to find agent
                task = self.dal.get_task(task_id)
                if task and task['agent_id']:
                    # Update agent status to error
                    self.dal.update_agent_status(task['agent_id'], "error")
                    logger.warning(f"Agent {task['agent_id']} marked as error: {error}")
            except Exception as e:
                logger.error(f"Error handling task failure: {e}")
        
        def on_workflow_started(event):
            """Handle workflow start events"""
            workflow_id = event.data.get('workflow_id')
            logger.info(f"Workflow {workflow_id} started - agents may be assigned")
        
        # Register event handlers
        self.dal.event_bus.on(EventChannel.TASK_COMPLETED, on_task_completed)
        self.dal.event_bus.on(EventChannel.TASK_FAILED, on_task_failed)
        self.dal.event_bus.on(EventChannel.WORKFLOW_STARTED, on_workflow_started)
        
        logger.info("Event handlers registered for cross-service communication")
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Agent Designer Service (Integrated) starting up...")
        # Verify database connection
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")
        
        # Load current agents
        agents = self.dal.list_agents()
        logger.info(f"Loaded {len(agents)} agents from shared database")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Agent Designer Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = AgentDesignerService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("AGENT_DESIGNER_PORT", 8002))
    logger.info(f"Starting Agent Designer Service (Integrated) on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()