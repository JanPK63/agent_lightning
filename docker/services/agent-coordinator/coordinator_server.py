"""
Enterprise Agent Coordinator Server
FastAPI server for multi-agent coordination
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise Agent Coordinator", version="1.0.0")

class Agent(BaseModel):
    id: str
    name: str
    type: str
    status: str = "idle"
    capabilities: List[str] = []
    current_task: Optional[str] = None
    last_activity: datetime

class Task(BaseModel):
    id: str
    description: str
    agent_id: Optional[str] = None
    status: str = "pending"
    priority: int = 5
    created_at: datetime
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class AgentRegistry:
    """Enterprise agent registry"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        
    def register_agent(self, agent: Agent):
        """Register new agent"""
        self.agents[agent.id] = agent
        
    def assign_task(self, task_id: str, agent_id: str):
        """Assign task to agent"""
        if task_id in self.tasks and agent_id in self.agents:
            self.tasks[task_id].agent_id = agent_id
            self.tasks[task_id].status = "assigned"
            self.tasks[task_id].assigned_at = datetime.utcnow()
            self.agents[agent_id].current_task = task_id
            self.agents[agent_id].status = "busy"
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
    
    def complete_task(self, task_id: str):
        """Mark task as completed"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            if task.agent_id and task.agent_id in self.agents:
                self.agents[task.agent_id].current_task = None
                self.agents[task.agent_id].status = "idle"
    
    def get_available_agents(self) -> List[Agent]:
        """Get list of available agents"""
        return [agent for agent in self.agents.values() if agent.status == "idle"]

# Global registry
registry = AgentRegistry()

class RegisterAgentRequest(BaseModel):
    name: str
    type: str
    capabilities: List[str] = []

class CreateTaskRequest(BaseModel):
    description: str
    priority: int = 5

class AssignTaskRequest(BaseModel):
    task_id: str
    agent_id: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agent-coordinator", "port": 8011}

@app.post("/agents/register")
async def register_agent(request: RegisterAgentRequest):
    """Register new agent"""
    try:
        agent_id = str(uuid.uuid4())
        agent = Agent(
            id=agent_id,
            name=request.name,
            type=request.type,
            capabilities=request.capabilities,
            last_activity=datetime.utcnow()
        )
        
        registry.register_agent(agent)
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "message": f"Agent {request.name} registered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    try:
        agents = [
            {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type,
                "status": agent.status,
                "capabilities": agent.capabilities,
                "current_task": agent.current_task,
                "last_activity": agent.last_activity
            }
            for agent in registry.agents.values()
        ]
        
        return {
            "status": "success",
            "agents": agents,
            "total_agents": len(agents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/available")
async def get_available_agents():
    """Get available agents"""
    try:
        available = registry.get_available_agents()
        agents = [
            {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type,
                "capabilities": agent.capabilities
            }
            for agent in available
        ]
        
        return {
            "status": "success",
            "available_agents": agents,
            "count": len(agents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks")
async def create_task(request: CreateTaskRequest):
    """Create new task"""
    try:
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            description=request.description,
            priority=request.priority,
            created_at=datetime.utcnow()
        )
        
        registry.tasks[task_id] = task
        registry.task_queue.append(task_id)
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Task created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/assign")
async def assign_task(request: AssignTaskRequest):
    """Assign task to agent"""
    try:
        if request.task_id not in registry.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if request.agent_id not in registry.agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        registry.assign_task(request.task_id, request.agent_id)
        
        return {
            "status": "success",
            "message": f"Task {request.task_id} assigned to agent {request.agent_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str):
    """Mark task as completed"""
    try:
        if task_id not in registry.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        registry.complete_task(task_id)
        
        return {
            "status": "success",
            "message": f"Task {task_id} marked as completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    try:
        tasks = [
            {
                "id": task.id,
                "description": task.description,
                "agent_id": task.agent_id,
                "status": task.status,
                "priority": task.priority,
                "created_at": task.created_at,
                "assigned_at": task.assigned_at,
                "completed_at": task.completed_at
            }
            for task in registry.tasks.values()
        ]
        
        return {
            "status": "success",
            "tasks": tasks,
            "total_tasks": len(tasks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get coordination metrics"""
    try:
        total_agents = len(registry.agents)
        available_agents = len(registry.get_available_agents())
        busy_agents = total_agents - available_agents
        
        total_tasks = len(registry.tasks)
        pending_tasks = len([t for t in registry.tasks.values() if t.status == "pending"])
        completed_tasks = len([t for t in registry.tasks.values() if t.status == "completed"])
        
        return {
            "status": "success",
            "metrics": {
                "total_agents": total_agents,
                "available_agents": available_agents,
                "busy_agents": busy_agents,
                "total_tasks": total_tasks,
                "pending_tasks": pending_tasks,
                "completed_tasks": completed_tasks,
                "queue_length": len(registry.task_queue)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🤖 Starting Enterprise Agent Coordinator on http://0.0.0.0:8011")
    uvicorn.run(app, host="0.0.0.0", port=8011)