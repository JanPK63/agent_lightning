#!/usr/bin/env python3
"""
Enterprise Agent Coordinator Service
Manages multi-agent coordination, task distribution, and agent lifecycle
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
from enum import Enum
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="Agent Coordinator Service", version="1.0.0")

class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    TRAINING = "training"
    ERROR = "error"
    OFFLINE = "offline"

class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class Agent(BaseModel):
    id: str
    type: str
    status: AgentStatus = AgentStatus.IDLE
    capabilities: List[str] = []
    current_task: Optional[str] = None
    last_heartbeat: str = None

class Task(BaseModel):
    id: str
    type: str
    priority: TaskPriority = TaskPriority.NORMAL
    description: str
    requirements: List[str] = []
    assigned_agent: Optional[str] = None
    status: str = "pending"

class TaskAssignment(BaseModel):
    task_id: str
    agent_id: str

# Enterprise coordination state
agents = {}
tasks = {}
task_queue = []
coordination_metrics = {
    "total_agents": 0,
    "active_agents": 0,
    "completed_tasks": 0,
    "failed_tasks": 0
}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agent-coordinator"}

@app.post("/agents/register")
async def register_agent(agent: Agent):
    agent.last_heartbeat = datetime.now().isoformat()
    agents[agent.id] = agent.dict()
    coordination_metrics["total_agents"] = len(agents)
    return {"status": "registered", "agent_id": agent.id}

@app.post("/agents/{agent_id}/heartbeat")
async def agent_heartbeat(agent_id: str, status: AgentStatus):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agents[agent_id]["status"] = status
    agents[agent_id]["last_heartbeat"] = datetime.now().isoformat()
    
    # Update active agents count
    coordination_metrics["active_agents"] = sum(
        1 for agent in agents.values() 
        if agent["status"] not in [AgentStatus.OFFLINE, AgentStatus.ERROR]
    )
    
    return {"status": "updated"}

@app.post("/tasks/submit")
async def submit_task(task: Task):
    task.id = f"task_{len(tasks)}"
    tasks[task.id] = task.dict()
    task_queue.append(task.id)
    
    # Try to assign immediately
    await try_assign_task(task.id)
    
    return {"task_id": task.id, "status": "submitted"}

@app.post("/tasks/assign")
async def assign_task(assignment: TaskAssignment):
    if assignment.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if assignment.agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check agent availability
    agent = agents[assignment.agent_id]
    if agent["status"] != AgentStatus.IDLE:
        raise HTTPException(status_code=400, detail="Agent not available")
    
    # Assign task
    tasks[assignment.task_id]["assigned_agent"] = assignment.agent_id
    tasks[assignment.task_id]["status"] = "assigned"
    agents[assignment.agent_id]["current_task"] = assignment.task_id
    agents[assignment.agent_id]["status"] = AgentStatus.BUSY
    
    return {"status": "assigned"}

@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str, success: bool = True):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    task["status"] = "completed" if success else "failed"
    
    # Free up agent
    if task["assigned_agent"]:
        agent_id = task["assigned_agent"]
        if agent_id in agents:
            agents[agent_id]["current_task"] = None
            agents[agent_id]["status"] = AgentStatus.IDLE
    
    # Update metrics
    if success:
        coordination_metrics["completed_tasks"] += 1
    else:
        coordination_metrics["failed_tasks"] += 1
    
    return {"status": "updated"}

@app.get("/coordination/status")
async def get_coordination_status():
    return {
        "metrics": coordination_metrics,
        "agents": len(agents),
        "pending_tasks": len([t for t in tasks.values() if t["status"] == "pending"]),
        "active_tasks": len([t for t in tasks.values() if t["status"] == "assigned"])
    }

async def try_assign_task(task_id: str):
    """Attempt to automatically assign a task to an available agent"""
    if task_id not in tasks:
        return
    
    task = tasks[task_id]
    if task["status"] != "pending":
        return
    
    # Find suitable agent
    for agent_id, agent in agents.items():
        if agent["status"] == AgentStatus.IDLE:
            # Check if agent has required capabilities
            if all(req in agent["capabilities"] for req in task["requirements"]):
                # Assign task
                task["assigned_agent"] = agent_id
                task["status"] = "assigned"
                agent["current_task"] = task_id
                agent["status"] = AgentStatus.BUSY
                break

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)