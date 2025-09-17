"""
Enterprise Workflow Engine Server
FastAPI server for workflow management
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
from datetime import datetime

from langchain_agent_wrapper import LangChainAgentManager
from shared.database import db_manager
from redis_manager import redis_manager

logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise Workflow Engine", version="1.0.0")

# Initialize managers
agent_manager = LangChainAgentManager()

class WorkflowRequest(BaseModel):
    name: str
    description: str
    tasks: List[Dict[str, Any]]
    created_by: str = "system"

class TaskRequest(BaseModel):
    workflow_id: str
    task_id: str
    agent_name: str
    prompt: str

@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    try:
        await db_manager.initialize()
        await redis_manager.initialize()
        logger.info("Workflow Engine services initialized")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "workflow-engine", "port": 8013}

@app.post("/workflows")
async def create_workflow(request: WorkflowRequest):
    """Create new workflow"""
    try:
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        workflow_data = {
            "workflow_id": workflow_id,
            "name": request.name,
            "description": request.description,
            "status": "pending",
            "created_by": request.created_by,
            "created_at": datetime.utcnow(),
            "tasks": request.tasks
        }
        
        await redis_manager.set_json(f"workflow:{workflow_id}", workflow_data, ttl=86400)
        
        return {"workflow_id": workflow_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow status"""
    try:
        workflow_data = await redis_manager.get_json(f"workflow:{workflow_id}")
        if not workflow_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/execute")
async def execute_task(request: TaskRequest):
    """Execute single task"""
    try:
        agent = agent_manager.get_agent(request.agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_name} not found")
        
        result = agent.invoke(request.prompt, request.task_id)
        
        return {
            "task_id": request.task_id,
            "agent": request.agent_name,
            "result": result,
            "status": "completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List available agents"""
    try:
        agents = agent_manager.list_agents()
        return {"agents": agents, "total": len(agents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        metrics = agent_manager.get_system_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🚀 Starting Enterprise Workflow Engine on http://0.0.0.0:8013")
    uvicorn.run(app, host="0.0.0.0", port=8013)