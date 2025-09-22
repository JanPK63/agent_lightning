#!/usr/bin/env python3
"""
Simple Agent Lightning Server
Minimal working server for testing task execution
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import json
import uuid
from datetime import datetime

app = FastAPI(title="Agent Lightning Simple Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    task: str
    agent_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}
    workflow_type: Optional[str] = "sequential"
    timeout: Optional[int] = 60

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

# Simple in-memory task storage
tasks = {}

@app.get("/")
async def root():
    return {"message": "Agent Lightning Simple Server", "status": "running"}

@app.post("/api/v1/agents/execute", response_model=TaskResponse)
async def execute_agent_task(request: AgentRequest):
    """Execute a task with an agent"""
    task_id = str(uuid.uuid4())

    # Create task record
    tasks[task_id] = {
        "id": task_id,
        "status": "pending",
        "request": request.dict(),
        "created_at": datetime.now(),
        "result": None,
        "error": None
    }

    # Process task asynchronously
    asyncio.create_task(process_task(task_id))

    return TaskResponse(
        task_id=task_id,
        status="pending",
        metadata={"created_at": datetime.now().isoformat()}
    )

@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and result"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    return TaskResponse(
        task_id=task_id,
        status=task["status"],
        result=task["result"],
        error=task["error"],
        metadata={"created_at": task["created_at"].isoformat()}
    )

async def process_task(task_id: str):
    """Process a task (simplified)"""
    task_record = tasks[task_id]

    try:
        # Update status
        task_record["status"] = "processing"

        # Simulate processing time
        await asyncio.sleep(2)

        # Mock result
        task_record["result"] = {
            "output": f"Task '{task_record['request']['task']}' completed successfully",
            "execution_time": 2.0,
            "agent_model": "mock_agent"
        }
        task_record["status"] = "completed"

    except Exception as e:
        task_record["error"] = str(e)
        task_record["status"] = "failed"

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Agent Lightning Simple Server")
    print("Endpoints:")
    print("  http://localhost:8000/")
    print("  POST /api/v1/agents/execute")
    print("  GET /api/v1/tasks/{task_id}")
    uvicorn.run(app, host="0.0.0.0", port=8000)