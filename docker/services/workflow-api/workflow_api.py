"""
Enterprise Workflow API Gateway
FastAPI gateway for workflow management services
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise Workflow API Gateway", version="1.0.0")

# Service endpoints
WORKFLOW_ENGINE_URL = "http://workflow-engine:8013"
MEMORY_MANAGER_URL = "http://memory-manager:8012"

class WorkflowRequest(BaseModel):
    name: str
    description: str
    tasks: List[Dict[str, Any]]
    created_by: str = "api"

class TaskExecutionRequest(BaseModel):
    workflow_id: str
    task_id: str
    agent_name: str
    prompt: str

class MemoryRequest(BaseModel):
    agent_id: str
    content: Dict[str, Any]
    memory_type: str
    importance: float = 0.5

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "workflow-api", "port": 8004}

@app.post("/workflows")
async def create_workflow(request: WorkflowRequest):
    """Create new workflow via workflow engine"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{WORKFLOW_ENGINE_URL}/workflows",
                json=request.dict(),
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Workflow engine unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow status"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{WORKFLOW_ENGINE_URL}/workflows/{workflow_id}",
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Workflow engine unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/execute")
async def execute_task(request: TaskExecutionRequest):
    """Execute task via workflow engine"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{WORKFLOW_ENGINE_URL}/tasks/execute",
                json=request.dict(),
                timeout=60.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Workflow engine unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List available agents"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{WORKFLOW_ENGINE_URL}/agents",
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Workflow engine unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/store")
async def store_memory(request: MemoryRequest):
    """Store memory via memory manager"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEMORY_MANAGER_URL}/memory/store",
                json=request.dict(),
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Memory manager unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/statistics/{agent_id}")
async def get_memory_statistics(agent_id: str):
    """Get memory statistics for agent"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEMORY_MANAGER_URL}/memory/statistics/{agent_id}",
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Memory manager unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/status")
async def get_system_status():
    """Get overall system status"""
    try:
        status = {"workflow_engine": "unknown", "memory_manager": "unknown"}
        
        # Check workflow engine
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{WORKFLOW_ENGINE_URL}/health", timeout=5.0)
                status["workflow_engine"] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            status["workflow_engine"] = "unavailable"
        
        # Check memory manager
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{MEMORY_MANAGER_URL}/health", timeout=5.0)
                status["memory_manager"] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            status["memory_manager"] = "unavailable"
        
        overall_status = "healthy" if all(s == "healthy" for s in status.values()) else "degraded"
        
        return {
            "status": overall_status,
            "services": status,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_system_metrics():
    """Get aggregated system metrics"""
    try:
        metrics = {}
        
        # Get workflow engine metrics
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{WORKFLOW_ENGINE_URL}/metrics", timeout=10.0)
                if response.status_code == 200:
                    metrics["workflow_engine"] = response.json()
        except:
            metrics["workflow_engine"] = {"error": "unavailable"}
        
        # Get memory manager metrics
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{MEMORY_MANAGER_URL}/metrics", timeout=10.0)
                if response.status_code == 200:
                    metrics["memory_manager"] = response.json()
        except:
            metrics["memory_manager"] = {"error": "unavailable"}
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🌐 Starting Enterprise Workflow API Gateway on http://0.0.0.0:8004")
    uvicorn.run(app, host="0.0.0.0", port=8004)