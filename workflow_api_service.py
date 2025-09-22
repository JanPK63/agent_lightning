"""
Enterprise Workflow API Service
REST API for workflow management with monitoring and control
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import logging

from enterprise_workflow_engine import EnterpriseWorkflowEngine, ENTERPRISE_WORKFLOWS

from monitoring.http_metrics_middleware import add_http_metrics_middleware
from enterprise_workflow_engine import EnterpriseWorkflowEngine, ENTERPRISE_WORKFLOWS

logger = logging.getLogger(__name__)


class WorkflowCreateRequest(BaseModel):
    name: str
    description: str
    tasks: List[Dict[str, Any]]
    created_by: str = "system"


class WorkflowExecuteRequest(BaseModel):
    workflow_id: str
    async_execution: bool = True
app = FastAPI(title="Enterprise Workflow API", version="1.0.0")

# Add HTTP metrics middleware for automatic request/response monitoring
app = add_http_metrics_middleware(app, service_name="workflow_api")


app = FastAPI(title="Enterprise Workflow API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global workflow engine
workflow_engine = EnterpriseWorkflowEngine()


@app.on_event("startup")
async def startup():
    logger.info("Enterprise Workflow API starting up...")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "enterprise_workflow_api",
        "status": "healthy",
        "metrics": workflow_engine.get_system_metrics()
    }


@app.post("/workflows")
async def create_workflow(request: WorkflowCreateRequest):
    """Create new workflow"""
    try:
        workflow_id = await workflow_engine.create_workflow(
            name=request.name,
            description=request.description,
            tasks=request.tasks,
            created_by=request.created_by
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "created",
            "message": f"Workflow '{request.name}' created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, background_tasks: BackgroundTasks):
    """Execute workflow"""
    try:
        if workflow_id not in workflow_engine.active_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Execute in background
        background_tasks.add_task(workflow_engine.execute_workflow, workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "status": "execution_started",
            "message": "Workflow execution started in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow status"""
    try:
        status = await workflow_engine.get_workflow_status(workflow_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows/{workflow_id}/tasks")
async def get_workflow_tasks(workflow_id: str):
    """Get workflow tasks with detailed status"""
    try:
        status = await workflow_engine.get_workflow_status(workflow_id)
        if "error" in status:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "workflow_id": workflow_id,
            "tasks": status.get("tasks", []),
            "progress": status.get("progress", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/templates")
async def get_workflow_templates():
    """Get available workflow templates"""
    return {
        "templates": ENTERPRISE_WORKFLOWS,
        "count": len(ENTERPRISE_WORKFLOWS)
    }


@app.post("/templates/{template_name}")
async def create_from_template(template_name: str, created_by: str = "system"):
    """Create workflow from template"""
    try:
        if template_name not in ENTERPRISE_WORKFLOWS:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template = ENTERPRISE_WORKFLOWS[template_name]
        workflow_id = await workflow_engine.create_workflow(
            name=template["name"],
            description=template["description"],
            tasks=template["tasks"],
            created_by=created_by
        )
        
        return {
            "workflow_id": workflow_id,
            "template": template_name,
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def get_available_agents():
    """Get available agents and their capabilities"""
    agents_info = {}
    for name, agent in workflow_engine.agent_manager.agents.items():
        agents_info[name] = {
            "name": name,
            "description": agent.agent_config.description,
            "tools": [tool.name for tool in agent.tools],
            "load": workflow_engine.agent_pool.agent_load.get(name, 0),
            "performance": workflow_engine.agent_pool.agent_performance.get(name, {})
        }
    
    return {
        "agents": agents_info,
        "total_count": len(agents_info)
    }


@app.get("/metrics")
async def get_system_metrics():
    """Get comprehensive system metrics"""
    return workflow_engine.get_system_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030, log_level="info")