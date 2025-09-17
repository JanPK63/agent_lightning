"""
Secure Agent API with JWT Authentication
Enhanced agent API with proper authentication and authorization
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from jwt_auth import get_current_user, require_admin, require_developer
from cached_agent_api import TaskRequest, TaskResponse, generate_cache_key, get_cached_response, cache_response
from database_connection import get_db_session
from shared.models import Agent, Task
import time
import uuid
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Secure Agent API", version="3.0")

@app.get("/health")
async def health_check():
    """Public health check"""
    return {"status": "healthy", "service": "secure_agent_api"}

@app.get("/agents")
async def list_agents(current_user: dict = Depends(get_current_user)):
    """List agents (authenticated users only)"""
    try:
        with get_db_session() as session:
            agents = session.query(Agent).all()
            return {
                "agents": [agent.to_dict() for agent in agents],
                "user": current_user["username"]
            }
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/execute", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest, 
    current_user: dict = Depends(get_current_user)
):
    """Execute task (authenticated users only)"""
    start_time = time.time()
    
    # Select agent if not specified
    if not request.agent_id:
        request.agent_id = "full_stack_developer"
    
    # Generate cache key including user context
    cache_key = f"user:{current_user['user_id']}:" + generate_cache_key(
        request.task, request.agent_id, request.model
    )
    
    # Check cache
    if request.use_cache:
        cached_result = get_cached_response(cache_key)
        if cached_result:
            cached_result["cached"] = True
            cached_result["execution_time"] = time.time() - start_time
            return TaskResponse(**cached_result)
    
    # Execute task
    try:
        task_id = str(uuid.uuid4())[:8]
        
        # Enhanced task processing based on user role
        if current_user["role"] == "admin":
            result = f"[ADMIN ACCESS] Advanced processing: {request.task}"
        elif current_user["role"] == "developer":
            result = f"[DEV ACCESS] Code generation: {request.task}"
        else:
            result = f"[USER ACCESS] Basic processing: {request.task}"
        
        execution_time = time.time() - start_time
        
        response_data = {
            "task_id": task_id,
            "status": "completed",
            "result": result,
            "agent_id": request.agent_id,
            "cached": False,
            "execution_time": execution_time,
            "metadata": {
                "model": request.model,
                "user": current_user["username"],
                "role": current_user["role"],
                "timestamp": time.time()
            }
        }
        
        # Cache response
        if request.use_cache:
            cache_response(cache_key, response_data, 1800)
        
        # Store in database with user context
        try:
            with get_db_session() as session:
                task = Task(
                    id=uuid.uuid4(),
                    agent_id=request.agent_id,
                    description=request.task,
                    status="completed",
                    result={"output": result, "user": current_user["username"]},
                    context={"model": request.model, "user_id": current_user["user_id"]}
                )
                session.add(task)
                session.commit()
        except Exception as e:
            logger.error(f"Database save error: {e}")
        
        return TaskResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
async def list_user_tasks(current_user: dict = Depends(get_current_user)):
    """List user's tasks"""
    try:
        with get_db_session() as session:
            # Users see only their tasks, admins see all
            if current_user["role"] == "admin":
                tasks = session.query(Task).limit(50).all()
            else:
                tasks = session.query(Task).filter(
                    Task.context.contains({"user_id": current_user["user_id"]})
                ).limit(20).all()
            
            return {
                "tasks": [task.to_dict() for task in tasks],
                "count": len(tasks)
            }
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/agents", dependencies=[Depends(require_admin)])
async def create_agent(agent_data: dict, current_user: dict = Depends(require_admin)):
    """Create new agent (admin only)"""
    try:
        with get_db_session() as session:
            agent = Agent(
                id=agent_data["id"],
                name=agent_data["name"],
                specialization=agent_data.get("specialization"),
                model=agent_data.get("model", "gpt-4o"),
                status="active"
            )
            session.add(agent)
            session.commit()
            
            return {"message": f"Agent {agent.id} created", "agent": agent.to_dict()}
    except Exception as e:
        logger.error(f"Agent creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create agent")

@app.get("/admin/stats", dependencies=[Depends(require_admin)])
async def admin_stats(current_user: dict = Depends(require_admin)):
    """Admin statistics"""
    try:
        with get_db_session() as session:
            agent_count = session.query(Agent).count()
            task_count = session.query(Task).count()
            
            return {
                "agents": agent_count,
                "tasks": task_count,
                "admin": current_user["username"]
            }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8891)