"""
Cached Agent API - Enhanced version with Redis caching
Improves performance by caching agent responses and knowledge
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import hashlib
import time
from redis_manager import redis_manager
from shared.database import db_manager
from shared.models import Agent, Knowledge, Task
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Cached Agent API", version="2.0")

class TaskRequest(BaseModel):
    task: str
    agent_id: Optional[str] = None
    model: str = "gpt-4o"
    use_cache: bool = True

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: str
    agent_id: str
    cached: bool = False
    execution_time: float
    metadata: Dict[str, Any]

def generate_cache_key(task: str, agent_id: str, model: str) -> str:
    """Generate cache key for task"""
    content = f"{task}:{agent_id}:{model}"
    return f"task_cache:{hashlib.md5(content.encode()).hexdigest()}"

def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached task response"""
    if not redis_manager.health_check():
        return None
    return redis_manager.get_cache(cache_key)

def cache_response(cache_key: str, response: Dict, ttl: int = 3600):
    """Cache task response"""
    if redis_manager.health_check():
        redis_manager.set_cache(cache_key, response, ttl)

@app.get("/health")
async def health_check():
    """Health check with cache status"""
    return {
        "status": "healthy",
        "redis_connected": redis_manager.health_check(),
        "timestamp": time.time()
    }

@app.get("/agents")
async def list_agents():
    """List all agents with caching"""
    cache_key = "agents_list"
    
    # Try cache first
    cached_agents = redis_manager.get_cache(cache_key)
    if cached_agents:
        return {"agents": cached_agents, "cached": True}
    
    # Get from database
    try:
        with db_manager.get_db() as session:
            agents = session.query(Agent).all()
            agent_list = [agent.to_dict() for agent in agents]
            
            # Cache for 5 minutes
            redis_manager.set_cache(cache_key, agent_list, 300)
            
            return {"agents": agent_list, "cached": False}
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute task with caching"""
    start_time = time.time()
    
    # Select agent if not specified
    if not request.agent_id:
        request.agent_id = "full_stack_developer"  # Default agent
    
    # Generate cache key
    cache_key = generate_cache_key(request.task, request.agent_id, request.model)
    
    # Check cache if enabled
    if request.use_cache:
        cached_result = get_cached_response(cache_key)
        if cached_result:
            cached_result["cached"] = True
            cached_result["execution_time"] = time.time() - start_time
            return TaskResponse(**cached_result)
    
    # Execute task (simplified for demo)
    try:
        # Simulate AI processing
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        # Simple task processing based on agent type
        if "calculate" in request.task.lower() or "math" in request.task.lower():
            result = f"Calculation result: {request.task}"
        elif "code" in request.task.lower() or "function" in request.task.lower():
            result = f"```python\n# Generated code for: {request.task}\ndef solution():\n    # Implementation here\n    pass\n```"
        else:
            result = f"Task completed: {request.task}"
        
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
                "timestamp": time.time()
            }
        }
        
        # Cache the response
        if request.use_cache:
            cache_response(cache_key, response_data, 1800)  # 30 minutes
        
        # Store in database
        try:
            with db_manager.get_db() as session:
                task = Task(
                    id=task_id,
                    agent_id=request.agent_id,
                    description=request.task,
                    status="completed",
                    result={"output": result},
                    context={"model": request.model}
                )
                session.add(task)
                session.commit()
        except Exception as e:
            logger.error(f"Database save error: {e}")
        
        return TaskResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    if not redis_manager.health_check():
        return {"error": "Redis not available"}
    
    try:
        info = redis_manager.client.info()
        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
        }
    except Exception as e:
        return {"error": str(e)}

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cache"""
    if not redis_manager.health_check():
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        redis_manager.client.flushdb()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)