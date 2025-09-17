"""
Internet-Enabled Agent API
All agents now have real internet access and web browsing capabilities
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from jwt_auth import get_current_user
from web_access_tool import web_tool
import time
import uuid
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Internet-Enabled Agent API", version="4.0")

class TaskRequest(BaseModel):
    task: str
    agent_id: Optional[str] = None
    model: str = "gpt-4o"
    enable_internet: bool = True

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: str
    agent_id: str
    internet_used: bool
    execution_time: float
    metadata: Dict[str, Any]

def process_with_internet(task: str, agent_id: str) -> tuple[str, bool]:
    """Process task with internet access"""
    internet_used = False
    
    # Always enable internet access for agents - they should have web browsing capabilities
    try:
        try:
            # Extract search query from task
            if 'current' in task.lower() or 'latest' in task.lower():
                search_query = task.replace('current', '').replace('latest', '').strip()
            elif 'search for' in task.lower():
                search_query = task.split('search for')[-1].strip()
            elif 'look up' in task.lower():
                search_query = task.split('look up')[-1].strip()
            else:
                search_query = task
            
        # Get current information from the web
        web_info = web_tool.get_current_info(search_query)
        internet_used = True
        
        # Generate response based on agent type with web data
        if agent_id == 'data_scientist':
            result = f"Based on current web data:\n{web_info}\n\nAs a data scientist, I can analyze this information and provide insights."
        elif agent_id == 'security_expert':
            result = f"Current security information:\n{web_info}\n\nSecurity assessment: I've accessed real-time data to provide current security insights."
        elif agent_id == 'devops_engineer':
            result = f"Latest DevOps information:\n{web_info}\n\nInfrastructure recommendations based on current industry trends."
        else:
            result = f"Current information from the web:\n{web_info}\n\nI have successfully browsed the internet to get you the latest information."
        
    except Exception as e:
        # Fallback with internet capability mention
        result = f"I have internet browsing capabilities but encountered an issue: {str(e)}. I can search the web, browse websites, and get current information. Please try rephrasing your request."
        internet_used = False
    
    return result, internet_used

@app.get("/health")
async def health_check():
    """Health check with internet connectivity test"""
    try:
        # Test internet connectivity
        test_result = web_tool.search_web("test", 1)
        internet_ok = not (test_result and test_result[0].get('error'))
    except:
        internet_ok = False
    
    return {
        "status": "healthy",
        "internet_access": internet_ok,
        "service": "internet_agent_api"
    }

@app.get("/agents")
async def list_agents(current_user: dict = Depends(get_current_user)):
    """List all internet-enabled agents"""
    agents = [
        {"id": "security_expert", "name": "Security Expert", "internet_enabled": True},
        {"id": "data_scientist", "name": "Data Scientist", "internet_enabled": True},
        {"id": "devops_engineer", "name": "DevOps Engineer", "internet_enabled": True},
        {"id": "full_stack_developer", "name": "Full Stack Developer", "internet_enabled": True},
        {"id": "system_architect", "name": "System Architect", "internet_enabled": True},
        {"id": "blockchain_developer", "name": "Blockchain Developer", "internet_enabled": True},
        {"id": "researcher", "name": "Researcher", "internet_enabled": True},
        {"id": "test_engineer", "name": "Test Engineer", "internet_enabled": True}
    ]
    
    return {
        "agents": agents,
        "total": len(agents),
        "all_internet_enabled": True,
        "user": current_user["username"]
    }

@app.post("/execute", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest,
    current_user: dict = Depends(get_current_user)
):
    """Execute task with internet access"""
    start_time = time.time()
    
    if not request.agent_id:
        request.agent_id = "researcher"  # Default to researcher for internet tasks
    
    try:
        task_id = str(uuid.uuid4())[:8]
        
        # Process with internet if enabled
        if request.enable_internet:
            result, internet_used = process_with_internet(request.task, request.agent_id)
        else:
            result = f"[OFFLINE MODE] {request.task} - Internet access disabled for this request"
            internet_used = False
        
        execution_time = time.time() - start_time
        
        response_data = {
            "task_id": task_id,
            "status": "completed",
            "result": result,
            "agent_id": request.agent_id,
            "internet_used": internet_used,
            "execution_time": execution_time,
            "metadata": {
                "model": request.model,
                "user": current_user["username"],
                "internet_enabled": request.enable_internet,
                "timestamp": time.time()
            }
        }
        
        return TaskResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-internet")
async def test_internet_access():
    """Test internet connectivity"""
    try:
        # Test web search
        search_results = web_tool.search_web("artificial intelligence", 2)
        
        # Test webpage fetch
        webpage_result = web_tool.fetch_webpage("https://httpbin.org/json")
        
        return {
            "search_test": "success" if search_results and not search_results[0].get('error') else "failed",
            "webpage_test": webpage_result.get('status', 'failed'),
            "search_results": search_results[:1],  # Show first result
            "message": "Internet access is working!" if search_results else "Internet access failed"
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Internet access test failed"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8892)