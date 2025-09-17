"""
Production API for Agent Lightning
REST and gRPC endpoints for production deployment
Provides programmatic access to all Agent Lightning features
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import json
import uuid
import time
from datetime import datetime
from enum import Enum
import grpc
from concurrent import futures
import numpy as np
from pathlib import Path
import jwt
from passlib.context import CryptContext
import redis
from collections import defaultdict
import logging

# Import Agent Lightning components
from mdp_agents import MDPAgent, AgentState, AgentAction
from multi_agent_system import MultiAgentSystem
from orchestration_workflows import create_workflow, WorkflowType, WorkflowTask
from reward_functions import RewardCalculator
from memory_manager import MemoryManager
from curriculum_learning import CurriculumLearning
from meta_learning import MetaLearner
from prompt_optimization import PromptOptimizer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRequest(BaseModel):
    task: str = Field(..., description="Task for agent to perform")
    agent_id: Optional[str] = Field(None, description="Specific agent ID")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    workflow_type: Optional[str] = Field("sequential", description="Workflow type")
    timeout: Optional[int] = Field(60, description="Timeout in seconds")


class TrainingRequest(BaseModel):
    dataset_path: Optional[str] = Field(None, description="Path to training dataset")
    num_epochs: Optional[int] = Field(10, description="Number of training epochs")
    batch_size: Optional[int] = Field(32, description="Batch size")
    learning_rate: Optional[float] = Field(0.001, description="Learning rate")
    distributed: Optional[bool] = Field(False, description="Use distributed training")


class PromptOptimizationRequest(BaseModel):
    base_prompt: str = Field(..., description="Base prompt to optimize")
    task_type: str = Field(..., description="Type of task")
    examples: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    optimization_method: Optional[str] = Field("evolutionary", description="Optimization method")


class ModelDeployRequest(BaseModel):
    model_id: str = Field(..., description="Model ID to deploy")
    deployment_name: str = Field(..., description="Deployment name")
    replicas: Optional[int] = Field(1, description="Number of replicas")
    resources: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuthToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


# FastAPI application
app = FastAPI(
    title="Agent Lightning API",
    description="Production API for Agent Lightning Framework",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProductionAPI:
    """
    Main production API service for Agent Lightning
    Handles REST endpoints, WebSocket connections, and gRPC services
    """
    
    def __init__(self):
        # Core components
        self.agents = {}
        self.multi_agent_system = MultiAgentSystem()
        self.memory_manager = MemoryManager()
        self.reward_calculator = RewardCalculator()
        self.curriculum_learning = CurriculumLearning()
        self.meta_learner = MetaLearner()
        self.prompt_optimizer = PromptOptimizer()
        
        # Task management
        self.tasks = {}
        self.task_queue = asyncio.Queue()
        self.active_connections = []
        
        # Authentication
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = "your-secret-key-here"  # In production, use environment variable
        
        # Rate limiting
        self.rate_limiter = defaultdict(list)
        self.max_requests_per_minute = 60
        
        # Caching
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except:
            self.redis_client = None
            logger.warning("Redis not available, caching disabled")
        
        logger.info("🚀 Production API initialized")
    
    async def create_agent(self, agent_id: str, role: str = "Assistant") -> MDPAgent:
        """Create and register an agent"""
        if agent_id not in self.agents:
            agent = MDPAgent(role=role)
            self.agents[agent_id] = agent
            logger.info(f"Created agent: {agent_id} with role: {role}")
        return self.agents[agent_id]
    
    async def process_task(self, request: AgentRequest) -> TaskResponse:
        """Process a task request"""
        task_id = str(uuid.uuid4())
        
        # Create task record
        self.tasks[task_id] = {
            "id": task_id,
            "status": TaskStatus.PENDING,
            "request": request,
            "created_at": datetime.now(),
            "result": None,
            "error": None
        }
        
        # Add to queue
        await self.task_queue.put(task_id)
        
        # Process in background
        asyncio.create_task(self._execute_task(task_id))
        
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            metadata={"created_at": datetime.now().isoformat()}
        )
    
    async def _execute_task(self, task_id: str):
        """Execute a task asynchronously"""
        task_record = self.tasks[task_id]
        request = task_record["request"]
        
        try:
            # Update status
            task_record["status"] = TaskStatus.PROCESSING
            await self._notify_clients(task_id, TaskStatus.PROCESSING)
            
            # Get or create agent
            agent_id = request.agent_id or "default_agent"
            agent = await self.create_agent(agent_id)
            
            # Create workflow
            workflow_type = WorkflowType[request.workflow_type.upper()]
            workflow = create_workflow(
                workflow_type,
                {agent_id: agent},
                memory_manager=self.memory_manager
            )
            
            # Create workflow task
            workflow_task = WorkflowTask(
                task_id=task_id,
                task_type="user_request",
                input_data={"task": request.task, "context": request.context},
                required_agents=[agent_id]
            )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                workflow.execute(workflow_task),
                timeout=request.timeout
            )
            
            # Process result
            task_record["result"] = {
                "output": result.results,
                "execution_time": result.execution_time,
                "transitions": len(result.transitions)
            }
            task_record["status"] = TaskStatus.COMPLETED
            
            # Cache result if Redis available
            if self.redis_client:
                self.redis_client.setex(
                    f"task:{task_id}",
                    3600,  # 1 hour TTL
                    json.dumps(task_record["result"])
                )
            
            # Notify clients
            await self._notify_clients(task_id, TaskStatus.COMPLETED)
            
        except asyncio.TimeoutError:
            task_record["error"] = "Task execution timeout"
            task_record["status"] = TaskStatus.FAILED
            await self._notify_clients(task_id, TaskStatus.FAILED)
            
        except Exception as e:
            task_record["error"] = str(e)
            task_record["status"] = TaskStatus.FAILED
            await self._notify_clients(task_id, TaskStatus.FAILED)
            logger.error(f"Task {task_id} failed: {e}")
    
    async def _notify_clients(self, task_id: str, status: TaskStatus):
        """Notify WebSocket clients of task status change"""
        message = json.dumps({
            "task_id": task_id,
            "status": status.value,
            "timestamp": datetime.now().isoformat()
        })
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass  # Connection might be closed
    
    async def train_model(self, request: TrainingRequest) -> Dict[str, Any]:
        """Train a model"""
        training_id = str(uuid.uuid4())
        
        # Start training in background
        asyncio.create_task(self._run_training(training_id, request))
        
        return {
            "training_id": training_id,
            "status": "started",
            "config": request.dict()
        }
    
    async def _run_training(self, training_id: str, request: TrainingRequest):
        """Run model training"""
        try:
            # Simulate training process
            logger.info(f"Starting training {training_id}")
            
            # In production, would integrate with actual training pipeline
            for epoch in range(request.num_epochs):
                await asyncio.sleep(1)  # Simulate training time
                
                # Log progress
                progress = {
                    "epoch": epoch + 1,
                    "total_epochs": request.num_epochs,
                    "loss": np.random.random(),
                    "accuracy": np.random.random()
                }
                
                # Cache progress
                if self.redis_client:
                    self.redis_client.setex(
                        f"training:{training_id}:progress",
                        300,
                        json.dumps(progress)
                    )
            
            logger.info(f"Training {training_id} completed")
            
        except Exception as e:
            logger.error(f"Training {training_id} failed: {e}")
    
    async def optimize_prompt(self, request: PromptOptimizationRequest) -> Dict[str, Any]:
        """Optimize a prompt"""
        # Use prompt optimizer
        if request.optimization_method == "evolutionary":
            # Find matching template
            template = self.prompt_optimizer.templates[0]  # Simplified
            
            # Create test tasks
            test_tasks = []
            for example in request.examples[:3]:
                test_tasks.append({
                    "type": request.task_type,
                    "input": example.get("input", ""),
                    "expected_output": example.get("output", "")
                })
            
            # Optimize
            optimized = self.prompt_optimizer.optimize_prompt_evolutionary(
                template, test_tasks
            )
            
            return {
                "original_prompt": request.base_prompt,
                "optimized_prompt": optimized.variation,
                "performance_score": optimized.performance,
                "optimization_method": request.optimization_method
            }
        
        else:
            # Use auto prompt engineering
            optimized = self.prompt_optimizer.auto_prompt_engineer(
                request.base_prompt,
                request.examples
            )
            
            return {
                "original_prompt": request.base_prompt,
                "optimized_prompt": optimized,
                "optimization_method": "auto_engineering"
            }
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        
        # Clean old requests
        self.rate_limiter[client_id] = [
            t for t in self.rate_limiter[client_id]
            if current_time - t < 60
        ]
        
        # Check limit
        if len(self.rate_limiter[client_id]) >= self.max_requests_per_minute:
            return False
        
        # Add current request
        self.rate_limiter[client_id].append(current_time)
        return True
    
    def generate_token(self, user_id: str) -> str:
        """Generate JWT token"""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow().timestamp() + 3600  # 1 hour expiry
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload.get("user_id")
        except:
            return None


# Initialize API service
api_service = ProductionAPI()


# REST Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Agent Lightning API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/api/docs",
            "health": "/health",
            "agents": "/api/v1/agents",
            "tasks": "/api/v1/tasks"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "operational",
            "redis": "operational" if api_service.redis_client else "unavailable",
            "agents": len(api_service.agents)
        }
    }


@app.post("/api/v1/agents/execute", response_model=TaskResponse)
async def execute_agent_task(request: AgentRequest):
    """Execute a task with an agent"""
    # Check rate limit
    client_id = "default"  # In production, extract from auth token
    if not api_service.check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return await api_service.process_task(request)


@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and result"""
    if task_id not in api_service.tasks:
        # Check cache
        if api_service.redis_client:
            cached = api_service.redis_client.get(f"task:{task_id}")
            if cached:
                return json.loads(cached)
        
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = api_service.tasks[task_id]
    return TaskResponse(
        task_id=task_id,
        status=task["status"],
        result=task["result"],
        error=task["error"],
        metadata={"created_at": task["created_at"].isoformat()}
    )


@app.post("/api/v1/training/start")
async def start_training(request: TrainingRequest):
    """Start model training"""
    return await api_service.train_model(request)


@app.get("/api/v1/training/{training_id}/progress")
async def get_training_progress(training_id: str):
    """Get training progress"""
    if api_service.redis_client:
        progress = api_service.redis_client.get(f"training:{training_id}:progress")
        if progress:
            return json.loads(progress)
    
    return {"error": "Training progress not found"}


@app.post("/api/v1/prompts/optimize")
async def optimize_prompt(request: PromptOptimizationRequest):
    """Optimize a prompt"""
    return await api_service.optimize_prompt(request)


@app.post("/api/v1/models/deploy")
async def deploy_model(request: ModelDeployRequest):
    """Deploy a model"""
    # Simplified deployment simulation
    deployment_id = str(uuid.uuid4())
    
    return {
        "deployment_id": deployment_id,
        "model_id": request.model_id,
        "deployment_name": request.deployment_name,
        "status": "deployed",
        "endpoint": f"/api/v1/models/{deployment_id}/predict",
        "replicas": request.replicas
    }


@app.get("/api/v1/agents")
async def list_agents():
    """List all agents"""
    return {
        "agents": [
            {
                "id": agent_id,
                "role": agent.role,
                "status": "active"
            }
            for agent_id, agent in api_service.agents.items()
        ]
    }


@app.post("/api/v1/agents/{agent_id}")
async def create_agent(agent_id: str, role: str = "Assistant"):
    """Create a new agent"""
    agent = await api_service.create_agent(agent_id, role)
    return {
        "agent_id": agent_id,
        "role": role,
        "status": "created"
    }


@app.delete("/api/v1/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent"""
    if agent_id in api_service.agents:
        del api_service.agents[agent_id]
        return {"status": "deleted", "agent_id": agent_id}
    
    raise HTTPException(status_code=404, detail="Agent not found")


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    api_service.active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Parse command
            command = json.loads(data)
            
            if command["type"] == "subscribe":
                # Subscribe to task updates
                task_id = command.get("task_id")
                if task_id in api_service.tasks:
                    task = api_service.tasks[task_id]
                    await websocket.send_json({
                        "task_id": task_id,
                        "status": task["status"].value,
                        "result": task["result"]
                    })
            
            elif command["type"] == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        api_service.active_connections.remove(websocket)


# Streaming endpoint for continuous generation
@app.post("/api/v1/agents/stream")
async def stream_agent_response(request: AgentRequest):
    """Stream agent responses"""
    
    async def generate() -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        agent_id = request.agent_id or "streaming_agent"
        agent = await api_service.create_agent(agent_id)
        
        # Simulate streaming generation
        response_parts = [
            "Processing your request...\n",
            "Analyzing the task...\n",
            "Generating response...\n",
            "Here is the result:\n",
            "Task completed successfully!\n"
        ]
        
        for part in response_parts:
            await asyncio.sleep(0.5)  # Simulate processing time
            yield f"data: {json.dumps({'content': part})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# Batch processing endpoint
@app.post("/api/v1/batch/process")
async def process_batch(requests: List[AgentRequest]):
    """Process multiple requests in batch"""
    tasks = []
    
    for request in requests:
        task = await api_service.process_task(request)
        tasks.append(task)
    
    return {
        "batch_id": str(uuid.uuid4()),
        "tasks": tasks,
        "total": len(tasks)
    }


# Metrics endpoint
@app.get("/api/v1/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "total_agents": len(api_service.agents),
            "active_tasks": sum(1 for t in api_service.tasks.values() 
                              if t["status"] == TaskStatus.PROCESSING),
            "completed_tasks": sum(1 for t in api_service.tasks.values() 
                                 if t["status"] == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in api_service.tasks.values() 
                              if t["status"] == TaskStatus.FAILED),
            "queue_size": api_service.task_queue.qsize(),
            "active_connections": len(api_service.active_connections)
        }
    }


# Authentication endpoints
@app.post("/api/v1/auth/token", response_model=AuthToken)
async def login(username: str, password: str):
    """Authenticate and get token"""
    # Simplified authentication
    if username == "admin" and password == "admin":  # In production, use proper auth
        token = api_service.generate_token(username)
        return AuthToken(access_token=token)
    
    raise HTTPException(status_code=401, detail="Invalid credentials")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("🚀 Agent Lightning API starting up...")
    
    # Initialize background task processor
    asyncio.create_task(process_task_queue())


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("🛑 Agent Lightning API shutting down...")
    
    # Close connections
    for connection in api_service.active_connections:
        await connection.close()


async def process_task_queue():
    """Background task queue processor"""
    while True:
        try:
            # Process tasks from queue
            if not api_service.task_queue.empty():
                task_id = await api_service.task_queue.get()
                # Task is already being processed in _execute_task
                logger.info(f"Processing task from queue: {task_id}")
            
            await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Queue processor error: {e}")
            await asyncio.sleep(1)


# Example usage
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Agent Lightning Production API")
    print("=" * 60)
    print("\nEndpoints:")
    print("  REST API: http://localhost:8000")
    print("  WebSocket: ws://localhost:8000/ws")
    print("  Documentation: http://localhost:8000/api/docs")
    print("  Health Check: http://localhost:8000/health")
    print("\nTo run the API server:")
    print("  uvicorn production_api:app --reload --port 8000")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)