#!/usr/bin/env python3
"""
Workflow Engine Microservice
Handles workflow execution, task orchestration, and process automation
Based on the architecture from technical_architecture.md
"""

import os
import sys
import json
import asyncio
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging
from collections import deque

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskStatus(str, Enum):
    """Individual task status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"


class TaskType(str, Enum):
    """Types of workflow tasks"""
    AI_INFERENCE = "ai_inference"
    API_CALL = "api_call"
    DATABASE_QUERY = "database_query"
    DATA_TRANSFORM = "data_transform"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"
    WAIT = "wait"
    HUMAN_APPROVAL = "human_approval"
    CUSTOM = "custom"


class TriggerType(str, Enum):
    """Workflow trigger types"""
    MANUAL = "manual"
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    EVENT = "event"
    API = "api"


# Pydantic Models
class WorkflowTask(BaseModel):
    """Individual task in a workflow"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType
    name: str
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    next: List[str] = Field(default_factory=list)  # Next task IDs
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_attempts": 3, "delay": 5})
    timeout: int = 300  # seconds
    depends_on: List[str] = Field(default_factory=list)  # Task dependencies


class WorkflowTrigger(BaseModel):
    """Workflow trigger configuration"""
    type: TriggerType
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class WorkflowDefinition(BaseModel):
    """Complete workflow definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    trigger: WorkflowTrigger
    tasks: List[WorkflowTask]
    variables: Dict[str, Any] = Field(default_factory=dict)
    error_handling: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TaskResult(BaseModel):
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    retry_count: int = 0


class WorkflowExecution(BaseModel):
    """Workflow execution instance"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    agent_id: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    task_results: List[TaskResult] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class ExecuteWorkflowRequest(BaseModel):
    """Request to execute a workflow"""
    workflow_id: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    async_execution: bool = True
    callback_url: Optional[str] = None


class CreateWorkflowRequest(BaseModel):
    """Request to create a new workflow"""
    name: str
    description: Optional[str] = None
    trigger: WorkflowTrigger
    tasks: List[WorkflowTask]
    variables: Dict[str, Any] = Field(default_factory=dict)


class WorkflowEngineService:
    """Main Workflow Engine Service"""
    
    def __init__(self):
        self.app = FastAPI(title="Workflow Engine Service", version="1.0.0")
        
        # In-memory storage (would be database in production)
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.execution_queue: deque = deque()
        self.task_executors: Dict[TaskType, Any] = {}
        
        # Metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration_ms": 0
        }
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
        self._setup_task_executors()
        self._load_sample_workflows()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_event_handlers(self):
        """Setup startup and shutdown handlers"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Start background tasks"""
            asyncio.create_task(self._execution_worker())
            logger.info("Workflow Engine Service started")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            logger.info("Workflow Engine Service shut down")
    
    def _setup_task_executors(self):
        """Setup task type executors"""
        self.task_executors = {
            TaskType.AI_INFERENCE: self._execute_ai_inference,
            TaskType.API_CALL: self._execute_api_call,
            TaskType.DATABASE_QUERY: self._execute_database_query,
            TaskType.DATA_TRANSFORM: self._execute_data_transform,
            TaskType.CONDITIONAL: self._execute_conditional,
            TaskType.LOOP: self._execute_loop,
            TaskType.PARALLEL: self._execute_parallel,
            TaskType.WAIT: self._execute_wait,
            TaskType.HUMAN_APPROVAL: self._execute_human_approval,
            TaskType.CUSTOM: self._execute_custom
        }
    
    def _load_sample_workflows(self):
        """Load sample workflow definitions"""
        # Customer Support Workflow
        customer_support = WorkflowDefinition(
            id="workflow-customer-support",
            name="Customer Support Agent",
            description="Automated customer support workflow",
            trigger=WorkflowTrigger(
                type=TriggerType.WEBHOOK,
                config={"path": "/webhook/support", "method": "POST"}
            ),
            tasks=[
                WorkflowTask(
                    id="classify-intent",
                    type=TaskType.AI_INFERENCE,
                    name="Classify Intent",
                    config={
                        "model": "gpt-4",
                        "prompt": "Classify the customer intent: {{input.message}}",
                        "max_tokens": 100
                    },
                    next=["route-to-handler"]
                ),
                WorkflowTask(
                    id="route-to-handler",
                    type=TaskType.CONDITIONAL,
                    name="Route to Handler",
                    config={
                        "conditions": [
                            {"if": "{{classify-intent.output.intent == 'billing'}}", "then": "billing-handler"},
                            {"if": "{{classify-intent.output.intent == 'technical'}}", "then": "technical-handler"}
                        ],
                        "default": "general-handler"
                    }
                ),
                WorkflowTask(
                    id="billing-handler",
                    type=TaskType.AI_INFERENCE,
                    name="Billing Handler",
                    config={
                        "model": "gpt-3.5-turbo",
                        "prompt": "Handle billing inquiry: {{input.message}}"
                    }
                ),
                WorkflowTask(
                    id="technical-handler",
                    type=TaskType.AI_INFERENCE,
                    name="Technical Handler",
                    config={
                        "model": "gpt-4",
                        "prompt": "Provide technical support for: {{input.message}}"
                    }
                ),
                WorkflowTask(
                    id="general-handler",
                    type=TaskType.AI_INFERENCE,
                    name="General Handler",
                    config={
                        "model": "gpt-3.5-turbo",
                        "prompt": "Respond to general inquiry: {{input.message}}"
                    }
                )
            ]
        )
        self.workflows[customer_support.id] = customer_support
        
        # Data Processing Pipeline
        data_pipeline = WorkflowDefinition(
            id="workflow-data-pipeline",
            name="Data Processing Pipeline",
            description="ETL data processing workflow",
            trigger=WorkflowTrigger(
                type=TriggerType.SCHEDULE,
                config={"cron": "0 */6 * * *"}  # Every 6 hours
            ),
            tasks=[
                WorkflowTask(
                    id="extract-data",
                    type=TaskType.DATABASE_QUERY,
                    name="Extract Data",
                    config={
                        "query": "SELECT * FROM raw_data WHERE processed = false",
                        "database": "analytics"
                    },
                    next=["transform-data"]
                ),
                WorkflowTask(
                    id="transform-data",
                    type=TaskType.DATA_TRANSFORM,
                    name="Transform Data",
                    config={
                        "operations": [
                            {"type": "filter", "condition": "value > 0"},
                            {"type": "map", "function": "normalize"},
                            {"type": "aggregate", "by": "category"}
                        ]
                    },
                    next=["load-data"]
                ),
                WorkflowTask(
                    id="load-data",
                    type=TaskType.DATABASE_QUERY,
                    name="Load Data",
                    config={
                        "operation": "insert",
                        "table": "processed_data",
                        "database": "warehouse"
                    }
                )
            ]
        )
        self.workflows[data_pipeline.id] = data_pipeline
        
        logger.info(f"Loaded {len(self.workflows)} sample workflows")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Service health check"""
            return {
                "status": "healthy",
                "service": "workflow_engine",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics
            }
        
        # Workflow management
        @self.app.post("/api/v1/workflows", response_model=WorkflowDefinition)
        async def create_workflow(request: CreateWorkflowRequest):
            """Create a new workflow definition"""
            workflow = WorkflowDefinition(
                name=request.name,
                description=request.description,
                trigger=request.trigger,
                tasks=request.tasks,
                variables=request.variables
            )
            
            self.workflows[workflow.id] = workflow
            logger.info(f"Created workflow: {workflow.id}")
            return workflow
        
        @self.app.get("/api/v1/workflows", response_model=List[WorkflowDefinition])
        async def list_workflows():
            """List all workflow definitions"""
            return list(self.workflows.values())
        
        @self.app.get("/api/v1/workflows/{workflow_id}", response_model=WorkflowDefinition)
        async def get_workflow(workflow_id: str):
            """Get a specific workflow definition"""
            if workflow_id not in self.workflows:
                raise HTTPException(status_code=404, detail="Workflow not found")
            return self.workflows[workflow_id]
        
        @self.app.delete("/api/v1/workflows/{workflow_id}")
        async def delete_workflow(workflow_id: str):
            """Delete a workflow definition"""
            if workflow_id not in self.workflows:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            del self.workflows[workflow_id]
            logger.info(f"Deleted workflow: {workflow_id}")
            return {"message": "Workflow deleted successfully"}
        
        # Workflow execution
        @self.app.post("/api/v1/workflows/execute", response_model=WorkflowExecution)
        async def execute_workflow(request: ExecuteWorkflowRequest, background_tasks: BackgroundTasks):
            """Execute a workflow"""
            if request.workflow_id not in self.workflows:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            workflow = self.workflows[request.workflow_id]
            
            # Create execution instance
            execution = WorkflowExecution(
                workflow_id=request.workflow_id,
                input_data=request.input_data,
                context={
                    "workflow_name": workflow.name,
                    "variables": workflow.variables.copy()
                }
            )
            
            self.executions[execution.id] = execution
            
            if request.async_execution:
                # Queue for async execution
                self.execution_queue.append(execution.id)
                logger.info(f"Queued workflow execution: {execution.id}")
            else:
                # Execute synchronously
                await self._execute_workflow(execution.id)
            
            return execution
        
        @self.app.get("/api/v1/executions", response_model=List[WorkflowExecution])
        async def list_executions(status: Optional[WorkflowStatus] = None):
            """List workflow executions"""
            executions = list(self.executions.values())
            
            if status:
                executions = [e for e in executions if e.status == status]
            
            return executions
        
        @self.app.get("/api/v1/executions/{execution_id}", response_model=WorkflowExecution)
        async def get_execution(execution_id: str):
            """Get workflow execution details"""
            if execution_id not in self.executions:
                raise HTTPException(status_code=404, detail="Execution not found")
            
            return self.executions[execution_id]
        
        @self.app.post("/api/v1/executions/{execution_id}/cancel")
        async def cancel_execution(execution_id: str):
            """Cancel a running workflow execution"""
            if execution_id not in self.executions:
                raise HTTPException(status_code=404, detail="Execution not found")
            
            execution = self.executions[execution_id]
            
            if execution.status not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                raise HTTPException(status_code=400, detail="Execution cannot be cancelled")
            
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.now()
            
            logger.info(f"Cancelled execution: {execution_id}")
            return {"message": "Execution cancelled successfully"}
        
        @self.app.get("/api/v1/metrics")
        async def get_metrics():
            """Get workflow engine metrics"""
            return {
                "metrics": self.metrics,
                "queue_size": len(self.execution_queue),
                "active_workflows": len(self.workflows),
                "total_executions": len(self.executions),
                "executions_by_status": self._get_executions_by_status()
            }
    
    def _get_executions_by_status(self) -> Dict[str, int]:
        """Get count of executions by status"""
        status_count = {}
        for execution in self.executions.values():
            status = execution.status.value
            status_count[status] = status_count.get(status, 0) + 1
        return status_count
    
    async def _execution_worker(self):
        """Background worker to process queued executions"""
        while True:
            try:
                if self.execution_queue:
                    execution_id = self.execution_queue.popleft()
                    await self._execute_workflow(execution_id)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Execution worker error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_workflow(self, execution_id: str):
        """Execute a complete workflow"""
        if execution_id not in self.executions:
            logger.error(f"Execution {execution_id} not found")
            return
        
        execution = self.executions[execution_id]
        workflow = self.workflows.get(execution.workflow_id)
        
        if not workflow:
            execution.status = WorkflowStatus.FAILED
            execution.error = "Workflow definition not found"
            return
        
        try:
            # Start execution
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now()
            self.metrics["total_executions"] += 1
            
            logger.info(f"Starting workflow execution: {execution_id}")
            
            # Execute tasks in sequence
            for task in workflow.tasks:
                if execution.status == WorkflowStatus.CANCELLED:
                    break
                
                result = await self._execute_task(task, execution)
                execution.task_results.append(result)
                
                # Update context with task output
                if result.status == TaskStatus.SUCCESS and result.output:
                    execution.context[task.id] = {"output": result.output}
                elif result.status == TaskStatus.FAILED:
                    # Handle task failure
                    if workflow.error_handling.get("stop_on_error", True):
                        execution.status = WorkflowStatus.FAILED
                        execution.error = f"Task {task.id} failed: {result.error}"
                        break
            
            # Complete execution
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                self.metrics["successful_executions"] += 1
            
            execution.completed_at = datetime.now()
            execution.duration_ms = int((execution.completed_at - execution.started_at).total_seconds() * 1000)
            
            # Update metrics
            self._update_metrics(execution)
            
            logger.info(f"Completed workflow execution: {execution_id} - Status: {execution.status}")
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            self.metrics["failed_executions"] += 1
    
    async def _execute_task(self, task: WorkflowTask, execution: WorkflowExecution) -> TaskResult:
        """Execute a single workflow task"""
        logger.info(f"Executing task: {task.id} ({task.type})")
        
        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now()
        )
        
        try:
            # Get executor for task type
            executor = self.task_executors.get(task.type)
            
            if not executor:
                raise ValueError(f"No executor for task type: {task.type}")
            
            # Execute task with timeout
            task_output = await asyncio.wait_for(
                executor(task, execution),
                timeout=task.timeout
            )
            
            result.status = TaskStatus.SUCCESS
            result.output = task_output
            
        except asyncio.TimeoutError:
            result.status = TaskStatus.FAILED
            result.error = f"Task timeout after {task.timeout} seconds"
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            
            # Retry logic
            if result.retry_count < task.retry_policy["max_attempts"]:
                result.retry_count += 1
                result.status = TaskStatus.RETRY
                await asyncio.sleep(task.retry_policy["delay"])
                return await self._execute_task(task, execution)
        
        finally:
            result.completed_at = datetime.now()
            result.duration_ms = int((result.completed_at - result.started_at).total_seconds() * 1000)
        
        return result
    
    # Task Executors
    async def _execute_ai_inference(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute AI inference task"""
        # Simulate AI model call
        await asyncio.sleep(0.5)  # Simulate processing time
        
        return {
            "model": task.config.get("model", "gpt-3.5-turbo"),
            "response": f"AI response for: {task.config.get('prompt', 'No prompt')}",
            "tokens_used": 100
        }
    
    async def _execute_api_call(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute API call task"""
        # Simulate API call
        await asyncio.sleep(0.3)
        
        return {
            "status_code": 200,
            "response": {"data": "API response data"}
        }
    
    async def _execute_database_query(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute database query task"""
        # Simulate database operation
        await asyncio.sleep(0.2)
        
        return {
            "rows_affected": 10,
            "data": [{"id": 1, "value": "sample"}]
        }
    
    async def _execute_data_transform(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute data transformation task"""
        # Simulate data transformation
        await asyncio.sleep(0.1)
        
        return {
            "transformed_records": 100,
            "operations_applied": task.config.get("operations", [])
        }
    
    async def _execute_conditional(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute conditional branching task"""
        # Evaluate conditions and determine next task
        conditions = task.config.get("conditions", [])
        
        for condition in conditions:
            # Simple condition evaluation (would be more complex in production)
            if "billing" in str(execution.context):
                return {"next_task": "billing-handler", "condition_met": condition.get("if")}
        
        return {"next_task": task.config.get("default", ""), "condition_met": None}
    
    async def _execute_loop(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute loop task"""
        iterations = task.config.get("iterations", 1)
        
        results = []
        for i in range(iterations):
            await asyncio.sleep(0.1)
            results.append(f"Iteration {i+1}")
        
        return {"iterations": iterations, "results": results}
    
    async def _execute_parallel(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute parallel tasks"""
        # Simulate parallel execution
        parallel_tasks = task.config.get("tasks", [])
        
        # In production, would execute tasks in parallel
        await asyncio.sleep(0.5)
        
        return {
            "parallel_tasks": len(parallel_tasks),
            "all_completed": True
        }
    
    async def _execute_wait(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute wait task"""
        wait_time = task.config.get("seconds", 1)
        await asyncio.sleep(min(wait_time, 10))  # Cap at 10 seconds for demo
        
        return {"waited_seconds": wait_time}
    
    async def _execute_human_approval(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute human approval task"""
        # In production, would create approval request and wait
        await asyncio.sleep(0.5)
        
        return {
            "approved": True,
            "approver": "auto-approved",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_custom(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute custom task"""
        # Custom task execution logic
        await asyncio.sleep(0.3)
        
        return {
            "custom_result": "Custom task executed",
            "config": task.config
        }
    
    def _update_metrics(self, execution: WorkflowExecution):
        """Update service metrics"""
        if execution.duration_ms:
            # Update average duration
            total = self.metrics["successful_executions"] + self.metrics["failed_executions"]
            if total > 0:
                current_avg = self.metrics["average_duration_ms"]
                self.metrics["average_duration_ms"] = int(
                    (current_avg * (total - 1) + execution.duration_ms) / total
                )


def create_service():
    """Create and return the service instance"""
    return WorkflowEngineService()


if __name__ == "__main__":
    import uvicorn
    
    print("Workflow Engine Microservice")
    print("=" * 60)
    
    service = create_service()
    
    print("\n⚙️ Starting Workflow Engine Service on port 8003")
    print("\nEndpoints:")
    print("  • GET  /health - Health check")
    print("  • POST /api/v1/workflows - Create workflow")
    print("  • GET  /api/v1/workflows - List workflows")
    print("  • POST /api/v1/workflows/execute - Execute workflow")
    print("  • GET  /api/v1/executions - List executions")
    print("  • GET  /api/v1/metrics - Service metrics")
    
    print("\n📋 Sample Workflows:")
    print("  • Customer Support Agent")
    print("  • Data Processing Pipeline")
    
    uvicorn.run(service.app, host="0.0.0.0", port=8003, reload=False)