#!/usr/bin/env python3
"""
Workflow Engine Microservice - Integrated with Shared Database
Handles workflow execution, task orchestration, and process automation
Using shared PostgreSQL and Redis for state management
Based on SA-004: Workflow Engine Integration
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

# Import shared data access layer
from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

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
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"


# Constants
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# Pydantic models for API
class WorkflowCreate(BaseModel):
    """Workflow creation model"""
    name: str = Field(description="Workflow name")
    description: str = Field(description="Workflow description")
    steps: List[Dict[str, Any]] = Field(description="Workflow steps")
    created_by: str = Field(description="Creator ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Workflow context")


class WorkflowUpdate(BaseModel):
    """Workflow update model"""
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class TaskExecute(BaseModel):
    """Task execution request"""
    agent_id: str = Field(description="Agent to execute task")
    task_data: Dict[str, Any] = Field(description="Task parameters")


class WorkflowQueue:
    """Redis-based workflow execution queue"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.queue_key = "workflow:queue"
        self.processing_key = "workflow:processing"
    
    def enqueue(self, workflow_id: str, priority: int = 5):
        """Add workflow to execution queue"""
        score = time.time() - (priority * 1000)  # Higher priority = lower score
        self.redis.zadd(self.queue_key, {workflow_id: score})
        logger.info(f"Enqueued workflow {workflow_id} with priority {priority}")
    
    def dequeue(self) -> Optional[str]:
        """Get next workflow to execute"""
        # Atomic move from queue to processing
        result = self.redis.zpopmin(self.queue_key)
        if result:
            workflow_id = result[0][0]
            self.redis.hset(self.processing_key, workflow_id, time.time())
            logger.info(f"Dequeued workflow {workflow_id} for processing")
            return workflow_id
        return None
    
    def complete(self, workflow_id: str):
        """Mark workflow as completed"""
        self.redis.hdel(self.processing_key, workflow_id)
        logger.info(f"Workflow {workflow_id} removed from processing")
    
    def get_queue_size(self) -> int:
        """Get number of workflows in queue"""
        return self.redis.zcard(self.queue_key)
    
    def get_processing(self) -> List[str]:
        """Get workflows currently being processed"""
        return list(self.redis.hkeys(self.processing_key))


class WorkflowRecovery:
    """Handles workflow recovery after failures"""
    
    def __init__(self, dal: DataAccessLayer, queue: WorkflowQueue):
        self.dal = dal
        self.queue = queue
    
    async def recover_interrupted_workflows(self):
        """Recover workflows interrupted by service restart"""
        logger.info("Starting workflow recovery...")
        
        # Find all workflows that were running
        with self.dal.db.get_db() as session:
            from shared.models import Workflow
            interrupted = session.query(Workflow).filter(
                Workflow.status == WorkflowStatus.RUNNING.value
            ).all()
            
            recovered = 0
            for workflow in interrupted:
                workflow_id = str(workflow.id)
                logger.info(f"Recovering workflow {workflow_id}")
                
                # Check last checkpoint
                last_checkpoint = workflow.context.get('checkpoint') if workflow.context else None
                
                if last_checkpoint:
                    # Resume from checkpoint
                    await self.resume_from_checkpoint(workflow_id, last_checkpoint)
                else:
                    # Re-enqueue workflow
                    self.queue.enqueue(workflow_id, priority=10)  # High priority for recovery
                
                recovered += 1
            
            logger.info(f"Recovered {recovered} interrupted workflows")
    
    async def resume_from_checkpoint(self, workflow_id: str, checkpoint: Dict):
        """Resume workflow from checkpoint"""
        logger.info(f"Resuming workflow {workflow_id} from checkpoint: {checkpoint}")
        # Implementation would restore workflow state and continue execution
        self.queue.enqueue(workflow_id, priority=10)


class WorkflowEngineService:
    """Main Workflow Engine Service class - Integrated with shared database"""
    
    def __init__(self):
        self.app = FastAPI(title="Workflow Engine Service (Integrated)", version="2.0.0")
        
        # Initialize Data Access Layer
        self.dal = DataAccessLayer("workflow_engine")
        self.cache = get_cache()
        
        # Initialize workflow queue
        self.workflow_queue = WorkflowQueue(self.cache.redis_client)
        
        # Initialize recovery handler
        self.recovery = WorkflowRecovery(self.dal, self.workflow_queue)
        
        # Active workflow tracking
        self.active_workflows: Dict[str, Any] = {}
        
        logger.info("✅ Connected to shared database and cache")
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
        
        # Start background worker
        self.worker_task = None
        
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            health_status = self.dal.health_check()
            return {
                "service": "workflow_engine",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "queue_size": self.workflow_queue.get_queue_size(),
                "processing": len(self.workflow_queue.get_processing()),
                "active_workflows": len(self.active_workflows),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/workflows")
        async def create_workflow(workflow: WorkflowCreate):
            """Create and optionally start a workflow"""
            try:
                # Create workflow in database
                workflow_data = workflow.dict()
                workflow_data['status'] = WorkflowStatus.PENDING.value
                created_workflow = self.dal.create_workflow(workflow_data)
                
                logger.info(f"Created workflow {created_workflow['id']} in shared database")
                
                # Add to execution queue
                self.workflow_queue.enqueue(created_workflow['id'], priority=5)
                
                return created_workflow
            except Exception as e:
                logger.error(f"Failed to create workflow: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/workflows")
        async def list_workflows(status: Optional[str] = None):
            """List all workflows"""
            try:
                # Query workflows from database
                with self.dal.db.get_db() as session:
                    from shared.models import Workflow
                    query = session.query(Workflow)
                    if status:
                        query = query.filter(Workflow.status == status)
                    workflows = query.order_by(Workflow.created_at.desc()).limit(100).all()
                    
                    return {
                        "workflows": [w.to_dict() for w in workflows],
                        "count": len(workflows),
                        "source": "shared_database"
                    }
            except Exception as e:
                logger.error(f"Failed to list workflows: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/workflows/{workflow_id}")
        async def get_workflow(workflow_id: str):
            """Get specific workflow details"""
            try:
                # Get from cache first
                cache_key = f"workflow:{workflow_id}"
                workflow = self.cache.get(cache_key)
                
                if not workflow:
                    # Load from database
                    with self.dal.db.get_db() as session:
                        from shared.models import Workflow
                        db_workflow = session.query(Workflow).filter(
                            Workflow.id == workflow_id
                        ).first()
                        
                        if not db_workflow:
                            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
                        
                        workflow = db_workflow.to_dict()
                        # Cache for 5 minutes
                        self.cache.set(cache_key, workflow, ttl=300)
                
                # Add execution details if active
                if workflow_id in self.active_workflows:
                    workflow['execution_details'] = self.active_workflows[workflow_id]
                
                return workflow
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get workflow: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/workflows/{workflow_id}/status")
        async def update_workflow_status(workflow_id: str, status: str):
            """Update workflow status"""
            try:
                if status not in [s.value for s in WorkflowStatus]:
                    raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
                
                workflow = self.dal.update_workflow_status(workflow_id, status)
                if not workflow:
                    raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
                
                logger.info(f"Updated workflow {workflow_id} status to {status}")
                
                # Handle special statuses
                if status == WorkflowStatus.CANCELLED.value:
                    self._cancel_workflow(workflow_id)
                elif status == WorkflowStatus.PAUSED.value:
                    self._pause_workflow(workflow_id)
                elif status == WorkflowStatus.RUNNING.value:
                    # Resume execution
                    self.workflow_queue.enqueue(workflow_id, priority=8)
                
                return workflow
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update workflow status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/workflows/{workflow_id}/tasks")
        async def get_workflow_tasks(workflow_id: str):
            """Get all tasks for a workflow"""
            try:
                # Get tasks associated with workflow
                tasks = self.dal.list_tasks(agent_id=None)  # Get all tasks
                workflow_tasks = [t for t in tasks if t.get('context', {}).get('workflow_id') == workflow_id]
                
                return {
                    "workflow_id": workflow_id,
                    "tasks": workflow_tasks,
                    "count": len(workflow_tasks)
                }
            except Exception as e:
                logger.error(f"Failed to get workflow tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/workflows/{workflow_id}/tasks")
        async def create_workflow_task(workflow_id: str, task_execute: TaskExecute):
            """Create a task within a workflow"""
            try:
                # Create task with workflow context
                task_data = {
                    "agent_id": task_execute.agent_id,
                    "description": f"Workflow task for {workflow_id}",
                    "context": {
                        "workflow_id": workflow_id,
                        **task_execute.task_data
                    },
                    "priority": "normal"
                }
                
                created_task = self.dal.create_task(task_data)
                logger.info(f"Created task {created_task['id']} for workflow {workflow_id}")
                
                return created_task
            except Exception as e:
                logger.error(f"Failed to create workflow task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/execution/queue")
        async def get_execution_queue():
            """Get current execution queue status"""
            return {
                "queue_size": self.workflow_queue.get_queue_size(),
                "processing": self.workflow_queue.get_processing(),
                "active_workflows": list(self.active_workflows.keys())
            }
    
    async def _workflow_worker(self):
        """Background worker to process workflow queue"""
        logger.info("Workflow worker started")
        
        while True:
            try:
                # Get next workflow from queue
                workflow_id = self.workflow_queue.dequeue()
                
                if workflow_id:
                    # Execute workflow
                    await self._execute_workflow(workflow_id)
                    
                    # Mark as complete in queue
                    self.workflow_queue.complete(workflow_id)
                else:
                    # No workflows to process, wait
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in workflow worker: {e}")
                await asyncio.sleep(5)
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute a workflow asynchronously"""
        try:
            logger.info(f"Starting execution of workflow {workflow_id}")
            
            # Mark as active
            self.active_workflows[workflow_id] = {
                "started_at": datetime.utcnow().isoformat(),
                "current_step": 0,
                "status": "executing"
            }
            
            # Update status to running
            self.dal.update_workflow_status(workflow_id, WorkflowStatus.RUNNING.value)
            
            # Get workflow details
            with self.dal.db.get_db() as session:
                from shared.models import Workflow
                workflow = session.query(Workflow).filter(
                    Workflow.id == workflow_id
                ).first()
                
                if not workflow:
                    logger.error(f"Workflow {workflow_id} not found")
                    return
                
                steps = workflow.steps or []
            
            # Execute each step
            for i, step in enumerate(steps):
                if workflow_id not in self.active_workflows:
                    logger.info(f"Workflow {workflow_id} cancelled or paused")
                    break
                
                self.active_workflows[workflow_id]["current_step"] = i + 1
                
                # Save checkpoint
                self._save_checkpoint(workflow_id, i)
                
                # Execute step based on type
                success = await self._execute_step(workflow_id, step)
                
                if not success:
                    # Mark workflow as failed
                    self.dal.update_workflow_status(workflow_id, WorkflowStatus.FAILED.value)
                    
                    # Record failure metric
                    self.dal.record_metric('workflow_step_failure', 1, {
                        'workflow_id': workflow_id,
                        'step': i + 1
                    })
                    break
            else:
                # All steps completed successfully
                self.dal.update_workflow_status(workflow_id, WorkflowStatus.COMPLETED.value)
                logger.info(f"Workflow {workflow_id} completed successfully")
                
                # Record success metric
                self.dal.record_metric('workflow_completion', 1, {
                    'workflow_id': workflow_id,
                    'steps_completed': len(steps)
                })
            
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
                
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            self.dal.update_workflow_status(workflow_id, WorkflowStatus.FAILED.value)
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
    
    async def _execute_step(self, workflow_id: str, step: Dict[str, Any]) -> bool:
        """Execute a single workflow step"""
        try:
            step_type = step.get("type", "task")
            
            if step_type == "task":
                return await self._execute_task_step(workflow_id, step)
            elif step_type == "conditional":
                return await self._execute_conditional_step(workflow_id, step)
            elif step_type == "parallel":
                return await self._execute_parallel_step(workflow_id, step)
            else:
                logger.warning(f"Unknown step type: {step_type}")
                return True
                
        except Exception as e:
            logger.error(f"Error executing step: {e}")
            return False
    
    async def _execute_task_step(self, workflow_id: str, step: Dict) -> bool:
        """Execute a task step"""
        # Find available agent
        agent_id = step.get("agent_id")
        if not agent_id:
            agents = self.dal.list_agents()
            idle_agents = [a for a in agents if a['status'] == 'idle']
            if idle_agents:
                agent_id = idle_agents[0]['id']
            else:
                logger.warning(f"No available agents for workflow {workflow_id}")
                return False
        
        # Create task
        task_data = {
            "agent_id": agent_id,
            "description": step.get("description", "Workflow step"),
            "context": {
                "workflow_id": workflow_id,
                "step": step,
                "retry_count": 0
            }
        }
        
        task = self.dal.create_task(task_data)
        logger.info(f"Created task {task['id']} for workflow step")
        
        # Wait for task completion (simplified - in production use events)
        max_wait = 60  # seconds
        wait_time = 0
        while wait_time < max_wait:
            await asyncio.sleep(2)
            wait_time += 2
            
            # Check task status
            updated_task = self.dal.get_task(task['id'])
            if updated_task and updated_task['status'] == 'completed':
                return True
            elif updated_task and updated_task['status'] == 'failed':
                # Retry logic
                return await self._retry_task(task['id'], workflow_id, 0)
        
        logger.warning(f"Task {task['id']} timed out")
        return False
    
    async def _retry_task(self, task_id: str, workflow_id: str, retry_count: int) -> bool:
        """Retry a failed task"""
        if retry_count >= MAX_RETRIES:
            logger.error(f"Task {task_id} failed after {MAX_RETRIES} retries")
            return False
        
        logger.info(f"Retrying task {task_id} (attempt {retry_count + 1})")
        await asyncio.sleep(RETRY_DELAY * (retry_count + 1))  # Exponential backoff
        
        # Update task status to retry
        self.dal.update_task_status(task_id, TaskStatus.RETRY.value)
        
        # Re-execute (simplified)
        await asyncio.sleep(2)
        
        # Check result
        task = self.dal.get_task(task_id)
        if task and task['status'] == 'completed':
            return True
        else:
            return await self._retry_task(task_id, workflow_id, retry_count + 1)
    
    async def _execute_conditional_step(self, workflow_id: str, step: Dict) -> bool:
        """Execute a conditional step"""
        condition = step.get("condition", {})
        # Simplified condition evaluation
        logger.info(f"Evaluating condition for workflow {workflow_id}")
        return True
    
    async def _execute_parallel_step(self, workflow_id: str, step: Dict) -> bool:
        """Execute parallel tasks"""
        tasks = step.get("tasks", [])
        logger.info(f"Executing {len(tasks)} parallel tasks for workflow {workflow_id}")
        
        # Create all tasks
        task_results = []
        for task in tasks:
            result = await self._execute_step(workflow_id, task)
            task_results.append(result)
        
        # All must succeed
        return all(task_results)
    
    def _save_checkpoint(self, workflow_id: str, step_index: int):
        """Save workflow checkpoint"""
        checkpoint = {
            "step_index": step_index,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update workflow context with checkpoint
        with self.dal.db.get_db() as session:
            from shared.models import Workflow
            workflow = session.query(Workflow).filter(
                Workflow.id == workflow_id
            ).first()
            
            if workflow:
                if not workflow.context:
                    workflow.context = {}
                workflow.context['checkpoint'] = checkpoint
                session.commit()
                
                logger.debug(f"Saved checkpoint for workflow {workflow_id} at step {step_index}")
    
    def _cancel_workflow(self, workflow_id: str):
        """Cancel a running workflow"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = "cancelled"
            del self.active_workflows[workflow_id]
            logger.info(f"Workflow {workflow_id} cancelled")
    
    def _pause_workflow(self, workflow_id: str):
        """Pause a running workflow"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = "paused"
            logger.info(f"Workflow {workflow_id} paused")
    
    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""
        
        def on_task_completed(event):
            """Handle task completion events"""
            try:
                task_id = event.data.get('task_id')
                result = event.data.get('result')
                logger.info(f"Task {task_id} completed with result: {result}")
                
                # Check if task belongs to a workflow
                task = self.dal.get_task(task_id)
                if task and task.get('context', {}).get('workflow_id'):
                    workflow_id = task['context']['workflow_id']
                    # Update workflow progress
                    if workflow_id in self.active_workflows:
                        self.active_workflows[workflow_id]["last_task_completed"] = task_id
                        
            except Exception as e:
                logger.error(f"Error handling task completion: {e}")
        
        def on_task_failed(event):
            """Handle task failure events"""
            try:
                task_id = event.data.get('task_id')
                error = event.data.get('error')
                logger.warning(f"Task {task_id} failed: {error}")
                
                # Check if task belongs to a workflow
                task = self.dal.get_task(task_id)
                if task and task.get('context', {}).get('workflow_id'):
                    workflow_id = task['context']['workflow_id']
                    retry_count = task.get('context', {}).get('retry_count', 0)
                    
                    # Trigger retry if applicable
                    if retry_count < MAX_RETRIES:
                        logger.info(f"Scheduling retry for task {task_id}")
                        # In production, would re-enqueue task with increased retry count
                        
            except Exception as e:
                logger.error(f"Error handling task failure: {e}")
        
        def on_agent_status(event):
            """Handle agent status changes"""
            agent_id = event.data.get('agent_id')
            status = event.data.get('status')
            logger.info(f"Agent {agent_id} status changed to {status}")
        
        # Register event handlers
        self.dal.event_bus.on(EventChannel.TASK_COMPLETED, on_task_completed)
        self.dal.event_bus.on(EventChannel.TASK_FAILED, on_task_failed)
        self.dal.event_bus.on(EventChannel.AGENT_STATUS, on_agent_status)
        
        logger.info("Event handlers registered for cross-service communication")
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Workflow Engine Service (Integrated) starting up...")
        
        # Verify database connection
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")
        
        # Recover interrupted workflows
        await self.recovery.recover_interrupted_workflows()
        
        # Start background worker
        self.worker_task = asyncio.create_task(self._workflow_worker())
        
        logger.info(f"Workflow Engine ready with {self.workflow_queue.get_queue_size()} workflows in queue")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Workflow Engine Service shutting down...")
        
        # Stop background worker
        if self.worker_task:
            self.worker_task.cancel()
        
        # Pause all active workflows
        for workflow_id in list(self.active_workflows.keys()):
            self.dal.update_workflow_status(workflow_id, WorkflowStatus.PAUSED.value)
        
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = WorkflowEngineService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("WORKFLOW_ENGINE_PORT", 8003))
    logger.info(f"Starting Workflow Engine Service (Integrated) on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()