"""
Enterprise Agent Workflow Engine
Production-grade multi-agent orchestration with monitoring, fault tolerance, and scalability
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from langchain_agent_wrapper import LangChainAgentManager
from shared.database import db_manager
from shared.models import Task
from redis_manager import RedisManager

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class WorkflowTask:
    task_id: str
    name: str
    description: str
    agent_requirements: List[str]
    dependencies: List[str]
    timeout_seconds: int = 300
    retry_count: int = 3
    priority: int = 5
    metadata: Dict[str, Any] = None
    
    # Runtime fields
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_attempts: int = 0


@dataclass
class Workflow:
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    created_by: str
    
    # Runtime fields
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    metrics: Dict[str, Any] = None


class AgentPool:
    """Manages agent availability and load balancing"""
    
    def __init__(self, agent_manager: LangChainAgentManager):
        self.agent_manager = agent_manager
        self.agent_load: Dict[str, int] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        
    def get_best_agent(self, requirements: List[str]) -> Optional[str]:
        """Select best available agent based on requirements and load"""
        candidates = []
        
        for agent_name, agent in self.agent_manager.agents.items():
            # Check if agent meets requirements
            if self._meets_requirements(agent, requirements):
                load = self.agent_load.get(agent_name, 0)
                performance = self.agent_performance.get(agent_name, {}).get('success_rate', 1.0)
                score = performance / (1 + load)  # Higher performance, lower load = higher score
                candidates.append((agent_name, score))
        
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
        return None
    
    def _meets_requirements(self, agent, requirements: List[str]) -> bool:
        """Check if agent meets task requirements"""
        if not requirements:
            return True
        
        capabilities = agent.agent_config.capabilities
        for req in requirements:
            if req == "coding" and not capabilities.can_write_code:
                return False
            elif req == "testing" and not capabilities.can_test:
                return False
            elif req == "documentation" and not capabilities.can_write_documentation:
                return False
        return True
    
    def acquire_agent(self, agent_name: str):
        """Mark agent as busy"""
        self.agent_load[agent_name] = self.agent_load.get(agent_name, 0) + 1
    
    def release_agent(self, agent_name: str):
        """Mark agent as available"""
        if agent_name in self.agent_load:
            self.agent_load[agent_name] = max(0, self.agent_load[agent_name] - 1)
    
    def update_performance(self, agent_name: str, success: bool, execution_time: float):
        """Update agent performance metrics"""
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'avg_execution_time': 0.0,
                'success_rate': 1.0
            }
        
        perf = self.agent_performance[agent_name]
        perf['total_tasks'] += 1
        if success:
            perf['successful_tasks'] += 1
        
        perf['success_rate'] = perf['successful_tasks'] / perf['total_tasks']
        perf['avg_execution_time'] = (perf['avg_execution_time'] + execution_time) / 2


class WorkflowExecutor:
    """Executes workflow tasks with dependency resolution and fault tolerance"""
    
    def __init__(self, agent_pool: AgentPool, redis_manager: RedisManager):
        self.agent_pool = agent_pool
        self.redis = redis_manager
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def execute_workflow(self, workflow: Workflow) -> Workflow:
        """Execute complete workflow with monitoring and fault tolerance"""
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        
        try:
            # Store workflow state
            await self._save_workflow_state(workflow)
            
            # Execute tasks in dependency order
            while self._has_pending_tasks(workflow):
                ready_tasks = self._get_ready_tasks(workflow)
                
                if not ready_tasks:
                    # Check for deadlock
                    if self._has_running_tasks(workflow):
                        await asyncio.sleep(1)  # Wait for running tasks
                        continue
                    else:
                        # Deadlock detected
                        workflow.status = WorkflowStatus.FAILED
                        break
                
                # Execute ready tasks concurrently
                tasks_futures = []
                for task in ready_tasks:
                    future = asyncio.create_task(self._execute_task(workflow, task))
                    tasks_futures.append(future)
                
                # Wait for at least one task to complete
                if tasks_futures:
                    await asyncio.wait(tasks_futures, return_when=asyncio.FIRST_COMPLETED)
                
                # Update progress
                workflow.progress = self._calculate_progress(workflow)
                await self._save_workflow_state(workflow)
            
            # Finalize workflow
            if all(task.status == TaskStatus.COMPLETED for task in workflow.tasks):
                workflow.status = WorkflowStatus.COMPLETED
            else:
                workflow.status = WorkflowStatus.FAILED
            
            workflow.completed_at = datetime.utcnow()
            workflow.progress = 1.0 if workflow.status == WorkflowStatus.COMPLETED else workflow.progress
            
            # Generate metrics
            workflow.metrics = self._generate_metrics(workflow)
            
            await self._save_workflow_state(workflow)
            
        except Exception as e:
            logger.error(f"Workflow {workflow.workflow_id} failed: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()
            await self._save_workflow_state(workflow)
        
        return workflow
    
    async def _execute_task(self, workflow: Workflow, task: WorkflowTask):
        """Execute single task with retry logic and monitoring"""
        task.status = TaskStatus.ASSIGNED
        
        # Select best agent
        agent_name = self.agent_pool.get_best_agent(task.agent_requirements)
        if not agent_name:
            task.status = TaskStatus.FAILED
            task.error_message = "No suitable agent available"
            return
        
        task.assigned_agent = agent_name
        self.agent_pool.acquire_agent(agent_name)
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            # Build context from dependencies
            context = await self._build_task_context(workflow, task)
            
            # Execute with timeout
            start_time = time.time()
            agent = self.agent_pool.agent_manager.get_agent(agent_name)
            
            full_prompt = f"{task.description}\n\nContext:\n{context}"
            result = await asyncio.wait_for(
                asyncio.create_task(self._run_agent_task(agent, full_prompt, task.task_id)),
                timeout=task.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # Update agent performance
            self.agent_pool.update_performance(agent_name, True, execution_time)
            
            logger.info(f"Task {task.task_id} completed by {agent_name} in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error_message = f"Task timed out after {task.timeout_seconds} seconds"
            self.agent_pool.update_performance(agent_name, False, task.timeout_seconds)
            
        except Exception as e:
            task.error_message = str(e)
            
            # Retry logic
            if task.retry_attempts < task.retry_count:
                task.retry_attempts += 1
                task.status = TaskStatus.RETRYING
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_attempts}/{task.retry_count})")
                await asyncio.sleep(2 ** task.retry_attempts)  # Exponential backoff
                await self._execute_task(workflow, task)
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.utcnow()
                self.agent_pool.update_performance(agent_name, False, time.time() - start_time)
                logger.error(f"Task {task.task_id} failed permanently: {e}")
        
        finally:
            self.agent_pool.release_agent(agent_name)
    
    async def _run_agent_task(self, agent, prompt: str, task_id: str) -> str:
        """Run agent task in thread pool to avoid blocking"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            lambda: agent.invoke(prompt, f"workflow_{task_id}")
        )
    
    async def _build_task_context(self, workflow: Workflow, task: WorkflowTask) -> str:
        """Build context from completed dependency tasks"""
        context_parts = []
        
        for dep_id in task.dependencies:
            dep_task = next((t for t in workflow.tasks if t.task_id == dep_id), None)
            if dep_task and dep_task.status == TaskStatus.COMPLETED:
                context_parts.append(f"Task '{dep_task.name}': {dep_task.result[:300]}...")
        
        return "\n\n".join(context_parts) if context_parts else "No previous context."
    
    def _has_pending_tasks(self, workflow: Workflow) -> bool:
        """Check if workflow has pending tasks"""
        return any(task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.RUNNING, TaskStatus.RETRYING] 
                  for task in workflow.tasks)
    
    def _has_running_tasks(self, workflow: Workflow) -> bool:
        """Check if workflow has running tasks"""
        return any(task.status in [TaskStatus.RUNNING, TaskStatus.RETRYING] for task in workflow.tasks)
    
    def _get_ready_tasks(self, workflow: Workflow) -> List[WorkflowTask]:
        """Get tasks ready for execution (dependencies completed)"""
        ready = []
        for task in workflow.tasks:
            if task.status == TaskStatus.PENDING:
                deps_completed = all(
                    any(t.task_id == dep_id and t.status == TaskStatus.COMPLETED for t in workflow.tasks)
                    for dep_id in task.dependencies
                )
                if not task.dependencies or deps_completed:
                    ready.append(task)
        return ready
    
    def _calculate_progress(self, workflow: Workflow) -> float:
        """Calculate workflow progress percentage"""
        total_tasks = len(workflow.tasks)
        completed_tasks = sum(1 for task in workflow.tasks if task.status == TaskStatus.COMPLETED)
        return completed_tasks / total_tasks if total_tasks > 0 else 0.0
    
    def _generate_metrics(self, workflow: Workflow) -> Dict[str, Any]:
        """Generate workflow execution metrics"""
        total_time = (workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at else 0
        
        return {
            "total_execution_time": total_time,
            "total_tasks": len(workflow.tasks),
            "completed_tasks": sum(1 for t in workflow.tasks if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in workflow.tasks if t.status == TaskStatus.FAILED),
            "average_task_time": total_time / len(workflow.tasks) if workflow.tasks else 0,
            "agent_utilization": {
                agent: sum(1 for t in workflow.tasks if t.assigned_agent == agent)
                for agent in set(t.assigned_agent for t in workflow.tasks if t.assigned_agent)
            }
        }
    
    async def _save_workflow_state(self, workflow: Workflow):
        """Save workflow state to Redis for monitoring"""
        try:
            workflow_data = asdict(workflow)
            # Convert datetime objects to strings
            for key, value in workflow_data.items():
                if isinstance(value, datetime):
                    workflow_data[key] = value.isoformat()
            
            # Convert task datetime objects
            for task_data in workflow_data.get('tasks', []):
                for key, value in task_data.items():
                    if isinstance(value, datetime):
                        task_data[key] = value.isoformat()
            
            await self.redis.set_async(
                f"workflow:{workflow.workflow_id}", 
                json.dumps(workflow_data, default=str),
                ttl=86400  # 24 hours
            )
        except Exception as e:
            logger.error(f"Failed to save workflow state: {e}")


class EnterpriseWorkflowEngine:
    """
    Enterprise-grade workflow engine with monitoring, scaling, and fault tolerance
    """
    
    def __init__(self):
        self.agent_manager = LangChainAgentManager()
        self.redis_manager = RedisManager()
        self.agent_pool = AgentPool(self.agent_manager)
        self.executor = WorkflowExecutor(self.agent_pool, self.redis_manager)
        self.active_workflows: Dict[str, Workflow] = {}
        
        logger.info(f"Enterprise Workflow Engine initialized with {len(self.agent_manager.agents)} agents")
    
    async def create_workflow(self, name: str, description: str, tasks: List[Dict[str, Any]], created_by: str) -> str:
        """Create new workflow from task definitions"""
        workflow_id = str(uuid.uuid4())
        
        workflow_tasks = []
        for task_def in tasks:
            task = WorkflowTask(
                task_id=task_def["task_id"],
                name=task_def["name"],
                description=task_def["description"],
                agent_requirements=task_def.get("agent_requirements", []),
                dependencies=task_def.get("dependencies", []),
                timeout_seconds=task_def.get("timeout_seconds", 300),
                retry_count=task_def.get("retry_count", 3),
                priority=task_def.get("priority", 5),
                metadata=task_def.get("metadata", {})
            )
            workflow_tasks.append(task)
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            tasks=workflow_tasks,
            created_by=created_by,
            created_at=datetime.utcnow()
        )
        
        self.active_workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id}: {name} with {len(workflow_tasks)} tasks")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> Workflow:
        """Execute workflow asynchronously"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        return await self.executor.execute_workflow(workflow)
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get real-time workflow status"""
        if workflow_id not in self.active_workflows:
            # Try to load from Redis
            try:
                workflow_data = await self.redis_manager.get_async(f"workflow:{workflow_id}")
                if workflow_data:
                    return json.loads(workflow_data)
            except Exception:
                pass
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        return asdict(workflow)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        return {
            "total_agents": len(self.agent_manager.agents),
            "active_workflows": len(self.active_workflows),
            "agent_load": self.agent_pool.agent_load,
            "agent_performance": self.agent_pool.agent_performance,
            "system_status": "healthy"
        }


# Enterprise workflow templates
ENTERPRISE_WORKFLOWS = {
    "software_development": {
        "name": "Full Stack Development Pipeline",
        "description": "Complete software development lifecycle",
        "tasks": [
            {
                "task_id": "requirements_analysis",
                "name": "Requirements Analysis",
                "description": "Analyze and document software requirements",
                "agent_requirements": ["documentation"],
                "dependencies": [],
                "timeout_seconds": 600
            },
            {
                "task_id": "system_design",
                "name": "System Architecture Design",
                "description": "Design system architecture and database schema",
                "agent_requirements": ["coding"],
                "dependencies": ["requirements_analysis"],
                "timeout_seconds": 900
            },
            {
                "task_id": "backend_development",
                "name": "Backend Development",
                "description": "Implement backend API and business logic",
                "agent_requirements": ["coding"],
                "dependencies": ["system_design"],
                "timeout_seconds": 1800
            },
            {
                "task_id": "frontend_development",
                "name": "Frontend Development",
                "description": "Implement user interface and frontend logic",
                "agent_requirements": ["coding"],
                "dependencies": ["backend_development"],
                "timeout_seconds": 1800
            },
            {
                "task_id": "testing",
                "name": "Comprehensive Testing",
                "description": "Write and execute unit, integration, and end-to-end tests",
                "agent_requirements": ["testing", "coding"],
                "dependencies": ["frontend_development"],
                "timeout_seconds": 1200
            },
            {
                "task_id": "deployment",
                "name": "Production Deployment",
                "description": "Deploy application to production environment",
                "agent_requirements": ["coding"],
                "dependencies": ["testing"],
                "timeout_seconds": 600
            }
        ]
    }
}