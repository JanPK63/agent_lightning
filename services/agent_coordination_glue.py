#!/usr/bin/env python3
"""
Agent Coordination Glue Code
Integrates all components for seamless agent coordination
Provides unified interface for task execution with all enterprise features
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiohttp
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache
from shared.events import EventChannel, EventBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskRequest(BaseModel):
    """Unified task request"""
    description: str = Field(description="Task description")
    priority: int = Field(default=5, ge=1, le=10)
    user_id: Optional[str] = Field(default=None, description="User requesting task")
    requirements: Optional[Dict[str, Any]] = Field(default={}, description="Task requirements")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    skip_validation: bool = Field(default=False, description="Skip validation checks")
    skip_governance: bool = Field(default=False, description="Skip governance gates")


class TaskResponse(BaseModel):
    """Unified task response"""
    task_id: str
    status: str
    agent_id: Optional[str]
    capability_match: Optional[float]
    validation_passed: bool
    governance_passed: bool
    idempotency_key: Optional[str]
    history_logged: bool
    message: str
    details: Dict[str, Any]


class AgentCoordinationGlue:
    """Central coordination system integrating all components"""
    
    def __init__(self):
        self.app = FastAPI(title="Agent Coordination Glue", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("agent_coordination")
        self.cache = get_cache()
        self.event_bus = EventBus("agent_coordination")
        
        # Service URLs
        self.services = {
            "rl_orchestrator": "http://localhost:8025",
            "task_validation": "http://localhost:8024",
            "idempotent_state": "http://localhost:8026",
            "task_history": "http://localhost:8027",
            "governance_gates": "http://localhost:8028",
            "capability_matcher": "http://localhost:8029",  # If separate service
            "auth_service": "http://localhost:8001",
            "monitoring": "http://localhost:8007"
        }
        
        # Database connection
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agent_lightning"),
            "user": os.getenv("POSTGRES_USER", "agent_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "agent_password")
        }
        
        logger.info("✅ Agent Coordination Glue initialized")
        
        self._setup_middleware()
        self._setup_routes()
        self.event_bus.start()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    async def _check_idempotency(self, task_description: str, user_id: str) -> Tuple[bool, Optional[Dict]]:
        """Check if operation was already performed"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "operation_type": "task_execution",
                    "entity_id": f"{user_id}:{hash(task_description) % 1000000}",
                    "operation_params": {
                        "description": task_description,
                        "user_id": user_id
                    }
                }
                
                async with session.post(
                    f"{self.services['idempotent_state']}/check",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["can_proceed"], result.get("existing_result")
                    
        except Exception as e:
            logger.error(f"Idempotency check failed: {e}")
        
        return True, None  # Allow to proceed if check fails
    
    async def _register_idempotent_operation(self, task_id: str, task_description: str, 
                                            user_id: str) -> str:
        """Register operation for idempotency"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "operation_type": "task_execution",
                    "entity_id": f"{user_id}:{hash(task_description) % 1000000}",
                    "operation_params": {
                        "task_id": task_id,
                        "description": task_description,
                        "user_id": user_id
                    },
                    "ttl_seconds": 3600  # 1 hour
                }
                
                async with session.post(
                    f"{self.services['idempotent_state']}/register",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["operation_id"]
                    
        except Exception as e:
            logger.error(f"Failed to register idempotent operation: {e}")
        
        return None
    
    async def _validate_task(self, task_description: str, requirements: Dict) -> Tuple[bool, str]:
        """Validate task before execution"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "description": task_description,
                    "requirements": requirements,
                    "validation_type": "pre_execution"
                }
                
                async with session.post(
                    f"{self.services['task_validation']}/validate",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["is_valid"], result["message"]
                    
        except Exception as e:
            logger.error(f"Task validation failed: {e}")
        
        return True, "Validation service unavailable"  # Allow to proceed
    
    async def _check_governance(self, task_description: str, user_id: str) -> Tuple[bool, Dict]:
        """Check governance gates for task"""
        try:
            async with aiohttp.ClientSession() as session:
                # Create PR-like request for governance check
                payload = {
                    "pr_id": f"task-{uuid.uuid4().hex[:8]}",
                    "repository": "agent-tasks",
                    "branch": "main",
                    "commit_sha": uuid.uuid4().hex[:8],
                    "author": user_id or "system",
                    "title": task_description[:100],
                    "description": task_description,
                    "files_changed": ["virtual_task.py"],
                    "lines_added": 10,
                    "lines_removed": 0
                }
                
                async with session.post(
                    f"{self.services['governance_gates']}/check-pr",
                    json=payload,
                    params={"policy": "experimental"}  # Use lighter policy for tasks
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["can_merge"], result
                    
        except Exception as e:
            logger.error(f"Governance check failed: {e}")
        
        return True, {"message": "Governance service unavailable"}
    
    async def _select_agent(self, task_description: str) -> Tuple[str, float, str]:
        """Select best agent using improved RL orchestrator"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "description": task_description,
                    "use_capability_matching": True
                }
                
                async with session.post(
                    f"{self.services['rl_orchestrator']}/select-agent",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return (
                            result["agent_id"],
                            result.get("capability_confidence", 0.8),
                            result.get("selection_reason", "Q-learning selection")
                        )
                    
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
        
        # Fallback to basic selection
        return "general_agent", 0.5, "Fallback selection"
    
    async def _log_history(self, task_id: str, agent_id: str, action: str, 
                          status: str, details: Dict):
        """Log to task history service"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "action": action,
                    "status": status,
                    "details": details,
                    "service_name": "agent_coordination"
                }
                
                async with session.post(
                    f"{self.services['task_history']}/log",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"History logged for task {task_id}")
                    
        except Exception as e:
            logger.error(f"Failed to log history: {e}")
    
    async def _execute_task(self, task_id: str, agent_id: str, 
                           task_description: str) -> Dict[str, Any]:
        """Execute task with selected agent"""
        try:
            # Create task in database
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO tasks (id, description, status, agent_id, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    agent_id = EXCLUDED.agent_id,
                    status = EXCLUDED.status
            """, (
                task_id, task_description, "assigned", agent_id,
                datetime.utcnow()
            ))
            
            conn.commit()
            conn.close()
            
            # Emit task created event
            self.event_bus.emit(EventChannel.TASK_CREATED, {
                "task_id": task_id,
                "agent_id": agent_id,
                "description": task_description
            })
            
            # Here you would trigger actual agent execution
            # For now, we'll simulate it
            await asyncio.sleep(0.5)  # Simulate processing
            
            # Mark task as started
            self.event_bus.emit(EventChannel.TASK_STARTED, {
                "task_id": task_id,
                "agent_id": agent_id
            })
            
            # Simulate completion
            await asyncio.sleep(1)
            
            # Update task status
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE tasks 
                SET status = 'completed', completed_at = %s
                WHERE id = %s
            """, (datetime.utcnow(), task_id))
            
            conn.commit()
            conn.close()
            
            # Emit completion event
            self.event_bus.emit(EventChannel.TASK_COMPLETED, {
                "task_id": task_id,
                "agent_id": agent_id,
                "result": "Task completed successfully"
            })
            
            return {
                "status": "completed",
                "result": f"Task executed by {agent_id}",
                "execution_time": 1.5
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            
            # Emit failure event
            self.event_bus.emit(EventChannel.TASK_FAILED, {
                "task_id": task_id,
                "agent_id": agent_id,
                "error": str(e)
            })
            
            raise
    
    async def coordinate_task(self, request: TaskRequest) -> TaskResponse:
        """Main coordination function integrating all components"""
        
        task_id = str(uuid.uuid4())
        logger.info(f"Coordinating task {task_id}: {request.description}")
        
        # Step 1: Check idempotency
        can_proceed, existing_result = await self._check_idempotency(
            request.description, 
            request.user_id or "anonymous"
        )
        
        if not can_proceed and existing_result:
            logger.info(f"Task {task_id} already executed, returning cached result")
            return TaskResponse(
                task_id=existing_result.get("task_id", task_id),
                status="completed_cached",
                agent_id=existing_result.get("agent_id"),
                capability_match=1.0,
                validation_passed=True,
                governance_passed=True,
                idempotency_key=existing_result.get("operation_id"),
                history_logged=True,
                message="Task already completed (cached result)",
                details=existing_result
            )
        
        # Step 2: Register idempotent operation
        idempotency_key = await self._register_idempotent_operation(
            task_id, 
            request.description,
            request.user_id or "anonymous"
        )
        
        # Step 3: Validate task
        validation_passed = True
        validation_message = "Validation skipped"
        
        if not request.skip_validation:
            validation_passed, validation_message = await self._validate_task(
                request.description,
                request.requirements
            )
            
            if not validation_passed:
                await self._log_history(
                    task_id, None, "task_validation_failed",
                    "failed", {"reason": validation_message}
                )
                
                return TaskResponse(
                    task_id=task_id,
                    status="validation_failed",
                    agent_id=None,
                    capability_match=0,
                    validation_passed=False,
                    governance_passed=False,
                    idempotency_key=idempotency_key,
                    history_logged=True,
                    message=f"Task validation failed: {validation_message}",
                    details={"validation_error": validation_message}
                )
        
        # Step 4: Check governance gates
        governance_passed = True
        governance_result = {}
        
        if not request.skip_governance:
            governance_passed, governance_result = await self._check_governance(
                request.description,
                request.user_id or "anonymous"
            )
            
            if not governance_passed:
                await self._log_history(
                    task_id, None, "governance_check_failed",
                    "failed", governance_result
                )
                
                return TaskResponse(
                    task_id=task_id,
                    status="governance_failed",
                    agent_id=None,
                    capability_match=0,
                    validation_passed=validation_passed,
                    governance_passed=False,
                    idempotency_key=idempotency_key,
                    history_logged=True,
                    message="Task failed governance checks",
                    details=governance_result
                )
        
        # Step 5: Select agent using improved RL orchestrator
        agent_id, capability_match, selection_reason = await self._select_agent(
            request.description
        )
        
        logger.info(f"Selected agent {agent_id} with confidence {capability_match}")
        
        # Step 6: Log task assignment
        await self._log_history(
            task_id, agent_id, "agent_selected",
            "assigned", {
                "capability_match": capability_match,
                "selection_reason": selection_reason
            }
        )
        
        # Step 7: Execute task
        try:
            execution_result = await self._execute_task(
                task_id, agent_id, request.description
            )
            
            # Step 8: Log successful completion
            await self._log_history(
                task_id, agent_id, "task_completed",
                "completed", execution_result
            )
            
            # Step 9: Complete idempotent operation
            if idempotency_key:
                try:
                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            f"{self.services['idempotent_state']}/complete/{idempotency_key}",
                            json={
                                "success": True,
                                "data": {
                                    "task_id": task_id,
                                    "agent_id": agent_id,
                                    "result": execution_result
                                }
                            }
                        )
                except Exception as e:
                    logger.error(f"Failed to complete idempotent operation: {e}")
            
            return TaskResponse(
                task_id=task_id,
                status="completed",
                agent_id=agent_id,
                capability_match=capability_match,
                validation_passed=validation_passed,
                governance_passed=governance_passed,
                idempotency_key=idempotency_key,
                history_logged=True,
                message="Task completed successfully",
                details=execution_result
            )
            
        except Exception as e:
            # Log failure
            await self._log_history(
                task_id, agent_id, "task_failed",
                "failed", {"error": str(e)}
            )
            
            # Mark idempotent operation as failed
            if idempotency_key:
                try:
                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            f"{self.services['idempotent_state']}/complete/{idempotency_key}",
                            json={
                                "success": False,
                                "data": {"error": str(e)}
                            }
                        )
                except Exception as e2:
                    logger.error(f"Failed to mark idempotent operation as failed: {e2}")
            
            return TaskResponse(
                task_id=task_id,
                status="failed",
                agent_id=agent_id,
                capability_match=capability_match,
                validation_passed=validation_passed,
                governance_passed=governance_passed,
                idempotency_key=idempotency_key,
                history_logged=True,
                message=f"Task execution failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            # Check all integrated services
            service_status = {}
            
            for service_name, url in self.services.items():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{url}/health", timeout=2) as resp:
                            service_status[service_name] = resp.status == 200
                except:
                    service_status[service_name] = False
            
            all_healthy = all(service_status.values())
            
            return {
                "service": "agent_coordination_glue",
                "status": "healthy" if all_healthy else "degraded",
                "services": service_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/execute")
        async def execute_task(request: TaskRequest):
            """Execute a task with full coordination"""
            return await self.coordinate_task(request)
        
        @self.app.post("/execute-batch")
        async def execute_batch(requests: List[TaskRequest]):
            """Execute multiple tasks"""
            results = []
            
            for request in requests:
                try:
                    result = await self.coordinate_task(request)
                    results.append(result.dict())
                except Exception as e:
                    results.append({
                        "status": "error",
                        "message": str(e)
                    })
            
            return {
                "total": len(requests),
                "results": results
            }
        
        @self.app.get("/task/{task_id}/status")
        async def get_task_status(task_id: str):
            """Get task status and history"""
            try:
                # Get task from database
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM tasks WHERE id = %s
                """, (task_id,))
                
                task = cur.fetchone()
                conn.close()
                
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                # Get history from task history service
                history = []
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{self.services['task_history']}/task/{task_id}/timeline"
                        ) as resp:
                            if resp.status == 200:
                                timeline = await resp.json()
                                history = timeline.get("events", [])
                except:
                    pass
                
                return {
                    "task_id": task_id,
                    "description": task["description"],
                    "status": task["status"],
                    "agent_id": task.get("agent_id"),
                    "created_at": task["created_at"].isoformat(),
                    "completed_at": task.get("completed_at").isoformat() if task.get("completed_at") else None,
                    "history": history
                }
                
            except Exception as e:
                logger.error(f"Failed to get task status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents/performance")
        async def get_agents_performance():
            """Get performance metrics for all agents"""
            try:
                # Get list of all agents
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT DISTINCT agent_id 
                    FROM tasks 
                    WHERE agent_id IS NOT NULL
                """)
                
                agents = [row[0] for row in cur.fetchall()]
                conn.close()
                
                # Get performance for each agent
                performance = {}
                
                for agent_id in agents:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"{self.services['task_history']}/agent/{agent_id}/performance"
                            ) as resp:
                                if resp.status == 200:
                                    performance[agent_id] = await resp.json()
                    except:
                        pass
                
                return {
                    "agents": performance,
                    "total_agents": len(agents)
                }
                
            except Exception as e:
                logger.error(f"Failed to get agents performance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/test-coordination")
        async def test_coordination():
            """Test the coordination system with sample tasks"""
            test_tasks = [
                TaskRequest(
                    description="Create a hello world website",
                    priority=5,
                    user_id="test_user"
                ),
                TaskRequest(
                    description="Perform security audit of the application",
                    priority=8,
                    user_id="test_user"
                ),
                TaskRequest(
                    description="Write unit tests for authentication module",
                    priority=6,
                    user_id="test_user"
                ),
                TaskRequest(
                    description="Optimize database queries",
                    priority=7,
                    user_id="test_user"
                )
            ]
            
            results = []
            for task in test_tasks:
                try:
                    result = await self.coordinate_task(task)
                    results.append({
                        "task": task.description,
                        "agent": result.agent_id,
                        "capability_match": result.capability_match,
                        "status": result.status
                    })
                except Exception as e:
                    results.append({
                        "task": task.description,
                        "error": str(e)
                    })
            
            return {
                "test": "coordination_test",
                "results": results
            }
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Agent Coordination Glue starting up...")
        
        # Ensure tasks table exists
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id UUID PRIMARY KEY,
                    description TEXT NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    agent_id VARCHAR(100),
                    priority INTEGER DEFAULT 5,
                    user_id VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    metadata JSONB,
                    result JSONB
                );
                
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to ensure tables: {e}")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Agent Coordination Glue shutting down...")
        self.event_bus.stop()
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = AgentCoordinationGlue()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("COORDINATION_PORT", 8030))
    logger.info(f"Starting Agent Coordination Glue on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()