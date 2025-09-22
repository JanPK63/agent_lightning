#!/usr/bin/env python3
"""
RL Orchestrator Service - Complete implementation for task assignment and execution
Runs on port 8025 and provides the /assign-task endpoint expected by the system
"""

import os
import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Async HTTP client for probing agent health endpoints
import httpx

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging

# Import our task execution components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_executor_fix import TaskExecutionBridge, AgentExecutor
from agent_capability_matcher import AgentCapabilityMatcher
from services.ai_client_manager import AIClientManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class TaskAssignmentRequest(BaseModel):
    """Request for task assignment with capability validation"""
    task_id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Task description")
    priority: int = Field(5, ge=1, le=10, description="Task priority (1-10)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key for duplicate prevention")
    force_execute: bool = Field(False, description="Allow overriding confidence checks")

    # Environment and execution context fields
    target_environment: Optional[str] = Field("local", description="Target environment: 'local', 'remote', 'ubuntu', 'macos'")
    working_directory: Optional[str] = Field(None, description="Specific working directory for task execution")
    search_scope: Optional[str] = Field("local", description="Search scope: 'local', 'internet', 'both'")
    platform: Optional[str] = Field(None, description="Target platform: 'macos', 'linux', 'windows'")
    environment_vars: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    execution_mode: Optional[str] = Field("standard", description="Execution mode: 'standard', 'isolated', 'container'")

class TaskAssignmentResponse(BaseModel):
    """Response from task assignment"""
    task_id: str = Field(..., description="Unique task identifier")
    assigned_agent: str = Field(..., description="Agent assigned to the task")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assignment confidence score")
    status: str = Field(..., pattern="^(assigned|queued|failed)$", description="Task status")
    reason: str = Field(..., description="Reason for assignment decision")
    validation_passed: bool = Field(..., description="Whether validation passed")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    target_environment: str = Field(..., description="Target execution environment")
    working_directory: Optional[str] = Field(None, description="Working directory for execution")

class QTableResponse(BaseModel):
    """Q-table status response"""
    states: List[str] = Field(..., description="Learned states")
    total_entries: int = Field(..., description="Total Q-table entries")
    q_table: Dict[str, Dict[str, float]] = Field(..., description="Q-table data")

class RLOrchestratorService:
    """Complete RL Orchestrator Service for task assignment and execution"""

    def __init__(self):
        self.app = FastAPI(title="RL Orchestrator Service", version="1.0.0")

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize components
        self.task_bridge = TaskExecutionBridge()
        self.capability_matcher = AgentCapabilityMatcher()
        self.agent_executor = AgentExecutor()
        self.ai_client_manager = AIClientManager()

        # In-memory storage for task tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_history: List[Dict[str, Any]] = []

        # Simple Q-learning state
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9

        # Runtime agent registry populated via /agents/register
        # Structure: { agent_id: { "agent_id", "health_url", "execute_url", "capabilities": [...], "meta": {...} } }
        self.agent_registry: Dict[str, Dict[str, Any]] = {}

        # Seed Q-learning priors for common states
        self._seed_q_priors({
            "web_development": {"web_developer": 0.65},
            "data_analysis": {"data_analyst": 0.65},
            "security_task": {"security_expert": 0.65},
            "deployment_task": {"devops_engineer": 0.65},
            "general_task": {"general_assistant": 0.55, "web_developer": 0.60}
        })

        # Seed Q-learning priors for common states
        self._seed_q_priors({
            "web_development": {"web_developer": 0.65, "full_stack_developer": 0.6},
            "data_analysis": {"data_analyst": 0.65, "data_scientist": 0.6},
            "security_task": {"security_expert": 0.65},
            "deployment_task": {"devops_engineer": 0.65},
            "general_task": {"general_assistant": 0.55, "full_stack_developer": 0.6}
        })

        # Setup routes
        self._setup_routes()

        logger.info("✅ RL Orchestrator Service initialized")

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            """Service health check"""
            return {
                "service": "RL Orchestrator",
                "status": "healthy",
                "version": "1.0.0",
                "endpoints": [
                    "GET /",
                    "POST /assign-task",
                    "GET /tasks/{task_id}",
                    "GET /q-table",
                    "POST /task-feedback"
                ]
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for service monitoring"""
            return {
                "status": "healthy",
                "service": "RL Orchestrator",
                "timestamp": datetime.now().isoformat(),
                "active_tasks": len(self.active_tasks),
                "total_tasks_processed": len(self.task_history)
            }

        async def _probe_agent_health(agent_meta: Dict[str, Any], timeout: float = 3.0) -> Dict[str, Any]:
            """
            Probe an agent's health_url if provided and return status dict:
            { "agent_id": id, "name": name, "health": "ok"|"unreachable"|"unknown", "details": ... }
            """
            agent_id = agent_meta.get("agent_id")
            name = agent_meta.get("name")
            health_url = agent_meta.get("health_url")
            result = {
                "agent_id": agent_id,
                "name": name,
                "health": "unknown",
                "health_url": health_url,
                "details": None
            }

            if not health_url:
                # No probe URL available
                result["health"] = "no_probe_url"
                return result

            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        result["health"] = "ok"
                        result["details"] = resp.json() if resp.headers.get("content-type","").startswith("application/json") else resp.text[:500]
                    else:
                        result["health"] = "unhealthy"
                        result["details"] = {"status_code": resp.status_code, "text": resp.text[:500]}
            except Exception as e:
                result["health"] = "unreachable"
                result["details"] = str(e)

            return result

        @self.app.post("/agents/register")
        async def register_agent(payload: Dict[str, Any]):
            """
            Register an agent with the orchestrator at runtime.
            Payload should include:
              - agent_id (str)
              - health_url (optional)
              - execute_url (optional)
              - capabilities (optional list)
              - meta (optional dict)
            """
            try:
                agent_id = payload.get("agent_id")
                if not agent_id:
                    raise HTTPException(status_code=400, detail="agent_id is required")

                entry = {
                    "agent_id": agent_id,
                    "health_url": payload.get("health_url"),
                    "execute_url": payload.get("execute_url"),
                    "capabilities": payload.get("capabilities", []),
                    "meta": payload.get("meta", {})
                }
                self.agent_registry[agent_id] = entry
                logger.info(f"Registered agent {agent_id} with registry: {entry}")
                return {"status": "registered", "agent_id": agent_id, "entry": entry}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to register agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/agents")
        async def list_agents():
            """
            List known agents (from capability matcher and runtime registry) and attempt to probe their health_url (if present).
            Returns an array of agent metas with a health field.
            """
            try:
                agents = []
                # Pull capability matcher agents if available
                try:
                    matcher_agents = self.capability_matcher.get_available_agents()
                except Exception:
                    matcher_agents = []

                # Build agent metadata list (include any known details)
                for agent_id in matcher_agents:
                    agent_obj = self.capability_matcher.get_agent_info(agent_id)
                    meta = {
                        "agent_id": agent_id,
                        "name": agent_obj.name if agent_obj else agent_id,
                        "capabilities": getattr(agent_obj, "capabilities", []),
                        "base_confidence": getattr(agent_obj, "base_confidence", None),
                        "health_url": getattr(agent_obj, "health_url", None)
                    }
                    agents.append(meta)

                # Append runtime-registered agents (may add/override metadata)
                for agent_id, entry in self.agent_registry.items():
                    # Avoid duplicates: if already present from matcher, merge
                    existing = next((a for a in agents if a["agent_id"] == agent_id), None)
                    if existing:
                        existing.update({
                            "health_url": entry.get("health_url") or existing.get("health_url"),
                            "capabilities": list(set(existing.get("capabilities", []) + entry.get("capabilities", []))),
                            "registered": True
                        })
                    else:
                        agents.append({
                            "agent_id": agent_id,
                            "name": entry.get("meta", {}).get("name", agent_id),
                            "capabilities": entry.get("capabilities", []),
                            "base_confidence": None,
                            "health_url": entry.get("health_url"),
                            "registered": True
                        })

                # Probe all agents that expose health_url concurrently
                probe_tasks = [ _probe_agent_health(a) for a in agents ]
                probe_results = await asyncio.gather(*probe_tasks, return_exceptions=False)

                # Merge probe results back into agent list
                health_by_id = { p["agent_id"]: p for p in probe_results }
                for a in agents:
                    a["health"] = health_by_id.get(a["agent_id"], {}).get("health")
                    a["health_details"] = health_by_id.get(a["agent_id"], {}).get("details")

                return {"agents": agents, "count": len(agents)}

            except Exception as e:
                logger.error(f"Failed to list agents: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/agents/{agent_id}/ping")
        async def ping_agent(agent_id: str):
            """Ping a single agent by id (prefers runtime registry entry then capability matcher)"""
            try:
                entry = self.agent_registry.get(agent_id)
                if entry:
                    meta = {
                        "agent_id": agent_id,
                        "name": entry.get("meta", {}).get("name", agent_id),
                        "health_url": entry.get("health_url")
                    }
                    if not meta["health_url"]:
                        meta["health"] = "no_probe_url"
                        return meta
                    return await _probe_agent_health(meta)

                agent_obj = self.capability_matcher.get_agent_info(agent_id)
                if not agent_obj:
                    raise HTTPException(status_code=404, detail="Agent not known")

                health_url = getattr(agent_obj, "health_url", None)
                meta = {
                    "agent_id": agent_id,
                    "name": agent_obj.name,
                    "health_url": health_url
                }

                if not health_url:
                    meta["health"] = "no_probe_url"
                    return meta

                probe = await _probe_agent_health(meta)
                return probe

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to ping agent {agent_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/assign-task", response_model=TaskAssignmentResponse)
        async def assign_task(request: TaskAssignmentRequest, background_tasks: BackgroundTasks):
            """Assign a task to the most suitable agent"""
            try:
                logger.info(f"Received task assignment request: {request.task_id}")

                # Check for duplicate task (idempotency)
                if request.idempotency_key:
                    existing_task = self._find_task_by_idempotency_key(request.idempotency_key)
                    if existing_task:
                        logger.info(f"Returning existing task: {existing_task['task_id']}")
                        return TaskAssignmentResponse(**existing_task)

                # Match task to best agent
                agent_match = await self._match_task_to_agent(request.description)

                if not agent_match:
                    # Fallback to auto-selection
                    agent_match = {
                        "agent_id": "full_stack_developer",
                        "confidence": 0.5,
                        "reason": "Fallback to full_stack_developer due to no matches"
                    }

                # Validate assignment if confidence is low
                validation_passed = True
                warnings = []

                if agent_match["confidence"] < 0.6 and not request.force_execute:
                    validation_passed = False
                    warnings.append(f"Low confidence ({agent_match['confidence']:.2f}) - task queued for review")

                if request.priority >= 8 and agent_match["confidence"] < 0.8:
                    warnings.append("High priority task assigned to agent with moderate confidence")

                # Validate environment and directory
                if request.working_directory and not request.working_directory.startswith("/"):
                    warnings.append("Working directory should be an absolute path")

                if request.target_environment not in ["local", "remote", "ubuntu", "macos"]:
                    warnings.append(f"Unknown target environment: {request.target_environment}")

                # Create task record
                task_record = {
                    "task_id": request.task_id,
                    "description": request.description,
                    "assigned_agent": agent_match["agent_id"],
                    "confidence": agent_match["confidence"],
                    "status": "assigned" if validation_passed else "queued",
                    "reason": agent_match["reason"],
                    "validation_passed": validation_passed,
                    "warnings": warnings,
                    "priority": request.priority,
                    "metadata": request.metadata or {},
                    "idempotency_key": request.idempotency_key,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),

                    # Environment and execution context
                    "target_environment": request.target_environment,
                    "working_directory": request.working_directory,
                    "search_scope": request.search_scope,
                    "platform": request.platform,
                    "environment_vars": request.environment_vars or {},
                    "execution_mode": request.execution_mode
                }

                # Store task
                self.active_tasks[request.task_id] = task_record
                self.task_history.append(task_record)

                # Execute task if validation passed (or when force_execute) - schedule reliably
                # We use both BackgroundTasks (FastAPI) and asyncio.create_task to ensure execution
                # in environments where BackgroundTasks may not be processed immediately.
                if validation_passed or request.force_execute:
                    try:
                        # Register with FastAPI background tasks for compatibility
                        background_tasks.add_task(self._execute_task_async, request.task_id, agent_match["agent_id"])
                        # Also schedule an immediate asyncio task to avoid missed BackgroundTasks
                        asyncio.create_task(self._execute_task_async(request.task_id, agent_match["agent_id"]))
                        logger.info(f"Scheduled execution for task {request.task_id} (agent: {agent_match['agent_id']})")
                    except Exception as e:
                        logger.warning(f"Could not schedule execution via asyncio for task {request.task_id}: {e}")

                # Update Q-learning
                self._update_q_learning(request.description, agent_match["agent_id"], agent_match["confidence"])

                logger.info(f"Task {request.task_id} assigned to {agent_match['agent_id']} with confidence {agent_match['confidence']:.2f}")

                return TaskAssignmentResponse(**task_record)

            except Exception as e:
                logger.error(f"Task assignment failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/execute-now")
        async def execute_now(payload: Dict[str, Any]):
            """
            Force immediate execution of an existing task.
            Payload: { "task_id": "<id>" }
            """
            try:
                task_id = payload.get("task_id")
                if not task_id:
                    raise HTTPException(status_code=400, detail="task_id is required")

                # Find task
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                else:
                    task = next((t for t in self.task_history if t["task_id"] == task_id), None)
                    if task:
                        # move into active tasks for execution
                        self.active_tasks[task_id] = task

                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")

                # Determine agent to run
                agent_id = task.get("assigned_agent") or task.get("agent_id") or "full_stack_developer"

                # Schedule execution immediately and return status
                try:
                    asyncio.create_task(self._execute_task_async(task_id, agent_id))
                    return {"status": "scheduled", "task_id": task_id, "assigned_agent": agent_id}
                except Exception as e:
                    logger.error(f"Failed to schedule execution for {task_id}: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in execute-now: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/tasks/{task_id}")
        async def get_task_status(task_id: str):
            """Get task status"""
            try:
                if task_id in self.active_tasks:
                    return self.active_tasks[task_id]

                # Check history
                for task in self.task_history:
                    if task["task_id"] == task_id:
                        return task

                raise HTTPException(status_code=404, detail="Task not found")

            except Exception as e:
                logger.error(f"Failed to get task status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/q-table", response_model=QTableResponse)
        async def get_q_table():
            """Get Q-learning table status"""
            try:
                states = list(self.q_table.keys())
                total_entries = sum(len(actions) for actions in self.q_table.values())

                return QTableResponse(
                    states=states,
                    total_entries=total_entries,
                    q_table=self.q_table
                )

            except Exception as e:
                logger.error(f"Failed to get Q-table: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/task-feedback")
        async def provide_feedback(feedback: Dict[str, Any]):
            """Provide feedback for Q-learning improvement"""
            try:
                task_id = feedback.get("task_id")
                agent_id = feedback.get("agent_id")
                performance_score = feedback.get("performance_score", 5)
                suggestions = feedback.get("suggestions", "")

                # Convert performance to reward
                reward = (performance_score - 5) / 5  # Normalize to -1 to 1

                # Update Q-learning with feedback
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    task_description = task["description"]

                    # Update Q-value based on feedback
                    state = self._extract_task_state(task_description)
                    if state in self.q_table and agent_id in self.q_table[state]:
                        current_q = self.q_table[state][agent_id]
                        # Simple Q-learning update
                        new_q = current_q + self.learning_rate * (reward - current_q)
                        self.q_table[state][agent_id] = new_q

                logger.info(f"Received feedback for agent {agent_id}: score={performance_score}, reward={reward}")

                return {
                    "status": "accepted",
                    "agent_id": agent_id,
                    "reward_signal": reward,
                    "message": "Feedback recorded for Q-learning improvement"
                }

            except Exception as e:
                logger.error(f"Error processing feedback: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _match_task_to_agent(self, task_description: str) -> Optional[Dict[str, Any]]:
        """Match task to best available agent, preferring reachable (healthy) agents.

        Strategy:
        - Gather candidate agents from capability matcher and runtime registry.
        - Compute a confidence score per candidate using capability matcher.
        - Probe each candidate's health_url (if available) concurrently.
        - Prefer agents reporting health 'ok' and choose highest-confidence among them.
        - If no healthy agents, fall back to highest-confidence candidate.
        - If nothing found, apply simple keyword fallback.
        """
        try:
            task_lower = task_description.lower()

            # Gather candidate agent ids from capability matcher
            try:
                matcher_agent_ids = self.capability_matcher.get_available_agents()
            except Exception:
                matcher_agent_ids = []

            candidates = []

            # Helper to compute confidence using capability matcher internals
            for agent_id in matcher_agent_ids:
                agent_obj = self.capability_matcher.get_agent_info(agent_id)
                try:
                    # capability matcher exposes _calculate_match_confidence; use it to compute per-agent score
                    confidence, reason = self.capability_matcher._calculate_match_confidence(task_lower, agent_obj)
                except Exception:
                    # Fallback: use agent base_confidence if available
                    confidence = getattr(agent_obj, "base_confidence", 0.5)
                    reason = f"{agent_id}: base confidence fallback"

                # Determine probe URL (prefer capability matcher health_url, then runtime registry)
                health_url = getattr(agent_obj, "health_url", None)
                registry_entry = self.agent_registry.get(agent_id)
                if registry_entry and registry_entry.get("health_url"):
                    health_url = registry_entry.get("health_url")

                execute_url = None
                if registry_entry and registry_entry.get("execute_url"):
                    execute_url = registry_entry.get("execute_url")

                candidates.append({
                    "agent_id": agent_id,
                    "confidence": confidence,
                    "reason": reason,
                    "health_url": health_url,
                    "execute_url": execute_url,
                    "agent_obj": agent_obj
                })

            # Also include runtime-registered agents that may not be in capability matcher
            for agent_id, entry in self.agent_registry.items():
                if any(c["agent_id"] == agent_id for c in candidates):
                    continue
                candidates.append({
                    "agent_id": agent_id,
                    "confidence": entry.get("meta", {}).get("base_confidence", 0.5),
                    "reason": f"{agent_id}: registered runtime agent",
                    "health_url": entry.get("health_url"),
                    "execute_url": entry.get("execute_url"),
                    "agent_obj": None
                })

            # If we have health_urls to probe, probe them concurrently
            async def _probe_health(c):
                hurl = c.get("health_url")
                if not hurl:
                    c["health"] = "no_probe_url"
                    return c
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        resp = await client.get(hurl)
                        if resp.status_code == 200:
                            c["health"] = "ok"
                            # try to attach some details if available
                            try:
                                c["health_details"] = resp.json()
                            except Exception:
                                c["health_details"] = resp.text[:200]
                        else:
                            c["health"] = "unhealthy"
                            c["health_details"] = {"status_code": resp.status_code, "text": resp.text[:200]}
                except Exception as e:
                    c["health"] = "unreachable"
                    c["health_details"] = str(e)
                return c

            probe_tasks = [_probe_health(c) for c in candidates]
            if probe_tasks:
                # run probes but ignore exceptions to let selection fall back
                try:
                    probe_results = await asyncio.gather(*probe_tasks, return_exceptions=False)
                except Exception:
                    probe_results = candidates
            else:
                probe_results = candidates

            # Prefer agents that are healthy
            healthy = [c for c in probe_results if c.get("health") == "ok"]
            if healthy:
                # pick highest confidence among healthy
                best = max(healthy, key=lambda x: x.get("confidence", 0.0))
                logger.info(f"Selected healthy agent {best['agent_id']} (conf={best['confidence']:.3f}) for task")
                return {
                    "agent_id": best["agent_id"],
                    "confidence": best["confidence"],
                    "reason": best.get("reason", "Matched healthy agent")
                }

            # No healthy agents: pick highest confidence overall
            if probe_results:
                best = max(probe_results, key=lambda x: x.get("confidence", 0.0))
                logger.info(f"No healthy agents found; selected best candidate {best['agent_id']} (conf={best['confidence']:.3f})")
                return {
                    "agent_id": best["agent_id"],
                    "confidence": best["confidence"],
                    "reason": best.get("reason", "Best available candidate")
                }

            # Final keyword-based fallback (same logic as before)
            if "web" in task_lower or "website" in task_lower or "frontend" in task_lower:
                return {
                    "agent_id": "full_stack_developer",
                    "confidence": 0.8,
                    "reason": "Task involves web development (fallback)"
                }
            elif "data" in task_lower or "analysis" in task_lower or "machine learning" in task_lower:
                return {
                    "agent_id": "data_scientist",
                    "confidence": 0.8,
                    "reason": "Task involves data analysis or ML (fallback)"
                }
            elif "security" in task_lower or "audit" in task_lower or "vulnerability" in task_lower:
                return {
                    "agent_id": "security_expert",
                    "confidence": 0.8,
                    "reason": "Task involves security analysis (fallback)"
                }
            elif "deploy" in task_lower or "infrastructure" in task_lower or "kubernetes" in task_lower:
                return {
                    "agent_id": "devops_engineer",
                    "confidence": 0.8,
                    "reason": "Task involves deployment or infrastructure (fallback)"
                }
            else:
                return {
                    "agent_id": "full_stack_developer",
                    "confidence": 0.6,
                    "reason": "General development task - defaulting to full_stack_developer (fallback)"
                }

        except Exception as e:
            logger.error(f"Agent matching failed: {e}")
            return None

    async def _execute_task_async(self, task_id: str, agent_id: str):
        """Execute task asynchronously"""
        try:
            if task_id not in self.active_tasks:
                logger.error(f"Task {task_id} not found for execution")
                return

            task = self.active_tasks[task_id]
            task["status"] = "running"
            task["updated_at"] = datetime.now().isoformat()

            logger.info(f"Executing task {task_id} with agent {agent_id}")

            # Prepare comprehensive execution context
            execution_context = task.get("metadata", {})
            execution_context.update({
                "target_environment": task.get("target_environment", "local"),
                "working_directory": task.get("working_directory"),
                "search_scope": task.get("search_scope", "local"),
                "platform": task.get("platform"),
                "environment_vars": task.get("environment_vars", {}),
                "execution_mode": task.get("execution_mode", "standard"),
                "task_id": task_id
            })

            logger.info(f"Executing task {task_id} in environment: {task.get('target_environment', 'local')}, directory: {task.get('working_directory', 'default')}")

            # Execute the task using the task bridge
            result = await self.task_bridge.execute_agent_task(
                agent_id=agent_id,
                task_description=task["description"],
                context=execution_context
            )

            # Update task status
            task["status"] = "completed" if result.get("status") == "completed" else "failed"
            task["result"] = result
            task["updated_at"] = datetime.now().isoformat()

            # Auto-feedback to accelerate learning on success
            if task["status"] == "completed":
                self._apply_success_feedback(task["description"], agent_id, performance_score=8)

            # Auto-feedback to accelerate learning on success
            if task["status"] == "completed":
                self._apply_success_feedback(task["description"], agent_id, performance_score=8)

            logger.info(f"Task {task_id} completed with status: {task['status']}")

        except Exception as e:
            logger.error(f"Task execution failed for {task_id}: {e}")
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)
                self.active_tasks[task_id]["updated_at"] = datetime.now().isoformat()

    def _find_task_by_idempotency_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Find existing task by idempotency key"""
        for task in self.task_history:
            if task.get("idempotency_key") == key:
                return task
        return None

    def _extract_task_state(self, task_description: str) -> str:
        """Extract state representation from task description"""
        # Simple state extraction based on keywords
        task_lower = task_description.lower()

        if "web" in task_lower or "website" in task_lower:
            return "web_development"
        elif "data" in task_lower or "analysis" in task_lower:
            return "data_analysis"
        elif "security" in task_lower:
            return "security_task"
        elif "deploy" in task_lower:
            return "deployment_task"
        else:
            return "general_task"

    def _update_q_learning(self, task_description: str, agent_id: str, confidence: float):
        """Update Q-learning table"""
        state = self._extract_task_state(task_description)

        if state not in self.q_table:
            self.q_table[state] = {}

        if agent_id not in self.q_table[state]:
            self.q_table[state][agent_id] = 0.0

        # Update Q-value based on confidence
        current_q = self.q_table[state][agent_id]
        reward = confidence  # Use confidence as reward
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table[state][agent_id] = new_q

        def _seed_q_priors(self, priors: Dict[str, Dict[str, float]]):
            """Initialize Q-table with prior values for faster early learning"""
            for state, actions in priors.items():
                if state not in self.q_table:
                    self.q_table[state] = {}
                for agent, q in actions.items():
                    if agent not in self.q_table[state]:
                        self.q_table[state][agent] = q

        def _apply_success_feedback(self, task_description: str, agent_id: str, performance_score: int = 8):
            """Apply positive feedback to Q-table after successful execution"""
            try:
                state = self._extract_task_state(task_description)
                if state not in self.q_table:
                    self.q_table[state] = {}
                if agent_id not in self.q_table[state]:
                    self.q_table[state][agent_id] = 0.0

                # Convert performance to reward (-1..1). Score 8 ≈ +0.6 reward.
                reward = (performance_score - 5) / 5
                current_q = self.q_table[state][agent_id]
                new_q = current_q + self.learning_rate * (reward - current_q)
                self.q_table[state][agent_id] = new_q
                logger.info(f"Applied auto-feedback: state={state}, agent={agent_id}, reward={reward:.2f}, Q={new_q:.3f}")
            except Exception as e:
                logger.warning(f"Failed to apply success feedback: {e}")

    def _seed_q_priors(self, priors: Dict[str, Dict[str, float]]):
        """Initialize Q-table with prior values for faster early learning"""
        for state, actions in priors.items():
            if state not in self.q_table:
                self.q_table[state] = {}
            for agent, q in actions.items():
                if agent not in self.q_table[state]:
                    self.q_table[state][agent] = q

    def _apply_success_feedback(self, task_description: str, agent_id: str, performance_score: int = 8):
        """Apply positive feedback to Q-table after successful execution"""
        try:
            state = self._extract_task_state(task_description)
            if state not in self.q_table:
                self.q_table[state] = {}
            if agent_id not in self.q_table[state]:
                self.q_table[state][agent_id] = 0.0

            # Convert performance to reward (-1..1). Score 8 ≈ +0.6 reward.
            reward = (performance_score - 5) / 5
            current_q = self.q_table[state][agent_id]
            new_q = current_q + self.learning_rate * (reward - current_q)
            self.q_table[state][agent_id] = new_q
            logger.info(f"Applied auto-feedback: state={state}, agent={agent_id}, reward={reward:.2f}, Q={new_q:.3f}")
        except Exception as e:
            logger.warning(f"Failed to apply success feedback: {e}")

    def run(self, host: str = "0.0.0.0", port: int = 8025):
        """Run the service"""
        logger.info(f"Starting RL Orchestrator Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Global service instance
service = RLOrchestratorService()


# Legacy endpoint definitions (keeping for compatibility)
@service.app.post("/agents/{agent_id}/feedback")
async def provide_feedback(agent_id: str, feedback: dict):
    """
    Provide feedback on agent performance for reinforcement learning

    Args:
        agent_id: The agent ID
        feedback: Dict containing:
            - task_id: The task that was performed
            - performance_score: 0-10 rating of performance
            - suggestions: Text suggestions for improvement
            - issues: List of specific issues found
    """
    try:
        score = feedback.get("performance_score", 5)
        suggestions = feedback.get("suggestions", "")
        issues = feedback.get("issues", [])

        # Convert feedback to reward signal for RL
        reward = (score - 5) / 5  # Normalize to -1 to 1

        logger.info(f"Received feedback for agent {agent_id}: score={score}, reward={reward}")

        return {
            "status": "accepted",
            "agent_id": agent_id,
            "reward_signal": reward,
            "message": f"Feedback recorded. Q-learning updated."
        }

    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the service
    service.run()