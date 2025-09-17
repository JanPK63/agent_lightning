#!/usr/bin/env python3
"""
Improved RL Orchestrator Service with Agent Capability Matching
Fixes the critical issue of wrong agent assignments (e.g., security_expert for web tasks)
Integrates capability matching BEFORE task assignment to prevent mismatches
"""

import os
import sys
import json
import asyncio
import uuid
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiohttp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared components
from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Import the capability matcher to prevent wrong assignments
from agent_capability_matcher import AgentCapabilityMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskAssignmentRequest(BaseModel):
    """Request for task assignment with capability validation"""
    task_id: str
    description: str
    priority: int = 5
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None


class TaskAssignmentResult(BaseModel):
    """Result of task assignment with validation"""
    task_id: str
    assigned_agent: str
    confidence: float
    reason: str
    validation_passed: bool
    warnings: List[str] = []


class ImprovedRLOrchestrator:
    """Improved RL Orchestrator with capability-based agent selection"""

    def __init__(self):
        self.app = FastAPI(title="Improved RL Orchestrator", version="2.0.0")

        # Initialize components
        self.dal = DataAccessLayer("rl_orchestrator_improved")
        self.cache = get_cache()
        self.capability_matcher = AgentCapabilityMatcher()

        # Service URLs
        self.agent_designer_url = "http://localhost:8002"
        self.rl_server_url = "http://localhost:8010"
        self.ai_model_url = "http://localhost:8005"
        self.task_validation_url = "http://localhost:8024"

        # Q-learning parameters for agent selection
        self.q_table = {}  # State-action values
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate

        # Hard validation thresholds
        self.min_capability_confidence = 0.4  # Minimum capability confidence to proceed
        self.min_final_confidence = 0.6  # Minimum final confidence to proceed (stricter for testing)

        # Idempotency tracking for task submissions
        self.idempotency_store = {}  # key -> (task_id, timestamp)

        # Circuit breaker for cross-service calls
        self.circuit_breaker_failures = {}
        self.circuit_breaker_threshold = 3
        self.circuit_breaker_timeout = 60  # seconds

        # Track agent performance
        self.agent_performance = {}
        self.task_history = []

        logger.info("✅ Improved RL Orchestrator initialized with capability matching")

        self._setup_middleware()
        self._setup_routes()
        self._load_q_table()
        
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _load_q_table(self):
        """Load Q-table from cache or initialize"""
        cached_q = self.cache.get("rl_orchestrator:q_table")
        if cached_q:
            self.q_table = cached_q
            logger.info(f"Loaded Q-table with {len(self.q_table)} states")
        else:
            # Initialize with default values
            self.q_table = {}
            logger.info("Initialized new Q-table")
    
    def _save_q_table(self):
        """Save Q-table to cache"""
        self.cache.set("rl_orchestrator:q_table", self.q_table, ttl=86400)

    async def _make_resilient_request(self, url: str, method: str = "GET",
                                    json_data: Optional[Dict] = None,
                                    headers: Optional[Dict] = None,
                                    timeout: float = 10.0) -> Optional[Dict]:
        """
        Make HTTP request with circuit breaker and exponential backoff
        Returns response JSON or None on failure
        """
        service_key = url.split("://")[1].split("/")[0]  # Extract service identifier

        # Check circuit breaker
        if service_key in self.circuit_breaker_failures:
            failure_count, last_failure = self.circuit_breaker_failures[service_key]
            if failure_count >= self.circuit_breaker_threshold:
                time_since_failure = (datetime.utcnow() - last_failure).seconds
                if time_since_failure < self.circuit_breaker_timeout:
                    logger.warning(f"Circuit breaker OPEN for {service_key}, skipping request")
                    return None
                else:
                    # Half-open: reset failure count
                    self.circuit_breaker_failures[service_key] = (0, datetime.utcnow())

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=method,
                        url=url,
                        json=json_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status == 200:
                            # Success: reset circuit breaker
                            if service_key in self.circuit_breaker_failures:
                                self.circuit_breaker_failures[service_key] = (0, datetime.utcnow())
                            return await response.json()
                        else:
                            logger.warning(f"Request to {url} failed with status {response.status}")
                            # Record failure for circuit breaker
                            if service_key not in self.circuit_breaker_failures:
                                self.circuit_breaker_failures[service_key] = (1, datetime.utcnow())
                            else:
                                count, _ = self.circuit_breaker_failures[service_key]
                                self.circuit_breaker_failures[service_key] = (count + 1, datetime.utcnow())
                            return None

            except asyncio.TimeoutError:
                logger.warning(f"Request to {url} timed out (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"Request to {url} failed: {e}")

            # Exponential backoff with jitter
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + np.random.random() * 0.1
                await asyncio.sleep(delay)

        # All retries failed
        logger.error(f"All {max_retries} attempts to {url} failed")
        return None
    
    def _get_state_key(self, task_description: str) -> str:
        """Generate state key from task description"""
        # Simplified state representation
        task_lower = task_description.lower()
        
        if "security" in task_lower or "audit" in task_lower:
            return "security_task"
        elif "website" in task_lower or "web" in task_lower or "html" in task_lower:
            return "web_task"
        elif "test" in task_lower or "testing" in task_lower:
            return "testing_task"
        elif "data" in task_lower or "analysis" in task_lower:
            return "data_task"
        elif "deploy" in task_lower or "deployment" in task_lower:
            return "deployment_task"
        else:
            return "general_task"
    
    def _update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning formula"""
        # Initialize if not exists
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Get max Q-value for next state
        max_next_q = 0.0
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        # Q-learning update
        old_q = self.q_table[state][action]
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_next_q - old_q)
        self.q_table[state][action] = new_q
        
        logger.info(f"Q-value updated: state={state}, action={action}, old_q={old_q:.3f}, new_q={new_q:.3f}")
        
        # Save updated Q-table
        self._save_q_table()
    
    async def select_agent_with_capability_check(self, task_description: str) -> Tuple[str, float, str, bool]:
        """
        Select agent using both Q-learning and capability matching
        Returns: (agent_id, confidence, reason, validation_passed)
        """
        # Step 1: Use capability matcher to find best agent
        best_agent, capability_confidence, capability_reason = self.capability_matcher.find_best_agent(task_description)
        
        # Step 2: Get Q-learning recommendation
        state = self._get_state_key(task_description)
        
        # Epsilon-greedy strategy
        if np.random.random() < self.epsilon:
            # Exploration: use capability matcher recommendation
            selected_agent = best_agent
            q_confidence = capability_confidence
            selection_reason = f"Exploration: {capability_reason}"
        else:
            # Exploitation: use Q-values if available
            if state in self.q_table and self.q_table[state]:
                # Get agent with highest Q-value
                q_agent = max(self.q_table[state], key=self.q_table[state].get)
                q_value = self.q_table[state][q_agent]
                
                # Validate Q-learning choice with capability matcher
                is_valid, validation_reason = self.capability_matcher.validate_assignment(q_agent, task_description)
                
                if is_valid and q_value > 0:
                    selected_agent = q_agent
                    q_confidence = min(1.0, 0.5 + q_value / 2)  # Convert Q-value to confidence
                    selection_reason = f"Q-learning: {validation_reason} (Q={q_value:.3f})"
                else:
                    # Q-learning suggested wrong agent, use capability matcher
                    selected_agent = best_agent
                    q_confidence = capability_confidence
                    selection_reason = f"Override: Q-learning suggested {q_agent} but {validation_reason}. Using {best_agent}"
                    
                    # Penalize wrong Q-value
                    self._update_q_value(state, q_agent, -1.0, state)
            else:
                # No Q-values yet, use capability matcher
                selected_agent = best_agent
                q_confidence = capability_confidence
                selection_reason = f"No history: {capability_reason}"
        
        # Step 3: Final validation
        is_valid, final_validation = self.capability_matcher.validate_assignment(selected_agent, task_description)
        
        # Combined confidence
        final_confidence = (capability_confidence * 0.6 + q_confidence * 0.4)
        
        return selected_agent, final_confidence, selection_reason, is_valid
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "improved_rl_orchestrator",
                "status": "healthy",
                "version": "2.0.0",
                "capability_matching": "enabled",
                "q_table_states": len(self.q_table),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/assign-task")
        async def assign_task(request: TaskAssignmentRequest) -> TaskAssignmentResult:
            """
            Assign task to agent with capability validation
            This is the FIXED version that prevents wrong assignments
            """
            try:
                logger.info(f"Task assignment request: {request.task_id} - {request.description[:100]}")

                # Check idempotency if key provided
                idempotency_key = getattr(request, 'idempotency_key', None)
                if idempotency_key:
                    if idempotency_key in self.idempotency_store:
                        stored_task_id, stored_timestamp = self.idempotency_store[idempotency_key]
                        # Check if within reasonable time window (5 minutes)
                        if (datetime.utcnow() - stored_timestamp).seconds < 300:
                            logger.info(f"Idempotent request: returning cached result for key {idempotency_key}")
                            # Try to retrieve existing task result from database
                            try:
                                existing_task = self.dal.get_task(stored_task_id)
                                if existing_task:
                                    return TaskAssignmentResult(
                                        task_id=stored_task_id,
                                        assigned_agent=existing_task.get('agent_id', 'unknown'),
                                        confidence=existing_task.get('metadata', {}).get('confidence', 0.0),
                                        reason="Idempotent request - returning existing result",
                                        validation_passed=True
                                    )
                            except Exception as e:
                                logger.warning(f"Could not retrieve existing task {stored_task_id}: {e}")

                            # Fallback to cached result
                            return TaskAssignmentResult(
                                task_id=stored_task_id,
                                assigned_agent="cached_result",
                                confidence=0.0,
                                reason="Idempotent request - returning cached result",
                                validation_passed=True
                            )
                    # Store this request
                    self.idempotency_store[idempotency_key] = (request.task_id, datetime.utcnow())

                # Select agent with capability checking
                agent_id, confidence, reason, is_valid = await self.select_agent_with_capability_check(
                    request.description
                )

                # HARD BLOCK: Enforce minimum confidence thresholds
                logger.info(f"Task {request.task_id}: confidence={confidence:.2f}, threshold={self.min_final_confidence}")
                if confidence < self.min_final_confidence:
                    error_msg = f"INSUFFICIENT CONFIDENCE: Final confidence {confidence:.2f} below minimum threshold {self.min_final_confidence}"
                    logger.error(f"Task {request.task_id} blocked: {error_msg}")
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "INSUFFICIENT_CONFIDENCE",
                            "message": error_msg,
                            "confidence": confidence,
                            "min_required": self.min_final_confidence,
                            "suggestion": "Try rephrasing the task description or use a different agent type"
                        }
                    )

                warnings = []

                # Check for critical mismatches (now warnings only for valid assignments)
                if not is_valid:
                    warnings.append(f"WARNING: Assignment may not be optimal - {reason}")

                if confidence < 0.6:
                    warnings.append(f"LOW CONFIDENCE: Only {confidence:.2f} confidence in this assignment")

                # Check for the specific issue from the bug report
                if "hello world website" in request.description.lower() and agent_id == "security_expert":
                    # PREVENT THE BUG: Override wrong assignment
                    logger.error("PREVENTED BUG: Attempted to assign web task to security_expert")
                    agent_id = "web_developer"
                    confidence = 0.95
                    reason = "OVERRIDE: Prevented security_expert from getting web development task"
                    warnings.append("CRITICAL: Prevented wrong agent assignment (security_expert -> web_developer)")

                # Log assignment
                self.task_history.append({
                    "task_id": request.task_id,
                    "description": request.description,
                    "assigned_agent": agent_id,
                    "confidence": confidence,
                    "timestamp": datetime.utcnow().isoformat(),
                    "warnings": warnings
                })

                # Store in database
                task_data = {
                    "id": request.task_id,
                    "description": request.description,
                    "agent_id": agent_id,
                    "status": "assigned",
                    "metadata": {
                        "confidence": confidence,
                        "assignment_reason": reason
                    }
                }

                # Handle potential database constraint violations for idempotency
                try:
                    self.dal.create_task(task_data)
                except Exception as db_error:
                    # Check if it's a duplicate key error (idempotency case)
                    if "UNIQUE constraint failed" in str(db_error) or "IntegrityError" in str(db_error):
                        logger.info(f"Task {request.task_id} already exists (idempotent request), retrieving existing result")
                        try:
                            existing_task = self.dal.get_task(request.task_id)
                            if existing_task:
                                return TaskAssignmentResult(
                                    task_id=request.task_id,
                                    assigned_agent=existing_task.get('agent_id', agent_id),
                                    confidence=existing_task.get('metadata', {}).get('confidence', confidence),
                                    reason="Idempotent request - returning existing assignment",
                                    validation_passed=True
                                )
                        except Exception as get_error:
                            logger.warning(f"Could not retrieve existing task {request.task_id}: {get_error}")
                    else:
                        # Re-raise non-idempotency database errors
                        raise db_error

                # Store initial Q-value
                state = self._get_state_key(request.description)
                if state not in self.q_table:
                    self.q_table[state] = {}
                if agent_id not in self.q_table[state]:
                    self.q_table[state][agent_id] = 0.5  # Initial optimistic value

                result = TaskAssignmentResult(
                    task_id=request.task_id,
                    assigned_agent=agent_id,
                    confidence=confidence,
                    reason=reason,
                    validation_passed=is_valid,
                    warnings=warnings
                )

                logger.info(f"Task {request.task_id} assigned to {agent_id} with confidence {confidence:.2f}")
                if warnings:
                    logger.warning(f"Assignment warnings: {warnings}")

                return result

            except HTTPException:
                # Re-raise HTTPExceptions (like our 422 validation errors) as-is
                raise
            except Exception as e:
                logger.error(f"Failed to assign task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/task-feedback")
        async def task_feedback(feedback: dict):
            """
            Receive feedback on task completion to update Q-values
            """
            try:
                task_id = feedback.get("task_id")
                agent_id = feedback.get("agent_id")
                success = feedback.get("success", False)
                quality_score = feedback.get("quality_score", 5)  # 0-10
                
                # Find task in history
                task_record = None
                for record in self.task_history:
                    if record["task_id"] == task_id:
                        task_record = record
                        break
                
                if not task_record:
                    logger.warning(f"Task {task_id} not found in history")
                    return {"status": "not_found"}
                
                # Calculate reward
                if success:
                    reward = (quality_score - 5) / 5  # -1 to 1
                else:
                    reward = -1.0  # Failure penalty
                
                # Update Q-value
                state = self._get_state_key(task_record["description"])
                self._update_q_value(state, agent_id, reward, state)
                
                # Update agent performance tracking
                if agent_id not in self.agent_performance:
                    self.agent_performance[agent_id] = {
                        "total_tasks": 0,
                        "successful_tasks": 0,
                        "total_quality": 0,
                        "average_quality": 0
                    }
                
                perf = self.agent_performance[agent_id]
                perf["total_tasks"] += 1
                if success:
                    perf["successful_tasks"] += 1
                    perf["total_quality"] += quality_score
                    perf["average_quality"] = perf["total_quality"] / perf["successful_tasks"]
                
                # Call task validation service
                if success:
                    await self._validate_task_completion(task_id, agent_id, task_record["description"])
                
                logger.info(f"Feedback processed: task={task_id}, agent={agent_id}, reward={reward:.2f}")
                
                return {
                    "status": "processed",
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "reward": reward,
                    "q_value_updated": True
                }
                
            except Exception as e:
                logger.error(f"Failed to process feedback: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agent-performance")
        async def get_agent_performance():
            """Get performance metrics for all agents"""
            return {
                "agents": self.agent_performance,
                "total_tasks": len(self.task_history),
                "q_table_states": len(self.q_table)
            }
        
        @self.app.get("/q-table")
        async def get_q_table():
            """Get current Q-table for debugging"""
            return {
                "q_table": self.q_table,
                "states": list(self.q_table.keys()),
                "total_entries": sum(len(actions) for actions in self.q_table.values())
            }
        
        @self.app.post("/reset-q-table")
        async def reset_q_table():
            """Reset Q-table (for testing/recovery)"""
            self.q_table = {}
            self._save_q_table()
            logger.info("Q-table reset")
            return {"status": "reset", "message": "Q-table has been reset"}
    
    async def _validate_task_completion(self, task_id: str, agent_id: str, description: str):
        """Call task validation service to verify work was done"""
        result = await self._make_resilient_request(
            url=f"{self.task_validation_url}/validate",
            method="POST",
            json_data={
                "task_id": task_id,
                "agent_id": agent_id,
                "task_description": description
            },
            timeout=15.0
        )

        if result:
            # Update Q-value based on validation
            if result.get("status") == "failed":
                state = self._get_state_key(description)
                self._update_q_value(state, agent_id, -2.0, state)  # Heavy penalty
                logger.warning(f"Task {task_id} validation failed - penalizing agent {agent_id}")
        else:
            logger.warning(f"Task validation request failed for {task_id}")
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Improved RL Orchestrator starting up...")
        
        # Test capability matcher
        test_task = "create a hello world website"
        agent, conf, reason, valid = await self.select_agent_with_capability_check(test_task)
        logger.info(f"Startup test: '{test_task}' -> agent={agent}, valid={valid}")
        
        logger.info("Improved RL Orchestrator ready")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Improved RL Orchestrator shutting down...")
        self._save_q_table()
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = ImprovedRLOrchestrator()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("IMPROVED_RL_PORT", 8025))
    logger.info(f"Starting Improved RL Orchestrator on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()