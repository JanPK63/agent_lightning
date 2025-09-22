#!/usr/bin/env python3
"""
Agent Service Template - Base template for implementing agent services

This template provides a FastAPI service that wraps an agentlightning.LitAgent
and provides the standard endpoints expected by the RL Orchestrator.

Usage:
    Subclass this template and implement the specific agent logic in the
    agent class that inherits from LitAgent.
"""

import os
import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import agentlightning components
from agentlightning.litagent import LitAgent
from agentlightning.types import NamedResources, Task, TaskInput, Rollout
from agentlightning.llm_providers import llm_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgentService:
    """Base class for agent services"""

    def __init__(self, agent_class: type[LitAgent], agent_config: Dict[str, Any]):
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.agent: Optional[LitAgent] = None

        # Service configuration
        self.agent_id = agent_config.get('agent_id')
        self.name = agent_config.get('name', self.agent_id)
        self.port = agent_config.get('port', 9000)
        self.capabilities = agent_config.get('capabilities', [])

        # Initialize FastAPI app
        self.app = FastAPI(
            title=f"{self.name} Agent Service",
            version="1.0.0",
            description=f"Agent service for {self.name}"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self._setup_routes()

        # Initialize agent
        self._initialize_agent()

        logger.info(f"✅ {self.name} Agent Service initialized on port {self.port}")

    def _initialize_agent(self):
        """Initialize the LitAgent instance"""
        try:
            self.agent = self.agent_class(agent_id=self.agent_id)
            logger.info(f"✅ Agent {self.agent_id} initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent {self.agent_id}: {e}")
            raise

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            """Service information"""
            return {
                "service": f"{self.name} Agent",
                "agent_id": self.agent_id,
                "version": "1.0.0",
                "capabilities": self.capabilities,
                "status": "healthy" if self.agent else "unhealthy"
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy" if self.agent else "unhealthy",
                "agent_id": self.agent_id,
                "name": self.name,
                "capabilities": self.capabilities,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - time.time()  # TODO: track actual start time
            }

        @self.app.post("/execute")
        async def execute_task(request: Request):
            """Execute a task using the agent"""
            try:
                payload = await request.json()
                task_id = payload.get("task_id")
                task_description = payload.get("task_description", "")
                context = payload.get("context", {})

                if not task_id:
                    raise HTTPException(status_code=400, detail="task_id is required")

                if not self.agent:
                    raise HTTPException(status_code=503, detail="Agent not initialized")

                logger.info(f"Executing task {task_id} with agent {self.agent_id}")

                # Execute the task
                result = await self._execute_task(task_id, task_description, context)

                return {
                    "status": "completed",
                    "task_id": task_id,
                    "agent_id": self.agent_id,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/capabilities")
        async def get_capabilities():
            """Get agent capabilities"""
            return {
                "agent_id": self.agent_id,
                "capabilities": self.capabilities,
                "specialization": self.agent_config.get('specialization')
            }

    async def _execute_task(self, task_id: str, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the agent - override in subclasses for specific logic"""
        try:
            # Create a Task object
            task_input = TaskInput(task_description)
            task = Task(
                rollout_id=task_id,
                input=task_input,
                metadata=context
            )

            # Get resources (LLM, prompts, etc.)
            resources = self._get_resources()

            # Execute rollout
            if hasattr(self.agent, 'training_rollout_async'):
                result = await self.agent.training_rollout_async(task.input, task.rollout_id, resources)
            else:
                # Fallback to sync method in thread
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.agent.training_rollout, task.input, task.rollout_id, resources
                )

            # Process result
            if isinstance(result, Rollout):
                return {
                    "final_reward": result.final_reward,
                    "triplets": len(result.triplets) if result.triplets else 0,
                    "status": "completed"
                }
            elif isinstance(result, float):
                return {
                    "final_reward": result,
                    "status": "completed"
                }
            elif isinstance(result, list):
                return {
                    "result": result,
                    "status": "completed"
                }
            else:
                return {
                    "result": str(result),
                    "status": "completed"
                }

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    def _get_resources(self) -> NamedResources:
        """Get resources for the agent - override in subclasses"""
        # Default resources - can be extended
        return {}

    def run(self, host: str = "0.0.0.0"):
        """Run the service"""
        logger.info(f"Starting {self.name} Agent Service on {host}:{self.port}")
        uvicorn.run(self.app, host=host, port=self.port)


class TemplateAgent(LitAgent):
    """Template agent class - override training_rollout_async for specific behavior"""

    def __init__(self, agent_id: str, specialization: str = "general"):
        super().__init__(agent_id=agent_id)
        self.specialization = specialization

    async def training_rollout_async(self, task: TaskInput, rollout_id: str, resources: NamedResources) -> Rollout:
        """Implement the agent's task execution logic here"""
        # This is a template - override in specific agent implementations

        # Example: Simple text processing
        if isinstance(task, str):
            # Process the task and return a result
            result_text = f"Processed task: {task[:100]}..."

            # Create a simple rollout
            rollout = Rollout(
                rollout_id=rollout_id,
                final_reward=0.8,  # Example reward
                triplets=[
                    {
                        "prompt": task,
                        "response": result_text,
                        "reward": 0.8
                    }
                ]
            )
            return rollout

        # Default fallback
        rollout = Rollout(
            rollout_id=rollout_id,
            final_reward=0.5,
            triplets=[]
        )
        return rollout


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Service Template")
    parser.add_argument("--agent-id", type=str, required=True, help="Agent ID")
    parser.add_argument("--name", type=str, help="Agent display name")
    parser.add_argument("--port", type=int, default=9000, help="Port to run on")
    parser.add_argument("--specialization", type=str, default="general", help="Agent specialization")

    args = parser.parse_args()

    # Configuration for the agent
    agent_config = {
        'agent_id': args.agent_id,
        'name': args.name or args.agent_id,
        'port': args.port,
        'specialization': args.specialization,
        'capabilities': [args.specialization, 'task_execution']
    }

    # Create and run the service
    service = BaseAgentService(TemplateAgent, agent_config)
    service.run()