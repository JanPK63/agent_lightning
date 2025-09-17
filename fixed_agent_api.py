#!/usr/bin/env python3
"""
Fixed Agent API - Properly connects task requests to agent execution
This fixes the main issue where agents just describe instead of executing
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

from agent_executor_fix import TaskExecutionBridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskRequest(BaseModel):
    """Task request model"""
    task: str
    agent_id: Optional[str] = "auto"
    model: Optional[str] = "gpt-4o"
    context: Dict[str, Any] = {}
    timeout: int = 60

class FixedAgentAPI:
    """Fixed API that actually makes agents work"""
    
    def __init__(self):
        self.app = FastAPI(title="Fixed Agent Lightning API", version="1.0.0")
        self.execution_bridge = TaskExecutionBridge()
        
        # Available agents (from the actual system)
        self.available_agents = {
            'full_stack_developer': {
                'name': 'Full Stack Developer',
                'specialization': 'Complete web application development',
                'model': 'claude-3-sonnet',
                'capabilities': ['frontend', 'backend', 'api-design', 'full-stack']
            },
            'data_scientist': {
                'name': 'Data Scientist',
                'specialization': 'Data analysis and machine learning',
                'model': 'claude-3-sonnet',
                'capabilities': ['data-analysis', 'machine-learning', 'visualization', 'statistics']
            },
            'security_expert': {
                'name': 'Security Expert',
                'specialization': 'Cybersecurity and secure coding',
                'model': 'claude-3-opus',
                'capabilities': ['security-analysis', 'vulnerability-assessment', 'compliance']
            },
            'devops_engineer': {
                'name': 'DevOps Engineer',
                'specialization': 'Infrastructure and deployment automation',
                'model': 'claude-3-haiku',
                'capabilities': ['infrastructure', 'deployment', 'monitoring', 'automation']
            },
            'system_architect': {
                'name': 'System Architect',
                'specialization': 'Software architecture and system design',
                'model': 'gpt-4o',
                'capabilities': ['architecture-design', 'system-design', 'scalability', 'integration']
            }
        }
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "Fixed Agent Lightning API",
                "status": "operational",
                "message": "Agents are now properly executing tasks!",
                "available_endpoints": [
                    "/agents - List available agents",
                    "/execute - Execute task with agent",
                    "/health - Health check"
                ]
            }
        
        @self.app.get("/health")
        async def health():
            """Health check"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "agents_available": len(self.available_agents),
                "execution_bridge": "operational"
            }
        
        @self.app.get("/agents")
        async def list_agents():
            """List available agents"""
            return {
                "agents": [
                    {
                        "id": agent_id,
                        **config,
                        "status": "available"
                    }
                    for agent_id, config in self.available_agents.items()
                ],
                "count": len(self.available_agents)
            }
        
        @self.app.post("/execute")
        async def execute_task(request: TaskRequest):
            """Execute a task with an agent - THIS ACTUALLY WORKS!"""
            
            try:
                # Generate task ID
                import uuid
                task_id = str(uuid.uuid4())[:8]
                
                # Auto-select agent if needed
                agent_id = request.agent_id
                if agent_id == "auto" or not agent_id:
                    agent_id = self._auto_select_agent(request.task)
                
                # Validate agent
                if agent_id not in self.available_agents:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Agent '{agent_id}' not found. Available: {list(self.available_agents.keys())}"
                    )
                
                logger.info(f"Executing task {task_id} with agent: {agent_id}")
                logger.info(f"Task: {request.task[:100]}...")
                
                # Execute the task
                start_time = time.time()
                result = await self.execution_bridge.execute_agent_task(
                    agent_id=agent_id,
                    task_description=request.task,
                    context=request.context,
                    model=request.model
                )
                execution_time = time.time() - start_time
                
                # Return with task ID and result
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result.get("result", ""),
                    "metadata": {
                        "agent_id": agent_id,
                        "agent_name": self.available_agents[agent_id]['name'],
                        "task_description": request.task,
                        "execution_time": round(execution_time, 2),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/chat/{agent_id}")
        async def chat_with_agent(agent_id: str, message: Dict[str, str]):
            """Chat with a specific agent"""
            
            if agent_id not in self.available_agents:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
            
            user_message = message.get("message", "")
            if not user_message:
                raise HTTPException(status_code=400, detail="Message is required")
            
            # Execute as a chat task
            result = await self.execution_bridge.execute_agent_task(
                agent_id=agent_id,
                task_description=f"Respond to this message: {user_message}",
                context={"type": "chat", "conversational": True}
            )
            
            return {
                "agent_id": agent_id,
                "agent_name": self.available_agents[agent_id]['name'],
                "user_message": user_message,
                "agent_response": result.get("result", ""),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/agents/{agent_id}")
        async def get_agent_info(agent_id: str):
            """Get information about a specific agent"""
            
            if agent_id not in self.available_agents:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
            
            return {
                "id": agent_id,
                **self.available_agents[agent_id],
                "status": "available"
            }
    
    def _auto_select_agent(self, task: str) -> str:
        """Auto-select the best agent for a task based on keywords"""
        
        task_lower = task.lower()
        
        # Keyword-based agent selection
        if any(word in task_lower for word in ['web', 'api', 'frontend', 'backend', 'react', 'node', 'database']):
            return 'full_stack_developer'
        elif any(word in task_lower for word in ['data', 'analysis', 'machine learning', 'ml', 'statistics', 'pandas']):
            return 'data_scientist'
        elif any(word in task_lower for word in ['security', 'vulnerability', 'hack', 'secure', 'auth']):
            return 'security_expert'
        elif any(word in task_lower for word in ['deploy', 'docker', 'kubernetes', 'ci/cd', 'infrastructure']):
            return 'devops_engineer'
        elif any(word in task_lower for word in ['architecture', 'design', 'system', 'scalable', 'microservice']):
            return 'system_architect'
        else:
            # Default to full-stack developer for general tasks
            return 'full_stack_developer'


def main():
    """Run the fixed API server"""
    import uvicorn
    
    api = FixedAgentAPI()
    
    print("\\n" + "="*60)
    print("⚡ FIXED AGENT LIGHTNING API")
    print("="*60)
    print("\\n🎯 Problem SOLVED: Agents now actually execute tasks!")
    print("\\n📍 Endpoints:")
    print("  • http://localhost:8888/ - API info")
    print("  • http://localhost:8888/agents - List agents")
    print("  • http://localhost:8888/execute - Execute tasks")
    print("  • http://localhost:8888/chat/{agent_id} - Chat with agent")
    print("\\n🤖 Available Agents:")
    for agent_id, config in api.available_agents.items():
        print(f"  • {agent_id}: {config['name']}")
    print("\\n✨ Features:")
    print("  • Actual task execution (not just descriptions)")
    print("  • Auto agent selection")
    print("  • Real AI integration")
    print("  • Proper error handling")
    print("\\n" + "="*60 + "\\n")
    
    uvicorn.run(api.app, host="0.0.0.0", port=8888)

if __name__ == "__main__":
    main()