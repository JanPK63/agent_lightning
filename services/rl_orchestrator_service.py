#!/usr/bin/env python3
"""
RL Orchestrator Service - Bridges Agent Designer with RL Training
Manages the connection between agents, training, memory, and rewards
Based on AI Agent Framework Implementation Guide
"""

import os
import sys
import json
import asyncio
import uuid
import time
from typing import Dict, List, Any, Optional
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

# Import shared data access layer
from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    """Training status for agents"""
    NEVER = "never"
    TRAINING = "training"
    TRAINED = "trained"
    UPDATING = "updating"


class KnowledgeItem(BaseModel):
    """Knowledge item for agent training"""
    content: str
    category: str
    importance: float = 1.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentTrainingRequest(BaseModel):
    """Request to train an agent"""
    agent_id: str
    knowledge_items: List[KnowledgeItem]
    training_config: Optional[Dict[str, Any]] = None


class MDPTransition(BaseModel):
    """MDP transition for RL training"""
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    done: bool = False


class RLOrchestrator:
    """Main RL Orchestrator that connects agents with training"""
    
    def __init__(self):
        self.app = FastAPI(title="RL Orchestrator Service", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("rl_orchestrator")
        self.cache = get_cache()
        
        # Service URLs
        self.agent_designer_url = "http://localhost:8002"
        self.rl_server_url = "http://localhost:8010"
        self.ai_model_url = "http://localhost:8005"
        
        # Training state tracking
        self.training_state = {}
        self.agent_knowledge = {}
        self.training_history = {}
        
        logger.info("✅ RL Orchestrator initialized")
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
        
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
                "service": "rl_orchestrator",
                "status": "healthy" if health_status['database'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "agents_tracked": len(self.training_state),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        @self.app.post("/agents/{agent_id}/train")
        async def train_agent(
            agent_id: str,
            request: AgentTrainingRequest,
            background_tasks: BackgroundTasks
        ):
            """Train an agent with knowledge items"""
            try:
                logger.info(f"Training request for agent {agent_id} with {len(request.knowledge_items)} items")
                
                # Update agent knowledge
                if agent_id not in self.agent_knowledge:
                    self.agent_knowledge[agent_id] = []
                    
                self.agent_knowledge[agent_id].extend(request.knowledge_items)
                
                # Update training state
                self.training_state[agent_id] = {
                    "status": TrainingStatus.TRAINING.value,
                    "total_knowledge": len(self.agent_knowledge[agent_id]),
                    "new_items": len(request.knowledge_items),
                    "last_trained": datetime.utcnow().isoformat(),
                    "training_started": datetime.utcnow().isoformat()
                }
                
                # Store in cache
                self.cache.set(f"agent_training:{agent_id}", self.training_state[agent_id], ttl=3600)
                
                # Start training in background
                background_tasks.add_task(
                    self._orchestrate_training,
                    agent_id,
                    request.knowledge_items,
                    request.training_config
                )
                
                return {
                    "agent_id": agent_id,
                    "status": "training_started",
                    "knowledge_items": len(request.knowledge_items)
                }
                
            except Exception as e:
                logger.error(f"Failed to start training: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/agents/{agent_id}/feedback")
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
                task_id = feedback.get("task_id", "unknown")
                
                # Convert feedback to reward signal for RL
                reward = (score - 5) / 5  # Normalize to -1 to 1
                
                # Initialize if needed
                if agent_id not in self.training_state:
                    self.training_state[agent_id] = {
                        "status": "idle",
                        "total_knowledge": 0,
                        "new_items": 0,
                        "last_trained": "Never",
                        "feedback_history": []
                    }
                
                # Add feedback history if not present
                if "feedback_history" not in self.training_state[agent_id]:
                    self.training_state[agent_id]["feedback_history"] = []
                
                self.training_state[agent_id]["feedback_history"].append({
                    "task_id": task_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "score": score,
                    "reward": reward,
                    "suggestions": suggestions,
                    "issues": issues
                })
                
                # If we have enough feedback, trigger a training session
                if len(self.training_state[agent_id]["feedback_history"]) >= 5:
                    # Generate MDP transitions from feedback
                    transitions = []
                    for fb in self.training_state[agent_id]["feedback_history"][-5:]:
                        transitions.append({
                            "state": f"task_{fb['task_id']}",
                            "action": "execute",
                            "reward": fb["reward"],
                            "next_state": "completed",
                            "done": True
                        })
                    
                    # Send to RL server for training
                    rl_response = requests.post(
                        f"{self.rl_server_url}/train",
                        json={
                            "agent_id": agent_id,
                            "algorithm": "DQN",
                            "transitions": transitions,
                            "num_episodes": 50
                        }
                    )
                    
                    if rl_response.status_code == 200:
                        self.training_state[agent_id]["last_trained"] = datetime.utcnow().isoformat()
                
                # Store in cache
                self.cache.set(f"agent_training:{agent_id}", self.training_state[agent_id], ttl=3600)
                
                logger.info(f"Received feedback for agent {agent_id}: score={score}, reward={reward}")
                
                return {
                    "status": "accepted",
                    "agent_id": agent_id,
                    "reward_signal": reward,
                    "feedback_count": len(self.training_state[agent_id]["feedback_history"]),
                    "message": f"Feedback recorded. {'Training triggered.' if len(self.training_state[agent_id]['feedback_history']) % 5 == 0 else ''}"
                }
                
            except Exception as e:
                logger.error(f"Error processing feedback: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents/{agent_id}/training-status")
        async def get_training_status(agent_id: str):
            """Get training status for an agent"""
            try:
                # Check cache first
                cached = self.cache.get(f"agent_training:{agent_id}")
                if cached:
                    return cached
                    
                # Default status
                return {
                    "status": TrainingStatus.NEVER.value,
                    "total_knowledge": 0,
                    "new_items": 0,
                    "last_trained": "Never"
                }
                
            except Exception as e:
                logger.error(f"Failed to get training status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/agents/training-summary")
        async def get_training_summary():
            """Get training summary for all agents"""
            try:
                summary = []
                
                # Get all agents from Agent Designer Service
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.agent_designer_url}/agents") as resp:
                        if resp.status == 200:
                            agents = await resp.json()
                            
                            for agent in agents.get("agents", []):
                                agent_id = agent.get("id")
                                training_info = self.training_state.get(agent_id, {
                                    "status": TrainingStatus.NEVER.value,
                                    "total_knowledge": 0,
                                    "new_items": 0,
                                    "last_trained": "Never"
                                })
                                
                                summary.append({
                                    "agent_id": agent_id,
                                    "agent_name": agent.get("name"),
                                    **training_info
                                })
                                
                return {
                    "agents": summary,
                    "total_agents": len(summary),
                    "trained_agents": sum(1 for s in summary if s.get("status") == TrainingStatus.TRAINED.value)
                }
                
            except Exception as e:
                logger.error(f"Failed to get training summary: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
    async def _orchestrate_training(self, agent_id: str, knowledge_items: List[KnowledgeItem], config: Dict):
        """Orchestrate the training process"""
        try:
            logger.info(f"Orchestrating training for agent {agent_id}")
            
            # Step 1: Create RL agent if not exists
            await self._ensure_rl_agent(agent_id)
            
            # Step 2: Generate MDP transitions from knowledge
            transitions = await self._generate_transitions(agent_id, knowledge_items)
            
            # Step 3: Store experiences in RL server
            for transition in transitions:
                await self._store_experience(agent_id, transition)
                
            # Step 4: Trigger training
            training_job = await self._trigger_training(agent_id, config)
            
            # Step 5: Store memories
            await self._store_memories(agent_id, knowledge_items)
            
            # Step 6: Wait for training completion
            await self._monitor_training(agent_id, training_job.get("job_id"))
            
            # Update status
            self.training_state[agent_id]["status"] = TrainingStatus.TRAINED.value
            self.training_state[agent_id]["training_completed"] = datetime.utcnow().isoformat()
            
            # Update cache
            self.cache.set(f"agent_training:{agent_id}", self.training_state[agent_id], ttl=3600)
            
            # Emit completion event
            self.dal.event_bus.emit(EventChannel.SYSTEM_EVENT, {
                "type": "agent_training_completed",
                "agent_id": agent_id,
                "knowledge_items": len(knowledge_items)
            })
            
            logger.info(f"Training completed for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Training orchestration failed: {e}")
            self.training_state[agent_id]["status"] = TrainingStatus.NEVER.value
            self.training_state[agent_id]["error"] = str(e)
            
    async def _ensure_rl_agent(self, agent_id: str):
        """Ensure RL agent exists in RL server"""
        try:
            async with aiohttp.ClientSession() as session:
                # First, get agent details from Agent Designer Service
                async with session.get(f"{self.agent_designer_url}/agents/{agent_id}") as resp:
                    if resp.status != 200:
                        logger.error(f"Agent {agent_id} not found in Agent Designer Service")
                        raise Exception(f"Agent {agent_id} not found")
                    
                    agent_data = await resp.json()
                    
                # Try to create agent in RL server (will succeed if new, fail if exists)
                async with session.post(
                    f"{self.rl_server_url}/agents/create",
                    json={
                        "agent_id": agent_id,
                        "state_space": 100,  # Default state space size
                        "action_space": 50,  # Default action space size
                        "algorithm": "DQN"
                    }
                ) as resp:
                    if resp.status in [200, 400]:  # 400 if already exists
                        logger.info(f"RL agent {agent_id} ready in RL Server")
                    else:
                        # If create fails, just log and continue - agent might already exist
                        logger.warning(f"Could not create RL agent (may already exist): {resp.status}")
                        
        except Exception as e:
            logger.error(f"Failed to ensure RL agent: {e}")
            # Don't raise - continue with training even if RL agent setup has issues
            
    async def _generate_transitions(self, agent_id: str, knowledge_items: List[KnowledgeItem]) -> List[MDPTransition]:
        """Generate MDP transitions from knowledge items"""
        transitions = []
        
        for i, item in enumerate(knowledge_items):
            # Create state from knowledge
            state = {
                "knowledge": item.content,
                "category": item.category,
                "importance": item.importance,
                "index": i
            }
            
            # Simulate action (in production, this would be actual agent output)
            action = {
                "type": "process_knowledge",
                "content": item.content,
                "processing_time": np.random.uniform(0.1, 1.0)
            }
            
            # Calculate reward based on importance
            reward = item.importance * np.random.uniform(0.5, 1.5)
            
            # Next state
            next_state = {
                "knowledge": item.content,
                "category": item.category,
                "importance": item.importance,
                "index": i + 1,
                "processed": True
            }
            
            # Create transition
            transition = MDPTransition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=(i == len(knowledge_items) - 1)
            )
            
            transitions.append(transition)
            
        return transitions
        
    async def _store_experience(self, agent_id: str, transition: MDPTransition):
        """Store experience in RL server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rl_server_url}/agents/{agent_id}/experience",
                    json={
                        "state": list(transition.state.values()) if isinstance(transition.state, dict) else transition.state,
                        "action": transition.action.get("type", 0) if isinstance(transition.action, dict) else transition.action,
                        "reward": transition.reward,
                        "next_state": list(transition.next_state.values()) if isinstance(transition.next_state, dict) else transition.next_state,
                        "done": transition.done
                    }
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Failed to store experience: {resp.status}")
                        
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            
    async def _trigger_training(self, agent_id: str, config: Dict) -> Dict:
        """Trigger training in RL server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rl_server_url}/training/start",
                    json={
                        "agent_id": agent_id,
                        "algorithm": config.get("algorithm", "DQN"),
                        "num_episodes": config.get("num_episodes", 100),
                        "batch_size": config.get("batch_size", 32),
                        "learning_rate": config.get("learning_rate", 0.001)
                    }
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        raise Exception(f"Failed to trigger training: {resp.status}")
                        
        except Exception as e:
            logger.error(f"Failed to trigger training: {e}")
            return {"job_id": None}
            
    async def _store_memories(self, agent_id: str, knowledge_items: List[KnowledgeItem]):
        """Store memories in RL server"""
        try:
            async with aiohttp.ClientSession() as session:
                for item in knowledge_items:
                    async with session.post(
                        f"{self.rl_server_url}/memory/{agent_id}/store",
                        json={
                            "memory_type": "semantic",
                            "content": {
                                "knowledge": item.content,
                                "category": item.category,
                                "importance": item.importance
                            },
                            "timestamp": item.timestamp,
                            "importance": item.importance
                        }
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(f"Failed to store memory: {resp.status}")
                            
        except Exception as e:
            logger.error(f"Failed to store memories: {e}")
            
    async def _monitor_training(self, agent_id: str, job_id: Optional[str]):
        """Monitor training progress"""
        if not job_id:
            return
            
        try:
            async with aiohttp.ClientSession() as session:
                max_attempts = 30
                for _ in range(max_attempts):
                    async with session.get(
                        f"{self.rl_server_url}/training/{job_id}/status"
                    ) as resp:
                        if resp.status == 200:
                            status = await resp.json()
                            if status.get("status") in ["completed", "failed"]:
                                break
                                
                    await asyncio.sleep(2)
                    
        except Exception as e:
            logger.error(f"Failed to monitor training: {e}")
            
    def _setup_event_handlers(self):
        """Setup event handlers"""
        
        def on_task_completed(event):
            """Handle task completion for reward signals"""
            task_id = event.data.get('task_id')
            agent_id = event.data.get('agent_id')
            
            if agent_id:
                # Update agent's last activity
                if agent_id in self.training_state:
                    self.training_state[agent_id]["last_activity"] = datetime.utcnow().isoformat()
                    
        # Register handlers
        self.dal.event_bus.on(EventChannel.TASK_COMPLETED, on_task_completed)
        
        logger.info("Event handlers registered for RL orchestrator")
        
    async def startup(self):
        """Startup tasks"""
        logger.info("RL Orchestrator starting up...")
        
        # Load existing training states from cache
        for key in self.cache.redis_client.keys("agent_training:*"):
            agent_id = key.decode().split(":")[-1]
            state = self.cache.get(key)
            if state:
                self.training_state[agent_id] = state
                
        logger.info(f"RL Orchestrator ready with {len(self.training_state)} agents tracked")
        
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("RL Orchestrator shutting down...")
        
        # Save all training states to cache
        for agent_id, state in self.training_state.items():
            self.cache.set(f"agent_training:{agent_id}", state, ttl=3600)
            
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = RLOrchestrator()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("RL_ORCHESTRATOR_PORT", 8011))
    logger.info(f"Starting RL Orchestrator Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()