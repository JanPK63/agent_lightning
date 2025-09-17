#!/usr/bin/env python3
"""
Enterprise Reinforcement Learning Server - Integrated with Agent Lightning
Provides advanced RL capabilities for AI agents including training, inference, and continuous learning
Based on SA-007: RL & Learning Integration Architecture
"""

import os
import sys
import json
import asyncio
import uuid
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
from collections import deque, defaultdict
import pickle

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import ray
from ray import tune
try:
    from ray.rllib.algorithms import ppo, dqn, a3c
except ImportError:
    # Fallback for older Ray versions
    try:
        from ray.rllib.agents import ppo, dqn, a3c
    except ImportError:
        ppo = dqn = a3c = None
        logger.warning("Ray RLlib not fully installed - distributed training disabled")
import wandb

# Import shared data access layer
from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache


class RLAlgorithm(str, Enum):
    """Supported RL algorithms"""
    PPO = "ppo"
    A2C = "a2c"
    SAC = "sac"
    TD3 = "td3"
    DQN = "dqn"
    RAINBOW = "rainbow"
    IMPALA = "impala"


class MemoryType(str, Enum):
    """Types of agent memory"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class TrainingStatus(str, Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


# Pydantic models
class AgentTrainingConfig(BaseModel):
    """Configuration for agent training"""
    agent_id: str = Field(description="Agent ID")
    algorithm: RLAlgorithm = Field(description="RL algorithm to use")
    environment: str = Field(description="Training environment")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm hyperparameters")
    num_episodes: int = Field(default=1000, description="Number of training episodes")
    batch_size: int = Field(default=64, description="Batch size for training")
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    gamma: float = Field(default=0.99, description="Discount factor")
    epsilon: float = Field(default=1.0, description="Exploration rate")
    epsilon_decay: float = Field(default=0.995, description="Epsilon decay rate")
    epsilon_min: float = Field(default=0.01, description="Minimum epsilon")
    use_curriculum: bool = Field(default=False, description="Use curriculum learning")
    distributed: bool = Field(default=False, description="Use distributed training")


class ExperienceBuffer(BaseModel):
    """Experience replay buffer entry"""
    state: List[float] = Field(description="Current state")
    action: int = Field(description="Action taken")
    reward: float = Field(description="Reward received")
    next_state: List[float] = Field(description="Next state")
    done: bool = Field(description="Episode done flag")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info")


class MemoryEntry(BaseModel):
    """Agent memory entry"""
    memory_type: MemoryType = Field(description="Type of memory")
    content: Dict[str, Any] = Field(description="Memory content")
    timestamp: str = Field(description="Creation timestamp")
    importance: float = Field(default=1.0, description="Memory importance score")
    access_count: int = Field(default=0, description="Number of times accessed")
    decay_rate: float = Field(default=0.01, description="Memory decay rate")


class RewardSignal(BaseModel):
    """Reward signal for agent learning"""
    agent_id: str = Field(description="Agent ID")
    task_id: str = Field(description="Task ID")
    reward: float = Field(description="Reward value")
    shaped_reward: float = Field(default=0.0, description="Shaped reward")
    components: Dict[str, float] = Field(default_factory=dict, description="Reward components")
    timestamp: str = Field(description="Timestamp")


@dataclass
class MDPAgent:
    """Markov Decision Process Agent"""
    agent_id: str
    state_space: int
    action_space: int
    algorithm: RLAlgorithm
    model: Any = None
    optimizer: Any = None
    memory: deque = field(default_factory=lambda: deque(maxlen=10000))
    epsilon: float = 1.0
    training_step: int = 0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action using epsilon-greedy strategy"""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space)
        
        if self.model is None:
            return np.random.randint(0, self.action_space)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, experience: ExperienceBuffer):
        """Store experience in replay buffer"""
        self.memory.append(experience)
    
    def update_epsilon(self, decay_rate: float, min_epsilon: float):
        """Update exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


class NeuralNetwork(nn.Module):
    """Deep Q-Network for agent learning"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class MemoryManager:
    """Manages different types of agent memory"""
    
    def __init__(self, agent_id: str, max_memories: int = 10000):
        self.agent_id = agent_id
        self.max_memories = max_memories
        self.memories = {
            MemoryType.EPISODIC: deque(maxlen=max_memories),
            MemoryType.SEMANTIC: deque(maxlen=max_memories),
            MemoryType.PROCEDURAL: deque(maxlen=max_memories),
            MemoryType.WORKING: deque(maxlen=100)  # Smaller working memory
        }
        self.memory_index = defaultdict(list)
        
    def store_memory(self, memory: MemoryEntry):
        """Store a memory entry"""
        self.memories[memory.memory_type].append(memory)
        
        # Index by content keywords
        if "keywords" in memory.content:
            for keyword in memory.content["keywords"]:
                self.memory_index[keyword].append(memory)
                
    def retrieve_memories(self, query: str, memory_type: Optional[MemoryType] = None, 
                         limit: int = 10) -> List[MemoryEntry]:
        """Retrieve relevant memories"""
        memories = []
        
        if memory_type:
            memories = list(self.memories[memory_type])
        else:
            for mem_type in self.memories:
                memories.extend(list(self.memories[mem_type]))
                
        # Sort by importance and recency
        memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        # Update access count
        for memory in memories[:limit]:
            memory.access_count += 1
            
        return memories[:limit]
    
    def consolidate_memories(self):
        """Consolidate and compress memories"""
        # Implement memory consolidation logic
        # This would merge similar memories and remove low-importance ones
        pass
    
    def decay_memories(self):
        """Apply decay to memory importance"""
        for memory_type in self.memories:
            for memory in self.memories[memory_type]:
                memory.importance *= (1 - memory.decay_rate)
                
                # Remove memories below threshold
                if memory.importance < 0.1:
                    self.memories[memory_type].remove(memory)


class RewardShaper:
    """Shapes rewards for better learning"""
    
    def __init__(self):
        self.reward_history = deque(maxlen=1000)
        self.baseline = 0.0
        
    def shape_reward(self, raw_reward: float, state: Dict[str, Any], 
                    action: Any, next_state: Dict[str, Any]) -> float:
        """Apply reward shaping techniques"""
        shaped_reward = raw_reward
        
        # Potential-based shaping
        if "progress" in state and "progress" in next_state:
            progress_diff = next_state["progress"] - state["progress"]
            shaped_reward += 0.1 * progress_diff
            
        # Curiosity bonus
        if "novelty" in state:
            shaped_reward += 0.05 * state["novelty"]
            
        # Normalize rewards
        self.reward_history.append(shaped_reward)
        if len(self.reward_history) > 100:
            mean_reward = np.mean(self.reward_history)
            std_reward = np.std(self.reward_history) + 1e-8
            shaped_reward = (shaped_reward - mean_reward) / std_reward
            
        return shaped_reward


class CurriculumLearning:
    """Manages curriculum learning for progressive training"""
    
    def __init__(self):
        self.difficulty_level = 0
        self.success_threshold = 0.7
        self.failure_threshold = 0.3
        self.window_size = 100
        self.performance_history = deque(maxlen=self.window_size)
        
    def update_difficulty(self, success: bool):
        """Update difficulty based on performance"""
        self.performance_history.append(1.0 if success else 0.0)
        
        if len(self.performance_history) >= self.window_size:
            success_rate = np.mean(self.performance_history)
            
            if success_rate > self.success_threshold:
                self.difficulty_level = min(10, self.difficulty_level + 1)
                self.performance_history.clear()
                logger.info(f"Increasing difficulty to level {self.difficulty_level}")
                
            elif success_rate < self.failure_threshold:
                self.difficulty_level = max(0, self.difficulty_level - 1)
                self.performance_history.clear()
                logger.info(f"Decreasing difficulty to level {self.difficulty_level}")
                
    def get_task_params(self) -> Dict[str, Any]:
        """Get task parameters for current difficulty"""
        return {
            "difficulty": self.difficulty_level,
            "complexity": 1 + self.difficulty_level * 0.2,
            "noise_level": max(0, 0.3 - self.difficulty_level * 0.03),
            "time_limit": max(100, 1000 - self.difficulty_level * 50)
        }


class RLServer:
    """Main RL Server class for agent training and inference"""
    
    def __init__(self):
        self.app = FastAPI(title="Enterprise RL Server", version="2.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("rl_server")
        self.cache = get_cache()
        
        # Agent management
        self.agents: Dict[str, MDPAgent] = {}
        self.training_jobs: Dict[str, Dict] = {}
        self.memory_managers: Dict[str, MemoryManager] = {}
        
        # Training components
        self.reward_shaper = RewardShaper()
        self.curriculum = CurriculumLearning()
        
        # Initialize Ray for distributed training with resource limits
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True, 
                num_cpus=4,
                object_store_memory=1_000_000_000,  # 1GB object store
                _memory=2_000_000_000,  # 2GB total memory
                _temp_dir="/tmp/ray",  # Use /tmp for Ray files
                dashboard_host="127.0.0.1",  # Bind dashboard to localhost
                include_dashboard=True,
                configure_logging=False,  # Disable Ray's verbose logging
                log_to_driver=False
            )
            
        # Initialize Weights & Biases for experiment tracking
        self.use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"
        if self.use_wandb:
            wandb.init(project="agent-lightning", entity="rl-team")
            
        logger.info("✅ RL Server initialized")
        
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
                "service": "rl_server",
                "status": "healthy" if health_status['database'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "active_agents": len(self.agents),
                "training_jobs": len(self.training_jobs),
                "ray_initialized": ray.is_initialized(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        @self.app.post("/agents/create")
        async def create_agent(request: Request):
            """Create a new RL agent"""
            try:
                # Parse JSON body
                body = await request.json()
                agent_id = body.get("agent_id")
                state_space = body.get("state_space", 100)
                action_space = body.get("action_space", 50)
                algorithm = RLAlgorithm(body.get("algorithm", "DQN"))
                
                if not agent_id:
                    raise HTTPException(status_code=400, detail="agent_id is required")
                
                if agent_id in self.agents:
                    # Agent already exists, just return success
                    return {"agent_id": agent_id, "status": "already_exists"}
                    
                # Create MDP agent
                agent = MDPAgent(
                    agent_id=agent_id,
                    state_space=state_space,
                    action_space=action_space,
                    algorithm=algorithm
                )
                
                # Initialize neural network based on algorithm
                if algorithm in [RLAlgorithm.DQN, RLAlgorithm.RAINBOW]:
                    agent.model = NeuralNetwork(state_space, 128, action_space)
                    agent.optimizer = optim.Adam(agent.model.parameters(), lr=3e-4)
                elif algorithm in [RLAlgorithm.PPO, RLAlgorithm.A2C]:
                    agent.model = PolicyNetwork(state_space, 128, action_space)
                    agent.optimizer = optim.Adam(agent.model.parameters(), lr=3e-4)
                    
                # Initialize memory manager
                self.memory_managers[agent_id] = MemoryManager(agent_id)
                
                # Store agent
                self.agents[agent_id] = agent
                
                # Persist to database
                agent_data = {
                    "agent_id": agent_id,
                    "state_space": state_space,
                    "action_space": action_space,
                    "algorithm": algorithm.value,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                self.cache.set(f"rl_agent:{agent_id}", agent_data, ttl=None)
                
                logger.info(f"Created RL agent {agent_id} with {algorithm.value}")
                return {"agent_id": agent_id, "status": "created"}
                
            except Exception as e:
                logger.error(f"Failed to create agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/agents/{agent_id}/action")
        async def get_action(
            agent_id: str,
            state: List[float],
            explore: bool = True
        ):
            """Get action from agent"""
            try:
                if agent_id not in self.agents:
                    raise HTTPException(status_code=404, detail="Agent not found")
                    
                agent = self.agents[agent_id]
                state_array = np.array(state)
                action = agent.select_action(state_array, explore=explore)
                
                return {
                    "action": action,
                    "epsilon": agent.epsilon,
                    "training_step": agent.training_step
                }
                
            except Exception as e:
                logger.error(f"Failed to get action: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/agents/{agent_id}/experience")
        async def store_experience(
            agent_id: str,
            experience: ExperienceBuffer
        ):
            """Store experience in agent's replay buffer"""
            try:
                if agent_id not in self.agents:
                    raise HTTPException(status_code=404, detail="Agent not found")
                    
                agent = self.agents[agent_id]
                agent.store_experience(experience)
                
                # Store as episodic memory
                if agent_id in self.memory_managers:
                    memory = MemoryEntry(
                        memory_type=MemoryType.EPISODIC,
                        content={
                            "state": experience.state,
                            "action": experience.action,
                            "reward": experience.reward,
                            "next_state": experience.next_state,
                            "done": experience.done
                        },
                        timestamp=datetime.utcnow().isoformat()
                    )
                    self.memory_managers[agent_id].store_memory(memory)
                    
                return {"status": "stored", "buffer_size": len(agent.memory)}
                
            except Exception as e:
                logger.error(f"Failed to store experience: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/training/start")
        async def start_training(
            config: AgentTrainingConfig,
            background_tasks: BackgroundTasks
        ):
            """Start training job for agent"""
            try:
                if config.agent_id not in self.agents:
                    raise HTTPException(status_code=404, detail="Agent not found")
                    
                job_id = str(uuid.uuid4())
                
                # Create training job
                self.training_jobs[job_id] = {
                    "job_id": job_id,
                    "agent_id": config.agent_id,
                    "config": config.dict(),
                    "status": TrainingStatus.PENDING.value,
                    "started_at": datetime.utcnow().isoformat(),
                    "metrics": {
                        "episodes": 0,
                        "total_reward": 0,
                        "avg_reward": 0,
                        "best_reward": float('-inf')
                    }
                }
                
                # Start training in background
                if config.distributed:
                    background_tasks.add_task(
                        self._distributed_training, job_id, config
                    )
                else:
                    background_tasks.add_task(
                        self._train_agent, job_id, config
                    )
                    
                logger.info(f"Started training job {job_id} for agent {config.agent_id}")
                return {"job_id": job_id, "status": "started"}
                
            except Exception as e:
                logger.error(f"Failed to start training: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/training/{job_id}/status")
        async def get_training_status(job_id: str):
            """Get training job status"""
            try:
                if job_id not in self.training_jobs:
                    raise HTTPException(status_code=404, detail="Training job not found")
                    
                return self.training_jobs[job_id]
                
            except Exception as e:
                logger.error(f"Failed to get training status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/rewards/signal")
        async def process_reward_signal(signal: RewardSignal):
            """Process reward signal for learning"""
            try:
                # Apply reward shaping
                shaped_reward = self.reward_shaper.shape_reward(
                    signal.reward,
                    {"task_id": signal.task_id},
                    None,
                    {"task_id": signal.task_id}
                )
                
                signal.shaped_reward = shaped_reward
                
                # Store reward signal
                self.cache.set(
                    f"reward:{signal.agent_id}:{signal.task_id}",
                    signal.dict(),
                    ttl=3600
                )
                
                # Emit event for training system
                self.dal.event_bus.emit(EventChannel.SYSTEM_EVENT, {
                    "type": "reward_signal",
                    "agent_id": signal.agent_id,
                    "reward": signal.shaped_reward
                })
                
                return {"shaped_reward": shaped_reward, "status": "processed"}
                
            except Exception as e:
                logger.error(f"Failed to process reward signal: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/memory/{agent_id}/store")
        async def store_memory(agent_id: str, memory: MemoryEntry):
            """Store memory for agent"""
            try:
                if agent_id not in self.memory_managers:
                    self.memory_managers[agent_id] = MemoryManager(agent_id)
                    
                self.memory_managers[agent_id].store_memory(memory)
                
                return {"status": "stored", "memory_type": memory.memory_type.value}
                
            except Exception as e:
                logger.error(f"Failed to store memory: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/memory/{agent_id}/retrieve")
        async def retrieve_memories(
            agent_id: str,
            query: str,
            memory_type: Optional[MemoryType] = None,
            limit: int = 10
        ):
            """Retrieve memories for agent"""
            try:
                if agent_id not in self.memory_managers:
                    return {"memories": [], "count": 0}
                    
                memories = self.memory_managers[agent_id].retrieve_memories(
                    query, memory_type, limit
                )
                
                return {
                    "memories": [m.dict() for m in memories],
                    "count": len(memories)
                }
                
            except Exception as e:
                logger.error(f"Failed to retrieve memories: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/curriculum/update")
        async def update_curriculum(success: bool):
            """Update curriculum learning difficulty"""
            try:
                self.curriculum.update_difficulty(success)
                params = self.curriculum.get_task_params()
                
                return {
                    "difficulty_level": self.curriculum.difficulty_level,
                    "task_params": params
                }
                
            except Exception as e:
                logger.error(f"Failed to update curriculum: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.websocket("/ws/training/{job_id}")
        async def training_websocket(websocket: WebSocket, job_id: str):
            """WebSocket for real-time training updates"""
            await websocket.accept()
            
            try:
                while job_id in self.training_jobs:
                    job = self.training_jobs[job_id]
                    await websocket.send_json({
                        "type": "training_update",
                        "status": job["status"],
                        "metrics": job["metrics"]
                    })
                    await asyncio.sleep(1)
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for job {job_id}")
                
    async def _train_agent(self, job_id: str, config: AgentTrainingConfig):
        """Train agent using specified algorithm"""
        try:
            self.training_jobs[job_id]["status"] = TrainingStatus.RUNNING.value
            agent = self.agents[config.agent_id]
            
            for episode in range(config.num_episodes):
                # Sample batch from replay buffer
                if len(agent.memory) < config.batch_size:
                    continue
                    
                batch = np.random.choice(agent.memory, config.batch_size)
                
                # Prepare batch data
                states = torch.FloatTensor([e.state for e in batch])
                actions = torch.LongTensor([e.action for e in batch])
                rewards = torch.FloatTensor([e.reward for e in batch])
                next_states = torch.FloatTensor([e.next_state for e in batch])
                dones = torch.FloatTensor([e.done for e in batch])
                
                # Training step based on algorithm
                if config.algorithm == RLAlgorithm.DQN:
                    # DQN update
                    current_q_values = agent.model(states).gather(1, actions.unsqueeze(1))
                    next_q_values = agent.model(next_states).max(1)[0].detach()
                    target_q_values = rewards + (config.gamma * next_q_values * (1 - dones))
                    
                    loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
                    
                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()
                    
                elif config.algorithm == RLAlgorithm.PPO:
                    # PPO update (simplified)
                    # In production, would implement full PPO with advantages
                    probs = agent.model(states)
                    dist = Categorical(probs)
                    log_probs = dist.log_prob(actions)
                    
                    # Simplified policy loss
                    loss = -(log_probs * rewards).mean()
                    
                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()
                    
                # Update metrics
                agent.training_step += 1
                agent.update_epsilon(config.epsilon_decay, config.epsilon_min)
                
                # Update job metrics
                self.training_jobs[job_id]["metrics"]["episodes"] = episode
                self.training_jobs[job_id]["metrics"]["total_reward"] += rewards.mean().item()
                self.training_jobs[job_id]["metrics"]["avg_reward"] = (
                    self.training_jobs[job_id]["metrics"]["total_reward"] / (episode + 1)
                )
                
                # Log to Weights & Biases
                if self.use_wandb:
                    wandb.log({
                        "episode": episode,
                        "loss": loss.item(),
                        "epsilon": agent.epsilon,
                        "avg_reward": self.training_jobs[job_id]["metrics"]["avg_reward"]
                    })
                    
                # Checkpoint periodically
                if episode % 100 == 0:
                    self._save_checkpoint(agent, job_id, episode)
                    
            self.training_jobs[job_id]["status"] = TrainingStatus.COMPLETED.value
            logger.info(f"Training job {job_id} completed")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.training_jobs[job_id]["status"] = TrainingStatus.FAILED.value
            self.training_jobs[job_id]["error"] = str(e)
            
    async def _distributed_training(self, job_id: str, config: AgentTrainingConfig):
        """Distributed training using Ray"""
        try:
            self.training_jobs[job_id]["status"] = TrainingStatus.RUNNING.value
            
            # Configure Ray RLlib trainer
            trainer_config = {
                "env": config.environment,
                "num_workers": 4,
                "num_gpus": 0,
                "lr": config.learning_rate,
                "gamma": config.gamma,
                "train_batch_size": config.batch_size,
            }
            
            # Select trainer based on algorithm
            if config.algorithm == RLAlgorithm.PPO:
                trainer = ppo.PPOTrainer(config=trainer_config)
            elif config.algorithm == RLAlgorithm.DQN:
                trainer = dqn.DQNTrainer(config=trainer_config)
            else:
                trainer = a3c.A3CTrainer(config=trainer_config)
                
            # Training loop
            for episode in range(config.num_episodes):
                result = trainer.train()
                
                # Update metrics
                self.training_jobs[job_id]["metrics"]["episodes"] = episode
                self.training_jobs[job_id]["metrics"]["avg_reward"] = result["episode_reward_mean"]
                
                # Checkpoint
                if episode % 100 == 0:
                    checkpoint = trainer.save()
                    logger.info(f"Checkpoint saved at {checkpoint}")
                    
            self.training_jobs[job_id]["status"] = TrainingStatus.COMPLETED.value
            logger.info(f"Distributed training job {job_id} completed")
            
        except Exception as e:
            logger.error(f"Distributed training failed: {e}")
            self.training_jobs[job_id]["status"] = TrainingStatus.FAILED.value
            
    def _save_checkpoint(self, agent: MDPAgent, job_id: str, episode: int):
        """Save model checkpoint"""
        try:
            checkpoint = {
                "agent_id": agent.agent_id,
                "episode": episode,
                "model_state": agent.model.state_dict() if agent.model else None,
                "optimizer_state": agent.optimizer.state_dict() if agent.optimizer else None,
                "epsilon": agent.epsilon,
                "training_step": agent.training_step
            }
            
            checkpoint_path = f"checkpoints/{job_id}_episode_{episode}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            
            logger.info(f"Saved checkpoint for job {job_id} at episode {episode}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""
        
        def on_task_completed(event):
            """Handle task completion for reward signals"""
            task_id = event.data.get('task_id')
            agent_id = event.data.get('agent_id')
            success = event.data.get('success', False)
            
            # Generate reward based on task completion
            reward = 1.0 if success else -0.5
            
            # Create reward signal
            signal = RewardSignal(
                agent_id=agent_id,
                task_id=task_id,
                reward=reward,
                components={
                    "completion": reward,
                    "time_bonus": 0.0,
                    "quality_bonus": 0.0
                },
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Process reward
            asyncio.create_task(self.process_reward_signal(signal))
            
        # Register handlers
        self.dal.event_bus.on(EventChannel.TASK_COMPLETED, on_task_completed)
        
        logger.info("Event handlers registered for RL server")
        
    async def startup(self):
        """Startup tasks"""
        logger.info("RL Server starting up...")
        
        # Verify connections
        health = self.dal.health_check()
        if not health['database']:
            logger.warning("Database not available")
        if not health['cache']:
            logger.warning("Cache not available")
            
        # Load existing agents from cache
        for key in self.cache.redis_client.keys("rl_agent:*"):
            agent_data = self.cache.get(key)
            if agent_data:
                # Recreate agent
                agent = MDPAgent(
                    agent_id=agent_data["agent_id"],
                    state_space=agent_data["state_space"],
                    action_space=agent_data["action_space"],
                    algorithm=RLAlgorithm(agent_data["algorithm"])
                )
                self.agents[agent.agent_id] = agent
                
        logger.info(f"RL Server ready with {len(self.agents)} agents loaded")
        
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("RL Server shutting down...")
        
        # Save agent states
        for agent_id, agent in self.agents.items():
            if agent.model:
                checkpoint_path = f"checkpoints/{agent_id}_final.pt"
                torch.save({
                    "model_state": agent.model.state_dict(),
                    "epsilon": agent.epsilon,
                    "training_step": agent.training_step
                }, checkpoint_path)
                
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = RLServer()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("RL_SERVER_PORT", 8008))
    logger.info(f"Starting Enterprise RL Server on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()