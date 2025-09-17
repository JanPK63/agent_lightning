"""
Ray Distributed Computing Configuration for Agent Lightning
Sets up distributed training infrastructure for scalable RL
"""

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.rl import RLTrainer
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
import os


class AgentLightningEnv(MultiAgentEnv):
    """
    Multi-agent environment for Agent Lightning
    Wraps agent execution in a Gym-compatible interface
    """
    
    def __init__(self, config: Dict):
        """Initialize the environment"""
        super().__init__()
        
        self.config = config
        self.num_agents = config.get("num_agents", 4)
        self.max_steps = config.get("max_steps", 50)
        self.shared_reward = config.get("shared_reward", True)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(10)  # Simplified action space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(768,),  # Embedding dimension
            dtype=np.float32
        )
        
        # Agent IDs
        self._agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        self.current_step = 0
        self.episode_rewards = {agent_id: 0.0 for agent_id in self._agent_ids}
        
    def reset(self, *, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_rewards = {agent_id: 0.0 for agent_id in self._agent_ids}
        
        # Initial observations for all agents
        obs = {}
        for agent_id in self._agent_ids:
            obs[agent_id] = self._get_observation(agent_id)
        
        infos = {agent_id: {} for agent_id in self._agent_ids}
        
        return obs, infos
    
    def step(self, actions: Dict):
        """Execute actions and return results"""
        obs = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        # Process actions for each agent
        for agent_id, action in actions.items():
            # Simulate agent execution
            obs[agent_id] = self._get_observation(agent_id)
            
            # Calculate reward (simplified)
            if self.shared_reward:
                reward = self._calculate_shared_reward(actions)
            else:
                reward = self._calculate_individual_reward(agent_id, action)
            
            rewards[agent_id] = reward
            self.episode_rewards[agent_id] += reward
            
            # Check termination
            terminateds[agent_id] = self.current_step >= self.max_steps
            truncateds[agent_id] = False
            infos[agent_id] = {"episode_reward": self.episode_rewards[agent_id]}
        
        # Set special keys for RLlib
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = False
        
        self.current_step += 1
        
        return obs, rewards, terminateds, truncateds, infos
    
    def _get_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for an agent"""
        # Simplified observation (in practice, would be agent state)
        obs = np.random.randn(768).astype(np.float32)
        return obs
    
    def _calculate_shared_reward(self, actions: Dict) -> float:
        """Calculate shared reward for cooperative agents"""
        # Simplified shared reward
        base_reward = 0.1
        cooperation_bonus = 0.05 * len(actions)
        return base_reward + cooperation_bonus
    
    def _calculate_individual_reward(self, agent_id: str, action: int) -> float:
        """Calculate individual reward for an agent"""
        # Simplified individual reward
        return 0.1 + np.random.random() * 0.1


class HierarchicalRLModel(TorchModelV2, nn.Module):
    """
    Hierarchical RL model for Agent Lightning
    Implements high-level and low-level policies
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        hidden_size = model_config.get("fcnet_hiddens", [256, 256])
        
        # High-level policy network
        self.high_level = nn.Sequential(
            nn.Linear(obs_space.shape[0], hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 64)
        )
        
        # Low-level policy network
        self.low_level = nn.Sequential(
            nn.Linear(obs_space.shape[0] + 64, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], num_outputs)
        )
        
        # Value function
        self.value_head = nn.Sequential(
            nn.Linear(obs_space.shape[0], hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], 1)
        )
        
        self._features = None
        self._value = None
    
    def forward(self, input_dict, state, seq_lens):
        """Forward pass"""
        obs = input_dict["obs"]
        
        # High-level features
        high_features = self.high_level(obs)
        
        # Combine with observation for low-level policy
        combined = torch.cat([obs, high_features], dim=-1)
        logits = self.low_level(combined)
        
        # Compute value
        self._value = self.value_head(obs).squeeze(-1)
        self._features = high_features
        
        return logits, state
    
    def value_function(self):
        """Return value function output"""
        return self._value


class RayDistributedTrainer:
    """
    Distributed training orchestrator using Ray
    Manages distributed RL training for Agent Lightning
    """
    
    def __init__(self, 
                 num_workers: int = 4,
                 num_gpus: int = 1,
                 checkpoint_dir: str = "./ray_checkpoints"):
        """
        Initialize Ray distributed trainer
        
        Args:
            num_workers: Number of parallel workers
            num_gpus: Number of GPUs to use
            checkpoint_dir: Directory for checkpoints
        """
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.checkpoint_dir = checkpoint_dir
        
        # Ray configuration
        self.ray_initialized = False
        self.trainer = None
        
        print(f"🌟 Ray Distributed Trainer initialized")
        print(f"   Workers: {num_workers}")
        print(f"   GPUs: {num_gpus}")
        print(f"   Checkpoint dir: {checkpoint_dir}")
    
    def initialize_ray(self, address: str = "auto"):
        """Initialize Ray cluster"""
        if not self.ray_initialized:
            ray.init(
                address=address,
                num_cpus=self.num_workers * 2,  # 2 CPUs per worker
                num_gpus=self.num_gpus,
                object_store_memory=4_000_000_000,  # 4GB object store
                _temp_dir="/tmp/ray",
                ignore_reinit_error=True
            )
            
            self.ray_initialized = True
            
            print(f"✅ Ray cluster initialized")
            print(f"   Nodes: {len(ray.nodes())}")
            print(f"   Available CPUs: {ray.available_resources().get('CPU', 0)}")
            print(f"   Available GPUs: {ray.available_resources().get('GPU', 0)}")
    
    def create_ppo_config(self) -> PPOConfig:
        """Create PPO configuration for hierarchical RL"""
        config = PPOConfig()
        
        # Training configuration
        config.training(
            lr=1e-5,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
            lambda_=0.95,
            gamma=0.99,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            kl_target=0.01,
            model={
                "custom_model": "hierarchical_rl_model",
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
                "use_lstm": True,
                "max_seq_len": 100,
                "lstm_cell_size": 256,
            }
        )
        
        # Resources configuration
        config.resources(
            num_gpus=self.num_gpus / self.num_workers if self.num_gpus > 0 else 0,
            num_cpus_per_worker=2,
        )
        
        # Rollout configuration
        config.rollouts(
            num_rollout_workers=self.num_workers,
            rollout_fragment_length=200,
            batch_mode="complete_episodes",
        )
        
        # Environment configuration
        config.environment(
            env=AgentLightningEnv,
            env_config={
                "num_agents": 4,
                "max_steps": 50,
                "shared_reward": True
            }
        )
        
        # Multi-agent configuration
        config.multi_agent(
            policies={
                f"agent_{i}": (None, None, None, {})
                for i in range(4)
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
        )
        
        # Evaluation configuration
        config.evaluation(
            evaluation_interval=10,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=True,
            evaluation_num_workers=1,
        )
        
        return config
    
    def train(self, num_iterations: int = 100):
        """Run distributed training"""
        if not self.ray_initialized:
            self.initialize_ray()
        
        # Register custom model
        ModelCatalog.register_custom_model(
            "hierarchical_rl_model",
            HierarchicalRLModel
        )
        
        # Create PPO trainer
        config = self.create_ppo_config()
        self.trainer = config.build()
        
        print(f"\n🚀 Starting distributed training for {num_iterations} iterations...")
        
        results = []
        for iteration in range(num_iterations):
            # Train for one iteration
            result = self.trainer.train()
            results.append(result)
            
            # Log progress
            if iteration % 10 == 0:
                print(f"\nIteration {iteration}:")
                print(f"  Episode reward mean: {result.get('episode_reward_mean', 0):.3f}")
                print(f"  Episodes this iter: {result.get('episodes_this_iter', 0)}")
                print(f"  Training time: {result.get('time_this_iter_s', 0):.1f}s")
            
            # Save checkpoint
            if iteration % 50 == 0 and iteration > 0:
                checkpoint = self.trainer.save(self.checkpoint_dir)
                print(f"💾 Checkpoint saved: {checkpoint}")
        
        print("\n✅ Training complete!")
        return results
    
    def evaluate(self, checkpoint_path: Optional[str] = None):
        """Evaluate the trained model"""
        if checkpoint_path and self.trainer:
            self.trainer.restore(checkpoint_path)
        
        if not self.trainer:
            print("❌ No trainer available for evaluation")
            return None
        
        print("\n📊 Evaluating model...")
        
        # Run evaluation
        eval_results = self.trainer.evaluate()
        
        print(f"✅ Evaluation Results:")
        print(f"  Episode reward mean: {eval_results.get('evaluation', {}).get('episode_reward_mean', 0):.3f}")
        print(f"  Episodes evaluated: {eval_results.get('evaluation', {}).get('episodes_this_iter', 0)}")
        
        return eval_results
    
    def shutdown(self):
        """Shutdown Ray cluster"""
        if self.trainer:
            self.trainer.stop()
        
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False
            print("🛑 Ray cluster shutdown complete")


# Utility functions for distributed execution
@ray.remote
class DistributedAgentWorker:
    """Remote worker for distributed agent execution"""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.episodes_processed = 0
    
    def process_episode(self, task: Dict) -> Dict:
        """Process a single episode"""
        # Simulate agent execution
        result = {
            "worker_id": self.worker_id,
            "task": task,
            "transitions": [],
            "reward": np.random.random(),
            "success": np.random.random() > 0.5
        }
        
        self.episodes_processed += 1
        return result
    
    def get_stats(self) -> Dict:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "episodes_processed": self.episodes_processed
        }


def setup_distributed_workers(num_workers: int = 4) -> List:
    """Setup distributed workers"""
    workers = [
        DistributedAgentWorker.remote(worker_id=i)
        for i in range(num_workers)
    ]
    print(f"✅ Created {num_workers} distributed workers")
    return workers


def distributed_data_collection(workers: List, tasks: List[Dict]) -> List[Dict]:
    """Collect data using distributed workers"""
    # Distribute tasks among workers
    futures = []
    for i, task in enumerate(tasks):
        worker_idx = i % len(workers)
        future = workers[worker_idx].process_episode.remote(task)
        futures.append(future)
    
    # Collect results
    results = ray.get(futures)
    
    print(f"✅ Collected {len(results)} episodes from {len(workers)} workers")
    return results


# Main execution
if __name__ == "__main__":
    print("🚀 Testing Ray Distributed Configuration")
    print("=" * 60)
    
    # Initialize Ray
    trainer = RayDistributedTrainer(
        num_workers=2,  # Reduced for testing
        num_gpus=0,  # Set to 0 for CPU-only testing
        checkpoint_dir="./ray_checkpoints"
    )
    
    trainer.initialize_ray()
    
    # Setup distributed workers
    print("\n🔧 Setting up distributed workers...")
    workers = setup_distributed_workers(num_workers=2)
    
    # Test distributed data collection
    print("\n📊 Testing distributed data collection...")
    test_tasks = [{"task_id": i, "content": f"Task {i}"} for i in range(10)]
    results = distributed_data_collection(workers, test_tasks)
    print(f"   Collected {len(results)} results")
    
    # Test training (reduced iterations for demo)
    print("\n🎯 Testing distributed training...")
    try:
        train_results = trainer.train(num_iterations=2)
        print("   Training test successful!")
    except Exception as e:
        print(f"   Training test failed: {e}")
    
    # Cleanup
    trainer.shutdown()
    
    print("\n✅ Ray distributed configuration test complete!")