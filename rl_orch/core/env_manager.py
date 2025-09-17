"""
Environment Manager for Gymnasium and PettingZoo integration
Replaces mock env_pool with real environment handling
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trajectory:
    """Single trajectory data"""
    states: List[Any]
    actions: List[Any]
    rewards: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]]
    
    def __len__(self):
        return len(self.states)

@dataclass
class TrajectoryBatch:
    """Batch of trajectories"""
    trajectories: List[Trajectory]
    
    def __len__(self):
        return len(self.trajectories)
    
    def total_steps(self):
        return sum(len(traj) for traj in self.trajectories)
    
    def to_dict(self):
        """Convert to dict format for compatibility"""
        all_states, all_actions, all_rewards, all_dones = [], [], [], []
        for traj in self.trajectories:
            all_states.extend(traj.states)
            all_actions.extend(traj.actions)
            all_rewards.extend(traj.rewards)
            all_dones.extend(traj.dones)
        
        return {
            "states": all_states,
            "actions": all_actions,
            "rewards": all_rewards,
            "dones": all_dones
        }

class EnvironmentPool:
    """Pool of environments for parallel rollouts"""
    
    def __init__(self, env_id: str, num_envs: int = 1, seed: Optional[int] = None):
        self.env_id = env_id
        self.num_envs = num_envs
        self.seed = seed
        self.envs = []
        self._init_envs()
    
    def _init_envs(self):
        """Initialize environment pool"""
        for i in range(self.num_envs):
            try:
                env = gym.make(self.env_id)
                if self.seed is not None:
                    env.reset(seed=self.seed + i)
                self.envs.append(env)
                logger.info(f"Created environment {i}: {self.env_id}")
            except Exception as e:
                logger.error(f"Failed to create environment {self.env_id}: {e}")
                # Fallback to CartPole if requested env fails
                env = gym.make("CartPole-v1")
                if self.seed is not None:
                    env.reset(seed=self.seed + i)
                self.envs.append(env)
                logger.warning(f"Using fallback CartPole-v1 for env {i}")
    
    def reset(self, env_idx: int = 0):
        """Reset specific environment"""
        if env_idx < len(self.envs):
            return self.envs[env_idx].reset()
        return None
    
    def step(self, env_idx: int, action):
        """Step specific environment"""
        if env_idx < len(self.envs):
            return self.envs[env_idx].step(action)
        return None, 0, True, True, {}
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()

class RolloutWorker:
    """Worker for collecting rollouts from environment"""
    
    def __init__(self, env_pool: EnvironmentPool):
        self.env_pool = env_pool
    
    def collect_trajectory(self, policy, env_idx: int = 0, max_steps: int = 1000) -> Trajectory:
        """Collect single trajectory"""
        states, actions, rewards, dones, infos = [], [], [], [], []
        
        state, info = self.env_pool.reset(env_idx)
        states.append(state)
        
        for step in range(max_steps):
            # Get action from policy (simplified random policy for now)
            action = self._get_action(policy, state)
            actions.append(action)
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env_pool.step(env_idx, action)
            
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)
            
            if terminated or truncated:
                break
            
            state = next_state
            states.append(state)
        
        return Trajectory(states, actions, rewards, dones, infos)
    
    def collect_rollouts(self, policy, steps_per_epoch: int) -> TrajectoryBatch:
        """Collect multiple trajectories up to steps_per_epoch"""
        trajectories = []
        total_steps = 0
        
        while total_steps < steps_per_epoch:
            for env_idx in range(self.env_pool.num_envs):
                if total_steps >= steps_per_epoch:
                    break
                
                traj = self.collect_trajectory(policy, env_idx)
                trajectories.append(traj)
                total_steps += len(traj)
        
        return TrajectoryBatch(trajectories)
    
    def _get_action(self, policy, state):
        """Get action from policy (simplified)"""
        # For now, use random action - in real implementation this would use the policy
        if hasattr(self.env_pool.envs[0], 'action_space'):
            return self.env_pool.envs[0].action_space.sample()
        return 0

class EnvironmentManager:
    """Main environment manager"""
    
    def __init__(self):
        self.env_pools: Dict[str, EnvironmentPool] = {}
        self.workers: Dict[str, RolloutWorker] = {}
    
    def make_envs(self, env_cfg) -> EnvironmentPool:
        """Create environment pool - replaces mock _make_envs"""
        pool_id = f"{env_cfg.id}_{env_cfg.num_envs}"
        
        if pool_id not in self.env_pools:
            pool = EnvironmentPool(
                env_id=env_cfg.id,
                num_envs=env_cfg.num_envs,
                seed=env_cfg.seed
            )
            worker = RolloutWorker(pool)
            
            self.env_pools[pool_id] = pool
            self.workers[pool_id] = worker
            
            logger.info(f"Created environment pool: {pool_id}")
        
        return self.env_pools[pool_id]
    
    def collect_rollouts(self, env_pool: EnvironmentPool, policy, steps_per_epoch: int) -> TrajectoryBatch:
        """Collect rollouts - replaces mock _collect_rollouts"""
        pool_id = f"{env_pool.env_id}_{env_pool.num_envs}"
        
        if pool_id in self.workers:
            return self.workers[pool_id].collect_rollouts(policy, steps_per_epoch)
        
        # Fallback
        worker = RolloutWorker(env_pool)
        return worker.collect_rollouts(policy, steps_per_epoch)
    
    def close_all(self):
        """Close all environment pools"""
        for pool in self.env_pools.values():
            pool.close()
        self.env_pools.clear()
        self.workers.clear()