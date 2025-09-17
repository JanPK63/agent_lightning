"""
Comprehensive Environment Manager
Consolidates EnvManager and RolloutWorker components from charter tools list
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
import logging
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .env_manager import EnvironmentPool, TrajectoryBatch, Trajectory

logger = logging.getLogger(__name__)

@dataclass
class EnvSpec:
    """Environment specification"""
    id: str
    num_envs: int = 1
    seed: Optional[int] = None
    max_episode_steps: Optional[int] = None
    render_mode: Optional[str] = None
    kwargs: Dict[str, Any] = None

@dataclass
class RolloutConfig:
    """Rollout collection configuration"""
    max_steps_per_episode: int = 1000
    max_episodes: Optional[int] = None
    deterministic: bool = False
    render: bool = False
    timeout: float = 30.0  # seconds

class PolicyInterface:
    """Interface for policy interaction"""
    
    def get_action(self, observation, deterministic: bool = False):
        """Get action from policy given observation"""
        raise NotImplementedError
    
    def reset(self):
        """Reset policy state if needed"""
        pass

class RandomPolicy(PolicyInterface):
    """Random policy for testing"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation, deterministic: bool = False):
        return self.action_space.sample()

class EnvManager:
    """Comprehensive environment manager"""
    
    def __init__(self):
        self.env_pools: Dict[str, EnvironmentPool] = {}
        self.env_specs: Dict[str, EnvSpec] = {}
        self._lock = threading.Lock()
    
    def register_env(self, name: str, env_spec: EnvSpec):
        """Register environment specification"""
        with self._lock:
            self.env_specs[name] = env_spec
            logger.info(f"Registered environment: {name}")
    
    def create_env_pool(self, name: str, env_spec: Optional[EnvSpec] = None) -> EnvironmentPool:
        """Create or get environment pool"""
        with self._lock:
            if name in self.env_pools:
                return self.env_pools[name]
            
            spec = env_spec or self.env_specs.get(name)
            if not spec:
                raise ValueError(f"Environment spec not found: {name}")
            
            pool = EnvironmentPool(
                env_id=spec.id,
                num_envs=spec.num_envs,
                seed=spec.seed
            )
            
            self.env_pools[name] = pool
            logger.info(f"Created environment pool: {name} with {spec.num_envs} environments")
            return pool
    
    def get_env_pool(self, name: str) -> Optional[EnvironmentPool]:
        """Get existing environment pool"""
        return self.env_pools.get(name)
    
    def close_env_pool(self, name: str):
        """Close specific environment pool"""
        with self._lock:
            if name in self.env_pools:
                self.env_pools[name].close()
                del self.env_pools[name]
                logger.info(f"Closed environment pool: {name}")
    
    def close_all(self):
        """Close all environment pools"""
        with self._lock:
            for name, pool in self.env_pools.items():
                pool.close()
            self.env_pools.clear()
            logger.info("Closed all environment pools")
    
    def list_environments(self) -> List[str]:
        """List all registered environments"""
        return list(self.env_specs.keys())

class RolloutWorker:
    """Enhanced rollout worker with comprehensive functionality"""
    
    def __init__(self, env_manager: EnvManager, worker_id: int = 0):
        self.env_manager = env_manager
        self.worker_id = worker_id
        self.logger = logging.getLogger(f"{__name__}.Worker{worker_id}")
    
    def collect_episode(self, 
                       env_name: str, 
                       policy: PolicyInterface, 
                       config: RolloutConfig,
                       env_idx: int = 0) -> Trajectory:
        """Collect single episode trajectory"""
        env_pool = self.env_manager.get_env_pool(env_name)
        if not env_pool:
            raise ValueError(f"Environment pool not found: {env_name}")
        
        states, actions, rewards, dones, infos = [], [], [], [], []
        
        # Reset environment
        state, info = env_pool.reset(env_idx)
        states.append(state)
        policy.reset()
        
        episode_steps = 0
        start_time = time.time()
        
        while episode_steps < config.max_steps_per_episode:
            # Check timeout
            if time.time() - start_time > config.timeout:
                self.logger.warning(f"Episode timeout after {config.timeout}s")
                break
            
            # Get action from policy
            action = policy.get_action(state, config.deterministic)
            actions.append(action)
            
            # Step environment
            next_state, reward, terminated, truncated, step_info = env_pool.step(env_idx, action)
            
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(step_info)
            
            episode_steps += 1
            
            if terminated or truncated:
                break
            
            state = next_state
            states.append(state)
        
        return Trajectory(states, actions, rewards, dones, infos)
    
    def collect_rollouts(self, 
                        env_name: str, 
                        policy: PolicyInterface, 
                        config: RolloutConfig,
                        num_episodes: Optional[int] = None) -> TrajectoryBatch:
        """Collect multiple episode trajectories"""
        env_pool = self.env_manager.get_env_pool(env_name)
        if not env_pool:
            raise ValueError(f"Environment pool not found: {env_name}")
        
        trajectories = []
        episodes_collected = 0
        target_episodes = num_episodes or config.max_episodes or 1
        
        while episodes_collected < target_episodes:
            # Use round-robin across environments in pool
            env_idx = episodes_collected % env_pool.num_envs
            
            try:
                trajectory = self.collect_episode(env_name, policy, config, env_idx)
                trajectories.append(trajectory)
                episodes_collected += 1
                
                self.logger.debug(f"Collected episode {episodes_collected}/{target_episodes}, "
                                f"reward: {sum(trajectory.rewards):.2f}, "
                                f"length: {len(trajectory)}")
                
            except Exception as e:
                self.logger.error(f"Failed to collect episode: {e}")
                break
        
        return TrajectoryBatch(trajectories)
    
    def collect_steps(self, 
                     env_name: str, 
                     policy: PolicyInterface, 
                     config: RolloutConfig,
                     target_steps: int) -> TrajectoryBatch:
        """Collect trajectories until target step count is reached"""
        trajectories = []
        total_steps = 0
        
        while total_steps < target_steps:
            trajectory = self.collect_episode(env_name, policy, config)
            trajectories.append(trajectory)
            total_steps += len(trajectory)
            
            self.logger.debug(f"Collected {total_steps}/{target_steps} steps")
        
        return TrajectoryBatch(trajectories)

class ParallelRolloutManager:
    """Manager for parallel rollout collection"""
    
    def __init__(self, env_manager: EnvManager, num_workers: int = 4):
        self.env_manager = env_manager
        self.num_workers = num_workers
        self.workers = [RolloutWorker(env_manager, i) for i in range(num_workers)]
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    def collect_parallel_rollouts(self, 
                                 env_name: str, 
                                 policy: PolicyInterface, 
                                 config: RolloutConfig,
                                 total_episodes: int) -> TrajectoryBatch:
        """Collect rollouts in parallel across workers"""
        episodes_per_worker = total_episodes // self.num_workers
        remaining_episodes = total_episodes % self.num_workers
        
        # Submit tasks to workers
        futures = []
        for i, worker in enumerate(self.workers):
            worker_episodes = episodes_per_worker + (1 if i < remaining_episodes else 0)
            if worker_episodes > 0:
                future = self.executor.submit(
                    worker.collect_rollouts, env_name, policy, config, worker_episodes
                )
                futures.append(future)
        
        # Collect results
        all_trajectories = []
        for future in as_completed(futures):
            try:
                batch = future.result(timeout=config.timeout)
                all_trajectories.extend(batch.trajectories)
            except Exception as e:
                logger.error(f"Parallel rollout failed: {e}")
        
        return TrajectoryBatch(all_trajectories)
    
    def collect_parallel_steps(self, 
                              env_name: str, 
                              policy: PolicyInterface, 
                              config: RolloutConfig,
                              total_steps: int) -> TrajectoryBatch:
        """Collect steps in parallel across workers"""
        steps_per_worker = total_steps // self.num_workers
        
        # Submit tasks to workers
        futures = []
        for worker in self.workers:
            future = self.executor.submit(
                worker.collect_steps, env_name, policy, config, steps_per_worker
            )
            futures.append(future)
        
        # Collect results
        all_trajectories = []
        for future in as_completed(futures):
            try:
                batch = future.result(timeout=config.timeout)
                all_trajectories.extend(batch.trajectories)
            except Exception as e:
                logger.error(f"Parallel step collection failed: {e}")
        
        return TrajectoryBatch(all_trajectories)
    
    def shutdown(self):
        """Shutdown parallel executor"""
        self.executor.shutdown(wait=True)

class ComprehensiveEnvManager:
    """Main comprehensive environment manager combining all components"""
    
    def __init__(self, num_workers: int = 4):
        self.env_manager = EnvManager()
        self.parallel_manager = ParallelRolloutManager(self.env_manager, num_workers)
        self.policies: Dict[str, PolicyInterface] = {}
    
    def register_environment(self, name: str, env_spec: EnvSpec):
        """Register environment"""
        self.env_manager.register_env(name, env_spec)
    
    def register_policy(self, name: str, policy: PolicyInterface):
        """Register policy"""
        self.policies[name] = policy
    
    def create_random_policy(self, env_name: str) -> PolicyInterface:
        """Create random policy for environment"""
        env_pool = self.env_manager.create_env_pool(env_name)
        if env_pool.envs:
            action_space = env_pool.envs[0].action_space
            return RandomPolicy(action_space)
        raise ValueError(f"Cannot create policy for {env_name}: no environments")
    
    def collect_rollouts(self, 
                        env_name: str, 
                        policy_name: str, 
                        config: RolloutConfig,
                        num_episodes: int,
                        parallel: bool = True) -> TrajectoryBatch:
        """Collect rollouts with specified configuration"""
        # Ensure environment pool exists
        self.env_manager.create_env_pool(env_name)
        
        # Get or create policy
        if policy_name not in self.policies:
            if policy_name == "random":
                self.policies[policy_name] = self.create_random_policy(env_name)
            else:
                raise ValueError(f"Policy not found: {policy_name}")
        
        policy = self.policies[policy_name]
        
        # Collect rollouts
        if parallel and self.parallel_manager.num_workers > 1:
            return self.parallel_manager.collect_parallel_rollouts(
                env_name, policy, config, num_episodes
            )
        else:
            worker = RolloutWorker(self.env_manager)
            return worker.collect_rollouts(env_name, policy, config, num_episodes)
    
    def shutdown(self):
        """Shutdown all components"""
        self.parallel_manager.shutdown()
        self.env_manager.close_all()