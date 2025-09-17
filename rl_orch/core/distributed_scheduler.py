"""
Distributed execution scheduler using Ray
Implements parallel rollout collection and distributed training
"""

import ray
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .env_manager import EnvironmentPool, TrajectoryBatch, RolloutWorker

logger = logging.getLogger(__name__)

@dataclass
class ResourceConfig:
    """Resource configuration for distributed execution"""
    num_workers: int = 4
    gpu: bool = False
    cpu_per_worker: float = 1.0
    memory_per_worker: int = 1024  # MB

@ray.remote
class DistributedRolloutWorker:
    """Ray actor for distributed rollout collection"""
    
    def __init__(self, env_id: str, seed: int = None):
        self.env_id = env_id
        self.seed = seed
        self.env_pool = None
        self.worker = None
        self._init_env()
    
    def _init_env(self):
        """Initialize environment in the remote worker"""
        from .config_models import EnvConfig
        
        env_cfg = EnvConfig(
            id=self.env_id,
            num_envs=1,  # Single env per worker
            seed=self.seed
        )
        
        # Create local environment pool
        self.env_pool = EnvironmentPool(
            env_id=env_cfg.id,
            num_envs=env_cfg.num_envs,
            seed=env_cfg.seed
        )
        self.worker = RolloutWorker(self.env_pool)
    
    def collect_trajectory(self, policy_params: Dict[str, Any], max_steps: int = 1000):
        """Collect single trajectory with given policy parameters"""
        return self.worker.collect_trajectory(policy_params, env_idx=0, max_steps=max_steps)
    
    def collect_rollouts(self, policy_params: Dict[str, Any], num_trajectories: int):
        """Collect multiple trajectories"""
        trajectories = []
        for _ in range(num_trajectories):
            traj = self.collect_trajectory(policy_params)
            trajectories.append(traj)
        return TrajectoryBatch(trajectories)
    
    def close(self):
        """Close environment"""
        if self.env_pool:
            self.env_pool.close()

class DistributedScheduler:
    """Distributed scheduler for parallel rollout collection"""
    
    def __init__(self, resource_cfg: ResourceConfig):
        self.resource_cfg = resource_cfg
        self.workers: List[ray.ObjectRef] = []
        self.is_initialized = False
        
    def initialize(self, env_id: str, seed: int = None):
        """Initialize Ray and create worker pool"""
        if not self.is_initialized:
            # Initialize Ray if not already done
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Create distributed workers
            self.workers = []
            for i in range(self.resource_cfg.num_workers):
                worker_seed = seed + i if seed is not None else None
                worker = DistributedRolloutWorker.remote(env_id, worker_seed)
                self.workers.append(worker)
            
            self.is_initialized = True
            logger.info(f"Initialized {len(self.workers)} distributed workers")
    
    def collect_rollouts_parallel(self, policy_params: Dict[str, Any], total_steps: int) -> TrajectoryBatch:
        """Collect rollouts in parallel across workers"""
        if not self.is_initialized:
            raise RuntimeError("Scheduler not initialized. Call initialize() first.")
        
        # Distribute work across workers
        steps_per_worker = total_steps // len(self.workers)
        trajectories_per_worker = max(1, steps_per_worker // 200)  # Assume ~200 steps per trajectory
        
        # Submit parallel rollout tasks
        futures = []
        for worker in self.workers:
            future = worker.collect_rollouts.remote(policy_params, trajectories_per_worker)
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        # Combine all trajectory batches
        all_trajectories = []
        for batch in results:
            all_trajectories.extend(batch.trajectories)
        
        return TrajectoryBatch(all_trajectories)
    
    def tick(self, throughput: float, queue: int):
        """Scheduler tick for monitoring and adjustment"""
        # Simple monitoring - could be extended for dynamic scaling
        logger.debug(f"Scheduler tick - Throughput: {throughput:.2f}, Queue: {queue}")
    
    def shutdown(self):
        """Shutdown all workers and Ray"""
        if self.workers:
            # Close all workers
            futures = [worker.close.remote() for worker in self.workers]
            ray.get(futures)
            self.workers.clear()
        
        self.is_initialized = False
        logger.info("Distributed scheduler shutdown complete")

@ray.remote
class DistributedLearner:
    """Ray actor for distributed learning"""
    
    def __init__(self, policy_config: Dict[str, Any], train_config: Dict[str, Any]):
        self.policy_config = policy_config
        self.train_config = train_config
        self.step_count = 0
    
    def update(self, batch_data: Dict[str, Any]):
        """Update policy with batch data"""
        # Simplified learning update
        self.step_count += 1
        
        # Simulate learning metrics
        loss = 0.1 * (1.0 / (1.0 + self.step_count * 0.01))  # Decreasing loss
        
        stats = {
            "loss": loss,
            "grad_norm": 0.5,
            "samples_per_sec": 1000,
            "learning_rate": self.train_config.get("lr", 0.001),
            "step": self.step_count
        }
        
        return loss, stats
    
    def get_policy_params(self):
        """Get current policy parameters"""
        return {"step": self.step_count, "config": self.policy_config}

class DistributedTrainingManager:
    """Manager for distributed training components"""
    
    def __init__(self):
        self.scheduler: Optional[DistributedScheduler] = None
        self.learner: Optional[ray.ObjectRef] = None
    
    def build_scheduler(self, resource_cfg: ResourceConfig) -> DistributedScheduler:
        """Build distributed scheduler"""
        self.scheduler = DistributedScheduler(resource_cfg)
        return self.scheduler
    
    def build_learner(self, policy_config: Dict[str, Any], train_config: Dict[str, Any]):
        """Build distributed learner"""
        self.learner = DistributedLearner.remote(policy_config, train_config)
        return self.learner
    
    def shutdown(self):
        """Shutdown all distributed components"""
        if self.scheduler:
            self.scheduler.shutdown()
        
        if self.learner:
            # Learner will be cleaned up by Ray
            self.learner = None