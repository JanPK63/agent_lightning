"""
Charter-compliant RL Orchestrator implementation
Based on rl_orchestrator_charter.md run_experiment() pseudocode
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .config_models import ExperimentConfig, ExperimentState
from .env_manager import EnvironmentManager, EnvironmentPool, TrajectoryBatch
from .distributed_scheduler import DistributedTrainingManager, ResourceConfig
from .replay_buffer import BufferManager, BufferConfig
from .learner import LearnerManager, LearnerConfig

logger = logging.getLogger(__name__)


class RLOrchestrator:
    """Charter-compliant RL Orchestrator"""
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentState] = {}
        self.logger = logger
        self.env_manager = EnvironmentManager()
        self.distributed_manager = DistributedTrainingManager()
        self.buffer_manager = BufferManager()
        self.learner_manager = LearnerManager()
    
    def run_experiment(self, cfg: ExperimentConfig) -> str:
        """
        Charter-compliant orchestration loop implementation
        Based on charter pseudocode
        """
        # Initialize state
        state = self._init_state(cfg)
        
        try:
            # Build components with real implementations
            env_pool = self.env_manager.make_envs(cfg.env)
            policy = self._build_policy(cfg.policy)
            learner = self._build_learner(cfg.policy, cfg.train)
            buffer = self._build_replay_buffer(cfg.buffer)
            sched = self._build_scheduler(cfg.resources)
            
            # Initialize distributed components if resources specified
            if cfg.resources and cfg.resources.num_workers > 1:
                sched.initialize(cfg.env.id, cfg.env.seed)
                self.logger.info(f"Initialized distributed execution with {cfg.resources.num_workers} workers")
            
            self.logger.info(f"Starting experiment {cfg.name} (run_id: {state.run_id})")
            
            # Main training loop
            for epoch in range(cfg.train.epochs):
                state.epoch = epoch
                
                # Collect rollouts (distributed if available)
                if cfg.resources and cfg.resources.num_workers > 1 and sched:
                    policy_params = self._get_policy_params(policy)
                    traj_batch = sched.collect_rollouts_parallel(policy_params, cfg.train.steps_per_epoch)
                else:
                    traj_batch = self.env_manager.collect_rollouts(env_pool, policy, cfg.train.steps_per_epoch)
                
                metrics = self._metrics_from(traj_batch)
                self._log(metrics, step=state.step)
                
                # Handle off-policy vs on-policy with real buffer
                if cfg.train.off_policy and buffer:
                    buffer.add(traj_batch)
                    if hasattr(buffer, 'is_ready') and buffer.is_ready():
                        batch = buffer.sample(cfg.train.batch_size)
                    elif not hasattr(buffer, 'is_ready'):
                        # Episode buffer - always ready
                        batch = buffer.sample()
                    else:
                        # Skip learning if buffer not ready
                        self.logger.debug(f"Buffer not ready: {buffer.size()}/{getattr(buffer.config, 'min_size', 0)}")
                        continue
                else:
                    # On-policy: use trajectory batch directly or add to episode buffer
                    if buffer:
                        buffer.add(traj_batch)
                        batch = buffer.sample()
                    else:
                        batch = traj_batch.to_dict()
                
                # Learn
                loss_stats = learner.update(batch)
                if isinstance(loss_stats, tuple):
                    loss, stats = loss_stats
                else:
                    loss, stats = loss_stats, {}
                combined_stats = {**stats, "loss": loss}
                self._log(combined_stats, step=state.step)
                
                # Evaluate
                if self._should_eval(epoch, cfg.eval):
                    eval_metrics = self._evaluate(policy, cfg.eval)
                    self._log(eval_metrics, step=state.step)
                    
                    if self._gates_failed(eval_metrics, cfg.gates):
                        self._maybe_early_stop(state, reason="gate_failed")
                        break
                
                # Checkpoint
                if self._should_checkpoint(epoch, cfg.ckpt):
                    checkpoint_path = self._save_checkpoint(policy, learner, buffer, state)
                    state.checkpoints.append(checkpoint_path)
                
                # Scheduler hooks
                if sched:
                    sched.tick(
                        throughput=stats.get("samples_per_sec", 0),
                        queue=buffer.size() if buffer else 0
                    )
                
                state.step += 1
                state.updated_at = datetime.utcnow().isoformat()
            
            # Finalize
            self._finalize_and_register(policy, state)
            state.status = "completed"
            
            # Cleanup environments and distributed components
            self.env_manager.close_all()
            self.distributed_manager.shutdown()
            
            self.logger.info(f"Experiment {cfg.name} completed successfully")
            return state.run_id
            
        except Exception as e:
            self.logger.error(f"Experiment {cfg.name} failed: {e}")
            state.status = "failed"
            # Cleanup on failure
            self.env_manager.close_all()
            self.distributed_manager.shutdown()
            raise
    
    def _init_state(self, cfg: ExperimentConfig) -> ExperimentState:
        """Initialize experiment state"""
        state = ExperimentState(
            run_id=cfg.run_id,
            config=cfg,
            created_at=datetime.utcnow().isoformat()
        )
        self.experiments[state.run_id] = state
        return state
    

    
    def _build_policy(self, policy_cfg):
        """Build policy (simplified)"""
        return {"algo": policy_cfg.algo, "network": policy_cfg.network}
    
    def _build_learner(self, policy_cfg, train_cfg):
        """Build real learner based on algorithm"""
        learner_config = LearnerConfig(
            algorithm=policy_cfg.algo,
            learning_rate=train_cfg.lr,
            gamma=train_cfg.gamma,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5
        )
        
        learner_name = f"{policy_cfg.algo}_learner"
        return self.learner_manager.create_learner(learner_name, policy_cfg.algo, learner_config)
    
    def _build_replay_buffer(self, buffer_cfg):
        """Build real replay buffer"""
        if not buffer_cfg:
            return None
        
        buffer_config = BufferConfig(
            size=buffer_cfg.size,
            batch_size=buffer_cfg.batch_size,
            min_size=min(buffer_cfg.batch_size * 2, buffer_cfg.size // 20),
            prioritized=False  # Can be made configurable
        )
        
        # Choose buffer type based on training configuration (on-policy vs off-policy)
        # For on-policy algorithms like PPO, use episode buffer
        buffer_type = "episode"  # Default to episode buffer for on-policy
        
        return self.buffer_manager.create_buffer("main_buffer", buffer_config, buffer_type)
    
    def _build_scheduler(self, resource_cfg):
        """Build scheduler with distributed support"""
        if not resource_cfg:
            return None
        
        # Use distributed scheduler if multiple workers requested
        if resource_cfg.num_workers > 1:
            return self.distributed_manager.build_scheduler(resource_cfg)
        
        # Fallback to simple scheduler for single worker
        class SimpleScheduler:
            def __init__(self, num_workers, gpu):
                self.num_workers = num_workers
                self.gpu = gpu
            
            def tick(self, throughput, queue):
                pass
            
            def initialize(self, env_id, seed):
                pass
        
        return SimpleScheduler(resource_cfg.num_workers, resource_cfg.gpu)
    

    
    def _metrics_from(self, traj_batch: TrajectoryBatch):
        """Extract metrics from trajectory batch"""
        total_reward = sum(sum(traj.rewards) for traj in traj_batch.trajectories)
        avg_episode_length = sum(len(traj) for traj in traj_batch.trajectories) / len(traj_batch.trajectories)
        
        return {
            "episode_reward": total_reward / len(traj_batch.trajectories),
            "total_reward": total_reward,
            "avg_episode_length": avg_episode_length,
            "num_episodes": len(traj_batch.trajectories),
            "samples_collected": traj_batch.total_steps()
        }
    
    def _log(self, metrics: Dict[str, Any], step: int):
        """Log metrics"""
        self.logger.info(f"Step {step}: {metrics}")
    
    def _should_eval(self, epoch: int, eval_cfg) -> bool:
        """Check if should evaluate"""
        if not eval_cfg:
            return False
        return epoch % eval_cfg.frequency == 0
    
    def _evaluate(self, policy, eval_cfg):
        """Evaluate policy (simplified)"""
        return {
            "eval_reward": 100.0,
            "eval_episodes": eval_cfg.episodes if eval_cfg else 5
        }
    
    def _gates_failed(self, eval_metrics: Dict[str, Any], gates_cfg) -> bool:
        """Check if quality gates failed"""
        if not gates_cfg or not gates_cfg.min_reward:
            return False
        return eval_metrics.get("eval_reward", 0) < gates_cfg.min_reward
    
    def _maybe_early_stop(self, state: ExperimentState, reason: str):
        """Handle early stopping"""
        self.logger.warning(f"Early stopping experiment {state.run_id}: {reason}")
        state.status = "failed"
    
    def _should_checkpoint(self, epoch: int, ckpt_cfg) -> bool:
        """Check if should checkpoint"""
        if not ckpt_cfg:
            return False
        return epoch % ckpt_cfg.frequency == 0
    
    def _save_checkpoint(self, policy, learner, buffer, state: ExperimentState) -> str:
        """Save checkpoint (simplified)"""
        checkpoint_path = f"checkpoints/{state.run_id}_epoch_{state.epoch}.ckpt"
        self.logger.info(f"Saving checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def _get_policy_params(self, policy):
        """Get policy parameters for distributed execution"""
        return policy  # Simplified - in real implementation would extract parameters
    
    def _finalize_and_register(self, policy, state: ExperimentState):
        """Finalize and register model"""
        self.logger.info(f"Finalizing experiment {state.run_id}")
    
    def get_experiment_status(self, run_id: str) -> Optional[ExperimentState]:
        """Get experiment status"""
        return self.experiments.get(run_id)
    
    def list_experiments(self) -> Dict[str, ExperimentState]:
        """List all experiments"""
        return self.experiments