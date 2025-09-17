"""
Charter-compliant Pydantic configuration models for RL Orchestrator
Based on rl_orchestrator_charter.md specifications
"""

from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


class EnvConfig(BaseModel):
    """Environment configuration"""
    id: str
    num_envs: int = 8
    seed: int = 42


class PolicyConfig(BaseModel):
    """Policy configuration"""
    algo: Literal["ppo", "sac", "dqn"]
    network: Dict[str, Any]
    discrete: bool


class TrainConfig(BaseModel):
    """Training configuration"""
    epochs: int
    steps_per_epoch: int
    off_policy: bool = False
    batch_size: int
    lr: float
    gamma: float


class BufferConfig(BaseModel):
    """Replay buffer configuration"""
    size: int = 100000
    batch_size: int = 32


class EvalConfig(BaseModel):
    """Evaluation configuration"""
    frequency: int = 10  # Every N epochs
    episodes: int = 5
    render: bool = False


class GatesConfig(BaseModel):
    """Quality gates configuration"""
    min_reward: Optional[float] = None
    max_episodes_without_improvement: int = 20


class ResourceConfig(BaseModel):
    """Resource configuration"""
    num_workers: int = 4
    gpu: bool = False


class CheckpointConfig(BaseModel):
    """Checkpoint configuration"""
    frequency: int = 50  # Every N epochs
    keep_last: int = 5


class ExperimentConfig(BaseModel):
    """Complete experiment configuration"""
    name: str
    env: EnvConfig
    policy: PolicyConfig
    train: TrainConfig
    buffer: Optional[BufferConfig] = None
    eval: Optional[EvalConfig] = None
    gates: Optional[GatesConfig] = None
    resources: Optional[ResourceConfig] = None
    ckpt: Optional[CheckpointConfig] = None
    
    # Runtime fields
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: Optional[str] = None


class ExperimentState(BaseModel):
    """Experiment runtime state"""
    run_id: str
    config: ExperimentConfig
    step: int = 0
    epoch: int = 0
    status: Literal["running", "paused", "completed", "failed"] = "running"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    checkpoints: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: Optional[str] = None