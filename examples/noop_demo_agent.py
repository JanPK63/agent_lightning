from typing import Optional
from agentlightning.litagent import LitAgent
from agentlightning.types import (
    TaskInput,
    Rollout,
    RolloutRawResult,
    NamedResources,
)


class SimpleNoOpLitAgent(LitAgent):
    """Minimal LitAgent that returns a fixed reward for demo purposes."""
    
    def __init__(self, trained_agents: Optional[str] = None) -> None:
        super().__init__(trained_agents=trained_agents)

    def training_rollout(
        self,
        task: TaskInput,
        rollout_id: str,
        resources: NamedResources
    ) -> RolloutRawResult:
        reward = 0.0
        if isinstance(task, dict):
            value = task.get("value")
            try:
                reward = float(value)
            except Exception:
                reward = 0.0
        elif isinstance(task, (int, float)):
            reward = float(task)
        return Rollout(rollout_id=rollout_id, final_reward=reward)