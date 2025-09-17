#!/usr/bin/env python3
from typing import Optional
from agentlightning.litagent import LitAgent
from agentlightning.types import (
    TaskInput,
    Rollout,
    RolloutRawResult,
    NamedResources,
    PromptTemplate,
)
from agentlightning.client import DevTaskLoader
from agentlightning.runner import AgentRunner
from agentlightning.tracer import TripletExporter, NoOpTracer


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


def main():
    tasks = [
        {"value": 1.0},
        {"value": 2.0},
        {"value": 3.5}
    ]
    resources = {"example_prompt": PromptTemplate(template="demo", engine="f-string")}
    client = DevTaskLoader(tasks=tasks, resources=resources)
    agent = SimpleNoOpLitAgent()
    triplet_exporter = TripletExporter()
    runner = AgentRunner(
        agent=agent,
        client=client,
        tracer=NoOpTracer(),
        triplet_exporter=triplet_exporter,
        max_tasks=3
    )
    runner.iter()
    print("Demo complete. Rollouts:", client.rollouts)
    assert len(client.rollouts) > 0, "No rollouts received"


if __name__ == "__main__":
    main()