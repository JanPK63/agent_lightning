# 🧠 RL Orchestrator --- Implementation Charter

This document describes the role, responsibilities, functions, tools,
capabilities, dependencies, and design patterns required for
implementing a Reinforcement Learning (RL) Orchestrator.

------------------------------------------------------------------------

## 1. Role

-   **Owner of the RL lifecycle**: coordinates environments, policies,
    learners, evaluators, replay buffers, and deployment targets.\
-   **Single control plane**: provides API/CLI to start, stop, resume,
    sweep, and promote experiments.\
-   **Stateful scheduler**: manages workloads locally or in distributed
    clusters and persists experiment metadata.

------------------------------------------------------------------------

## 2. Primary Responsibilities

1.  **Experiment Lifecycle**: config validation, spin up components,
    orchestrate train/eval cycles, checkpoint, resume, early stop.\
2.  **Data Pipeline**: rollouts → replay buffer → learner updates.\
3.  **Evaluation & Selection**: periodic evals; compute metrics; enforce
    gates (min reward, stability).\
4.  **Scaling**: parallel actors, distributed learners, resource-aware
    scheduling.\
5.  **Observability**: metrics, traces, artifacts, reproducibility.\
6.  **Safety/Guardrails**: reward/action sanity checks, divergence
    detection.\
7.  **Deployment**: package trained policy, register/export, versioning.

------------------------------------------------------------------------

## 3. Key Capabilities

-   Support multiple algorithms: PPO, A2C, DQN/QR-DQN, SAC, TD3.\
-   Env support: Gymnasium, PettingZoo, custom vectorized envs.\
-   Distributed execution: Ray, Torch Distributed.\
-   Curriculum & domain randomization.\
-   Hyperparameter sweeps with early stop.\
-   Checkpointing (policy, optimizer, buffer, RNG).\
-   Reproducibility: seeds, config hashing.\
-   Safety checks: NaN detection, reward clipping.\
-   Export: TorchScript, ONNX.

------------------------------------------------------------------------

## 4. Orchestration Loop (Pseudocode)

``` python
def run_experiment(cfg: ExperimentConfig):
    state = init_state(cfg)
    env_pool = make_envs(cfg.env)
    policy = build_policy(cfg.policy)
    learner = build_learner(policy, cfg.train)
    buffer = build_replay_buffer(cfg.buffer)
    sched  = build_scheduler(cfg.resources)

    for epoch in range(cfg.train.epochs):
        # Collect rollouts
        traj_batch = collect_rollouts(env_pool, policy, cfg.rollout)
        log(metrics_from(traj_batch), step=state.step)

        if cfg.train.off_policy:
            buffer.add(traj_batch)
            batch = buffer.sample(cfg.train.batch_size)
        else:
            batch = traj_batch

        # Learn
        loss, stats = learner.update(batch)
        log(stats | {"loss": loss}, step=state.step)

        # Evaluate
        if should_eval(epoch, cfg.eval):
            eval_metrics = evaluate(policy, cfg.eval)
            log(eval_metrics, step=state.step)
            if gates_failed(eval_metrics, cfg.gates):
                maybe_early_stop(state, reason="gate_failed")

        # Checkpoint
        if should_checkpoint(epoch, cfg.ckpt):
            save_checkpoint(policy, learner, buffer, state)

        # Scheduler hooks
        sched.tick(throughput=stats["samples_per_sec"], queue=buffer.size())

        state.step += 1

    finalize_and_register(policy, state)
```

------------------------------------------------------------------------

## 5. Interfaces, Tools & APIs

**External Interfaces**\
- CLI: `rlctl launch -f config.yaml`, `rlctl resume`, `rlctl sweep`,
`rlctl promote`.\
- Python API: `orchestrator.run(cfg)`, `orchestrator.resume(run_id)`.\
- HTTP API (optional via FastAPI): `POST /experiments`,
`GET /runs/{id}`.

**Tools (modules)**\
- EnvManager, RolloutWorker, ReplayBuffer, Learner, Evaluator,
Scheduler, Logger/Tracker, Registry, Guardrails.

------------------------------------------------------------------------

## 6. Data Models (Pydantic)

``` python
class EnvConfig(BaseModel):
    id: str
    num_envs: int = 8
    seed: int = 42

class PolicyConfig(BaseModel):
    algo: Literal["ppo","sac","dqn"]
    network: Dict[str, Any]
    discrete: bool

class TrainConfig(BaseModel):
    epochs: int
    steps_per_epoch: int
    off_policy: bool = False
    batch_size: int
    lr: float
    gamma: float

class ExperimentConfig(BaseModel):
    name: str
    env: EnvConfig
    policy: PolicyConfig
    train: TrainConfig
```

------------------------------------------------------------------------

## 7. Config Example (YAML)

``` yaml
name: "ppo_cartpole_mvp"
env:
  id: "CartPole-v1"
  num_envs: 16
  seed: 123
policy:
  algo: "ppo"
  network: { hidden_sizes: [128, 128], activation: "tanh" }
  discrete: true
train:
  epochs: 50
  steps_per_epoch: 4096
  off_policy: false
  batch_size: 2048
  lr: 3e-4
  gamma: 0.99
```

------------------------------------------------------------------------

## 8. Dependencies

-   **Core RL**: torch, numpy, gymnasium, pettingzoo.\
-   **Distributed**: ray, torch.distributed, accelerate.\
-   **Config**: pydantic, hydra-core.\
-   **Observability**: mlflow, wandb, tensorboard, rich.\
-   **Export**: onnx, torchscript.\
-   **Testing**: pytest, hypothesis, ruff, black.

------------------------------------------------------------------------

## 9. Observability & Artifacts

-   Metrics: reward, loss, KL, entropy, throughput.\
-   Data health: NaN %, replay buffer stats.\
-   System: GPU util, RAM, latency.\
-   Artifacts: checkpoints, configs, eval videos, exports.\
-   Lineage: run ID, parent sweep, commit hash.

------------------------------------------------------------------------

## 10. Failure Handling & Guardrails

-   NaN detection, gradient clipping.\
-   Replay poisoning detection.\
-   Actor heartbeat + auto-restart.\
-   Checkpoint integrity via checksums.\
-   Early stopping on poor metrics.

------------------------------------------------------------------------

## 11. Milestones

-   **M0 Skeleton**: config parsing, CLI stub.\
-   **M1 MVP**: PPO on CartPole with checkpointing + logging.\
-   **M2 Distributed**: rollout actors with Ray.\
-   **M3 Sweeps & Gates**: hyperparam sweeps, early stopping.\
-   **M4 Service API**: FastAPI endpoints.\
-   **M5 Hardening**: guardrails, retries, docs, tests.

------------------------------------------------------------------------

## 12. Folder Structure

    rl_orch/
      cli/rlctl.py
      core/orchestrator.py
      rl/{envs.py, rollout.py, replay.py, learner.py, eval.py}
      io/{logging.py, checkpoints.py}
      api/{server.py, schemas.py}
      configs/{experiment.yaml, sweep.yaml}
      export/{onnx.py, torchscript.py}
      tests/

------------------------------------------------------------------------

**✅ Conclusion**: This RL Orchestrator provides a control plane for
reproducible, scalable, and safe reinforcement learning experiments,
from config to deployment.
