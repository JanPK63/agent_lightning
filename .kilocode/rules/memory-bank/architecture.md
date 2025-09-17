# Memory Bank — architecture.md

Overzicht van systeemarchitectuur en belangrijkste componenten.

Architectuurcomponenten:

1. Training server
   - Beheert datasets, batches en RL-trainloops.
   - Implementatievoorbeelden: [`agent_client.py`](agent_client.py:1), server scaffolding in documentatie.

2. Agents
   - Lichtgewicht agentclients die samples verwerken en trajecten terugsturen.
   - Relevante modules: [`agent_executor_fix.py`](agent_executor_fix.py:1), [`agent_communication_protocol.py`](agent_communication_protocol.py:1)

3. Memory & Context
   - Episodic en semantic memory met retrieval en prune-logica.
   - Implementatie: [`memory_manager.py`](memory_manager.py:1), [`shared_memory_system.py`](shared_memory_system.py:1)

4. Reward & Evaluation
   - Centraliseer reward calculation en quality metrics.
   - Codevoorbeeld: [`reward_functions.py`](reward_functions.py:1)

5. Orchestration / Workflows
   - Stateful workflows via LangGraph; werkflows gedefinieerd in [`orchestration_workflows.py`](orchestration_workflows.py:1) en documentatie.

6. Distributed Compute
   - Ray voor schaalbare training; configuratie in [`ray_distributed_config.py`](ray_distributed_config.py:1)

7. Observability & Monitoring
   - Metrics via InfluxDB / Grafana; dashboards in [`grafana/dashboards/`](grafana/dashboards/:1) en backend in [`monitoring_dashboard.py`](monitoring_dashboard.py:1)

8. Deployment & CI
   - Container images, compose en Helm charts in [`deployments/`](deployments/:1)
   - CI workflows live onder [`.github/workflows/`](.github/workflows/:1)

Dataflow (hoog niveau):
- Dataset (JSONL) -> Server batching -> Agents (LLM calls) -> Trajecten -> Reward calculation -> RL update -> Checkpoint.

Integratiepunten:
- LLM providers: OpenAI, Anthropic (API-keys via `.env`; zie [`.env.example`](.env.example:1))
- Optional: vLLM / VERL runners voor large-scale training.

Falen en mitigaties:
- OOM bij grote LLMs: use smaller test datasets, checkpointing, gradient accumulation.
- Network timeouts: set timeouts and retries.

Best practices & opmerkingen:
- Houd core interfaces klein en testbaar (dependency injection).
- Documenteer 'waarom'-keuzes naast code in Memory Bank.

Referenties:
- Architectuurdocumenten in [`ai-agent-framework-implementation.md`](ai-agent-framework-implementation.md:1) en [`docs/`](docs/:1)

Contact / eigenaren: Agent Lightning team.