# Memory Bank — architecture.md

Overzicht van systeemarchitectuur en belangrijkste componenten.

Architectuurcomponenten:

ze1. Agent Lightning Core Framework (`agentlightning/`)
   - Stateful AI agent framework met ReAct patroon
   - Kerncomponenten: [`litagent.py`](agentlightning/litagent.py:1), [`client.py`](agentlightning/client.py:1), [`server.py`](agentlightning/server.py:1), [`trainer.py`](agentlightning/trainer.py:1), [`runner.py`](agentlightning/runner.py:1)

2. Agent Communicatie & Orchestratie
   - Inter-agent communicatie: [`communication.py`](agentlightning/communication.py:1)
   - Saga transacties: [`saga_client.py`](agentlightning/saga_client.py:1)
   - Workflow orchestratie via LangGraph integratie

3. Authentication & Authorization
   - OAuth2/OIDC: [`oauth.py`](agentlightning/oauth.py:1), [`oidc_providers.py`](agentlightning/oidc_providers.py:1)
   - RBAC systeem: [`rbac.py`](agentlightning/rbac.py:1)
   - API key management: [`auth.py`](agentlightning/auth.py:1)

4. LLM Providers & Integraties
   - Multi-provider LLM support: [`llm_providers.py`](agentlightning/llm_providers.py:1)
   - OpenAI, Anthropic, Grok integratie
   - Multi-modal content support: [`types.py`](agentlightning/types.py:1)

5. Evaluation & Metrics
   - Uitgebreide evaluatie metrics: [`evaluation_metrics.py`](agentlightning/evaluation_metrics.py:1)
   - Model versioning: [`model_versioning.py`](agentlightning/model_versioning.py:1)
   - A/B testing framework: [`agent_ab_testing.py`](agentlightning/agent_ab_testing.py:1)

6. Meta-Learning & Adaptatie
   - Dynamische prompt engineering: [`prompt_engineering.py`](agentlightning/prompt_engineering.py:1)
   - Meta-learning capabilities: [`meta_learning.py`](agentlightning/meta_learning.py:1)
   - Context-aware agent adaptatie

7. Distributed Training
   - Ray/PyTorch DDP support: [`distributed_training.py`](agentlightning/distributed_training.py:1)
   - Schaalbare training infrastructuur

8. Knowledge Management
   - Knowledge client: [`knowledge_client.py`](agentlightning/knowledge_client.py:1)
   - Context retrieval voor tasks

9. Observability & Monitoring
   - Tracing: `tracer/` directory
   - Instrumentation: `instrumentation/` directory
   - Metrics en logging: [`logging.py`](agentlightning/logging.py:1), [`reward.py`](agentlightning/reward.py:1)

10. Configuration & CLI
    - Config management: [`config.py`](agentlightning/config.py:1)
    - CLI tools: `cli/` directory

Dataflow (hoog niveau):
- Dataset (JSONL) -> Server batching -> Agents (LLM calls) -> Trajecten -> Reward calculation -> RL update -> Checkpoint.

## Kritieke Problemen Geïdentificeerd

### Agent Discovery Issues
**Probleem**: Agents worden gedefinieerd in `agent_capability_matcher.py` maar de daadwerkelijke agent services draaien niet.
- Health URLs (localhost:9001-9006) retourneren "unreachable"
- Geen automatische service discovery of registratie
- RL Orchestrator verwacht externe agent services maar deze bestaan niet

**Impact**: Agent discovery tests falen, agents worden als "unreachable" gemarkeerd.

### Task Execution Issues
**Probleem**: Tasks worden assigned maar executie faalt omdat agents niet beschikbaar zijn.
- `TaskExecutionBridge` valt terug op mock responses
- Geen echte agent services voor task execution
- Background task scheduling werkt niet betrouwbaar

**Impact**: Tasks blijven in "assigned" status, geen automatische executie.

### Architecture Gaps
**Ontbrekende Componenten**:
- Agent service orchestration layer
- Automatische agent startup/discovery
- Health monitoring en failover
- Service mesh voor agent communicatie

**Huidige Workarounds**:
- Mock agents voor testing (`services/mock_agent.py`)
- Handmatige agent registratie via `/agents/register`
- Fallback naar mock execution in `agent_executor_fix.py`

Integratiepunten:
- LLM providers: OpenAI, Anthropic (API-keys via `.env`; zie [`.env.example`](.env.example:1))
- Optional: vLLM / VERL runners voor large-scale training.

Falen en mitigaties:
- OOM bij grote LLMs: use smaller test datasets, checkpointing, gradient accumulation.
- Network timeouts: set timeouts and retries.
- **NIEUW**: Agent discovery failures: implement service discovery, health monitoring.
- **NIEUW**: Task execution failures: implement agent service orchestration, fallback mechanisms.

Best practices & opmerkingen:
- Houd core interfaces klein en testbaar (dependency injection).
- Documenteer 'waarom'-keuzes naast code in Memory Bank.

Referenties:
- Architectuurdocumenten in [`ai-agent-framework-implementation.md`](ai-agent-framework-implementation.md:1) en [`docs/`](docs/:1)

Contact / eigenaren: Agent Lightning team.