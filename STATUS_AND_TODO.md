# Agent Lightning — Python Implementation Status and TODO

Last updated: 2025-09-16

## Scope
This document summarizes the current status of the Python implementation (excluding Docker) and provides a prioritized TODO list. It incorporates known non-working areas: Spec-driven development, Agent knowledge, and Visual code builder.

---

## Current Status

### Core Runtime (MVP-ready)
- Server: [`agentlightning/server.py`](agentlightning/server.py:1)
  - FastAPI in-memory server with:
    - Task queue (async, thread-safe) and timeout-based stale requeue
    - Versioned resources (NamedResources) with latest retrieval
    - Rollout storage and retrieval
  - Endpoints: GET /task, GET /resources/latest, GET /resources/{id}, POST /rollout
  - Gaps: No persistence, auth, metrics, or rate limiting

- Client + Dev Task Loader: [`agentlightning/client.py`](agentlightning/client.py:1)
  - Sync/async polling; cached resource fetch; post rollouts
  - DevTaskLoader for in-process tasks/resources without server; captures rollouts

- Agent Abstraction: [`agentlightning/litagent.py`](agentlightning/litagent.py:1)
  - Abstract LitAgent (sync/async rollout) with lifecycle hooks; weakrefs to Runner/Trainer

- Runner: [`agentlightning/runner.py`](agentlightning/runner.py:1)
  - Loop: poll task → fetch resources → execute agent rollout → trace → export Triplets → post Rollout
### Task Assignment, Memory, and Knowledge

- Task Assignment
  - Implementation:
    - Core queue and assignment via server GET /task and stale requeue: [`agentlightning/server.py`](agentlightning/server.py:229)
    - Client polling and posting: [`agentlightning/client.py`](agentlightning/client.py:188)
    - Runner consumption and lifecycle: [`agentlightning/runner.py`](agentlightning/runner.py:137)
    - Higher-level coordination scaffolds (not wired): [`services/agent_coordinator_service.py`](services/agent_coordinator_service.py:6), [`services/agent_coordination_glue.py`](services/agent_coordination_glue.py:2)
  - Status: Core assignment is MVP-ready. Advanced coordination services require DAL/Redis integration and wiring to core Server/Runner.
  - Gaps:
    - No persistence of queue/processing state
    - No priority/fairness scheduling or backoff/jitter strategies
    - No auth/rate limiting on assignment endpoints

- Memory Function
  - Implementation:
    - RL-oriented memory via spans → Triplets: [`agentlightning/tracer/triplet.py`](agentlightning/tracer/triplet.py:501)
    - Dedicated services (scaffolds): [`services/memory_service.py`](services/memory_service.py:1), [`services/memory_retrieval.py`](services/memory_retrieval.py:6), [`services/memory_consolidation.py`](services/memory_consolidation.py:6)
  - Status: Triplet-based extraction is functional; full memory service requires DAL/cache setup and agent integration.
  - Gaps:
    - No standard client API for agents to store/retrieve episodic/semantic memory
    - Missing embeddings config and consolidation policy enforcement
    - Lack of tests and operational runbook

- Knowledge Function
  - Implementation:
    - Full CRUD + embeddings + retrieval service: [`services/knowledge_manager_service.py`](services/knowledge_manager_service.py:6)
    - Python client for agents: [`agentlightning/knowledge_client.py`](agentlightning/knowledge_client.py:1)
    - Integration with Runner/Agents for task context: [`agentlightning/runner.py`](agentlightning/runner.py:1)
    - DAL/cache integration with SQLite/PostgreSQL compatibility
  - Status: Fully operational end-to-end; knowledge ingestion/retrieval wired across services and agent runtime.
  - Features:
    - Complete CRUD operations with embeddings and semantic search
    - Python client used by Runner/Agents to fetch knowledge context per task
    - Comprehensive schema/migrations and tests
    - Event-driven architecture with proper event channels
  - Sync + async paths, integrates with tracer (AgentOps/OpenTelemetry)
  - Needs backoff/jitter, stronger error modes

- Tracing Foundation:
  - Base: [`agentlightning/tracer/base.py`](agentlightning/tracer/base.py:1)
  - Triplet Exporter: [`agentlightning/tracer/triplet.py`](agentlightning/tracer/triplet.py:1)
    - Converts OpenTelemetry spans to Triplets; supports reward matching policies and hierarchy repair
  - Concrete tracer implied: [`agentlightning/tracer/__init__.py`](agentlightning/tracer/__init__.py:1) re-exports AgentOpsTracer reference

- Types and Resources: [`agentlightning/types.py`](agentlightning/types.py:1)
  - Pydantic models for Task, Rollout, Triplet, ResourcesUpdate
  - Discriminated ResourceUnion (LLM, PromptTemplate)
  - ParallelWorkerBase lifecycle

Status: Core loop and data models are functional for local MVP scenarios.

---

### LangGraph Integration (Scaffold runs; simulated nodes)
- Service: [`services/langgraph_integration.py`](services/langgraph_integration.py:1)
  - Health, workflow CRUD, execution, results, checkpoints
  - Uses real StateGraph/MemorySaver when available; otherwise mocked
  - Nodes simulate outputs; not yet invoking real agent/tools
Status: Runnable scaffold; ready to wire nodes to actual operations (agents/tools).

---

### Non-working Focus Areas

1) Spec-driven development
- Likely dirs: [`spec_driven_workflow/`](spec_driven_workflow/:1), [`specs/`](specs/:1)
- Missing:
  - FastAPI service to accept/validate specs and translate to executable flow
  - Translator from spec → LangGraph WorkflowDefinition or queued tasks/resources on Server
  - Persistence and end-to-end tests
Status: Not wired, incomplete.

2) Agent knowledge
- References: [`agent_code_integration.py`](agent_code_integration.py:1), [`langchain_agent_wrapper.py`](langchain_agent_wrapper.py:1), minimal stub [`services/knowledge_manager_service.py`](services/knowledge_manager_service.py:6)
- Missing:
  - Full CRUD/embedding/retrieval service with DAL/cache and consistent schema
  - Integration point for agents/runner to fetch context per task
  - Tests and verified DB models
Status: Not operational; scattered references, incomplete service.

3) Visual code builder
- File: [`services/visual_builder_service_integrated.py`](services/visual_builder_service_integrated.py:8)
  - Large scaffold (graph, components, code generation, deploy, debugging)
  - Likely runtime issues: static assets, DAL wiring, incomplete CodeTranslator paths
  - Not connected to core Server/Runner or LangGraph
Status: Scaffold present; runtime likely broken; no minimal happy path validated.

---

### Other Services (Selective highlights)
- Memory/Performance/RL/Git/Security/CICD/Integration/WebSocket: multiple FastAPI services exist as scaffolds with DAL/cache patterns (e.g., [`services/memory_service.py`](services/memory_service.py:1), [`services/performance_metrics_service.py`](services/performance_metrics_service.py:1), [`services/rl_orchestrator_service.py`](services/rl_orchestrator_service.py:1)). These are not tightly integrated into the core agent loop. Maturity varies; many require environment setup (DB/Redis/Influx) and further wiring.

---

## Risks and Gaps
- Persistence: Core server is in-memory only
- Security: No auth/ACL on core endpoints
- Observability: Logging only; no metrics/health endpoints in core server
- Resilience: Runner lacks backoff/jitter; errors always followed by post_rollout
- Tracing: Concrete tracer wiring not validated; need no-op fallback
- E2E Examples: No canonical, tested example agent + workflow path
- Tests: Limited coverage across server, runner, TripletExporter, and services

---

## Recommended Roadmap (Prioritized TODO)

1) Minimal E2E demo (baseline) ✅ COMPLETED
- Implement a simple LitAgent implementing training_rollout
- Wire DevTaskLoader + AgentRunner to process a small task batch
- Provide a NoOpTracer if AgentOpsTracer is not available
- Add a script/example to run end-to-end and assert a rollout received
Owner: Core
ETA: Short

2) Spec-driven development MVP ✅ COMPLETED
- Add Spec Service (FastAPI) with endpoints:
  - POST /specs: validate + persist
  - POST /specs/{id}/plan: translate to LangGraph WorkflowDefinition or task list
  - POST /specs/{id}/execute: execute via [`services/langgraph_integration.py`](services/langgraph_integration.py:1) or queue tasks to Server
- Implement schema and translator module
- Add tests: spec validation, plan generation, E2E execution using DevTaskLoader
Owner: Platform
ETA: Medium

3) Agent knowledge service enablement ✅ COMPLETED
- Expand [`services/knowledge_manager_service.py`](services/knowledge_manager_service.py:6) to full CRUD + embed + query
- Integrate with DAL/cache, choose embedding model, add ingestion pipeline
- Provide a Python client and integrate with Runner/Agents to fetch context per task
- Add tests for ingestion and retrieval
Owner: Knowledge
ETA: Medium

4) Visual code builder minimal viable path ✅ COMPLETED
- Stabilize headless JSON API: projects, components, connections, generate-code, download package
- Implement one reliable code generation path (FastAPI scaffold) with tests
- Defer UI static assets; document headless usage
- Optional: export project to LangGraph WorkflowDefinition
Owner: Builder
ETA: Medium

5) Core hardening
- Persistence layer for tasks/resources/rollouts (DAL-backed)
- Auth (e.g., API keys/JWT) and rate limiting on core server
- Observability: metrics (Prometheus) + /health, structured logs
- Runner resilience: backoff, retries, failure reporting, max attempts
- Tracer: verify AgentOpsTracer; provide NoOpTracer fallback; sample config
Owner: Core
ETA: Medium

6) Testing and CI
- Unit tests: server queue/resources, runner happy/error paths, triplet extraction
- Integration tests: Spec → plan → execute; Knowledge ingest/search; Builder code generation
- Smoke tests using DevTaskLoader
Owner: QA
ETA: Ongoing

---

## Execution Checklist

- [x] Create NoOpTracer and sample LitAgent example
- [x] E2E demo script using DevTaskLoader and Runner
- [x] Minimal E2E demo (baseline) completed
- [ ] Spec Service: define spec schema and validation
- [ ] Spec translator: spec → LangGraph WorkflowDefinition or task batch
- [ ] Spec Service tests (unit/integration)
- [x] Knowledge Service: CRUD, embedding, retrieval, DAL integration
- [x] Knowledge Python client; integrate into agents/runner
- [x] Knowledge tests (ingest/query)
- [x] Visual Builder API: minimal endpoints and single code generation path
- [x] Visual Builder code generation tests
- [x] Core server persistence layer
- [ ] Core server auth and rate limiting
- [ ] Core server metrics and health
- [ ] Runner backoff/retry/failure modes
- [ ] Tracer verification and fallback implementation
- [ ] CI: add smoke/integration tests across above features

---

## Notes and References
- Core server: [`agentlightning/server.py`](agentlightning/server.py:1)
- Client/Dev loader: [`agentlightning/client.py`](agentlightning/client.py:1)
- Runner: [`agentlightning/runner.py`](agentlightning/runner.py:1)
- Agent API: [`agentlightning/litagent.py`](agentlightning/litagent.py:1)
- Tracing: [`agentlightning/tracer/base.py`](agentlightning/tracer/base.py:1), [`agentlightning/tracer/triplet.py`](agentlightning/tracer/triplet.py:1)
- LangGraph Service: [`services/langgraph_integration.py`](services/langgraph_integration.py:1)
- Knowledge Service (stub): [`services/knowledge_manager_service.py`](services/knowledge_manager_service.py:6)
- Visual Builder (scaffold): [`services/visual_builder_service_integrated.py`](services/visual_builder_service_integrated.py:8)