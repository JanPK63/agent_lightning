# Memory Bank — tech.md

Platform en taal:
- Python 3.10+ (aanbevolen: 3.10 of 3.11)

Belangrijke libraries:
- LangGraph (stateful workflow orchestration)
- LangChain (LLM application framework)
- agentlightning (project core framework)
- OpenAI / Anthropic / Grok SDKs (LLM providers)
- Ray (distributed compute framework)
- VERL, vLLM (optioneel voor grote-scale LLM training)
- PyTorch (deep learning framework)
- sentence-transformers (embeddings voor knowledge retrieval)
- FastAPI (web framework voor server componenten)
- aiohttp (async HTTP client)
- pydantic (data validation)
- uvicorn (ASGI server)
- httpx (async HTTP client voor OAuth)
- jwt (JSON Web Tokens voor auth)
- bcrypt (password hashing)
- psutil (system monitoring)
- opentelemetry (distributed tracing)
- agentops (agent observability)
- redis (optional voor pubsub messaging)

Dev-dependencies:
- pytest
- black
- isort
- ruff (of flake8)
- mypy
- pre-commit
- docker, docker-compose (voor containerized dev)

Aanbevolen versies / pinning:
- Python >=3.10, <3.13
- torch==2.7.0
- vllm==0.9.2 (optioneel)
- verl==0.5.0 (optioneel)
- Pin heavy GPU-libs in een optionele requirements file (bijv. requirements-gpu.txt)

Belangrijke environment variabelen:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- RAY_ADDRESS (indien Ray-cluster gebruikt)
- INFLUXDB_URL, INFLUXDB_TOKEN (monitoring)
- REDIS_URL (indien Redis cache/event-bus gebruikt)

Lokale setup (kort):
1. python -m venv .venv
2. source .venv/bin/activate
3. pip install -r requirements.txt
4. pip install -r requirements-dev.txt
5. pre-commit install

Commands voor ontwikkeling en tests:
- pre-commit run --all-files
- pytest --maxfail=1 --disable-warnings -q
- mypy .
- black . --check

CI-aanbevelingen:
- Gebruik matrix builds voor Python-versies (3.10, 3.11)
- Stap 1: installeer dev-deps
- Stap 2: run linters (ruff/flake8), formatter checks (black/isort)
- Stap 3: run mypy
- Stap 4: run pytest met coverage (gebruik kleine, mock datasets)
- Houd GPU- en lange trainingsjobs buiten CI; gebruik smoke-tests of mocks

Zware afhankelijkheden & GPU:
- Installeer vLLM/VERL/PyTorch GPU-varianten via dedicated script (scripts/setup_stable_gpu.sh)
- Instructies omgevingsspecifiek, documenteer in README en in `.env.example`

Tips voor versiebeheer en reproducibility:
- Pin kritieke dependency-sets in requirements files
- Gebruik Dockerfile / docker-compose voor consistente dev-omgeving
- Voeg reproducible example datasets in `examples/` (klein formaat voor CI)

Contact / maintainers:
- Agent Lightning team (zie README en Memory Bank voor eigenaren)