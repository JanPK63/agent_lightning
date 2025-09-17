# Memory Bank — product.md

Productvisie: Agent Lightning maakt het eenvoudig voor onderzoekers en engineers om AI-agents te trainen, te optimaliseren en te integreren in workflows met minimale codewijzigingen. Het product faciliteert selectieve optimalisatie, experimenteerbaarheid met RL-algoritmes en ondersteuning voor meerdere agent-frameworks.

Belangrijkste gebruikersproblemen:
- Moeite met het herhaalbaar trainen van agent-workflows zonder grote codewijzigingen.
- Gebrek aan gestandaardiseerde tooling voor reward-design en evaluatie.
- Moeilijkheden bij het integreren van meerdere LLMs en het behouden van context/state over lange interacties.

Belangrijkste functies (MVP):
- Stateful agent scaffold met eenvoudige interfaces (agent_client, agent_server)
- Memory & context management (episodic + semantic memory)
- Basis reward calculator en dataset-format (JSONL)
- Voorbeeld-workflow en demo in `examples/agent_demo/`
- Test-suite (unit + integratie) en CI workflow

Succescriteria:
- Nieuwe gebruiker kan de MVP lokaal draaien met de aangegeven quickstart (venv, deps installeren) in < 30 minuten.
- Unit tests draaien lokaal en in CI met minimaal 70% dekking voor kritieke modules.
- Basis training pipeline draait en produceert traceable metrics (logs & monitoring).

Doelgebruikers:
- ML-onderzoekers die RL op LLM-agents willen toepassen
- Engineering teams die agent-workflows willen integreren in productomgeving
- Academische gebruikers voor reproducible experiments

Release- en iteratiestrook:
- MVP: Memory Bank + minimal agent scaffold + tests (Q1)
- Iteratie 2: LangGraph-integratie en AutoGen support (Q2)
- Iteratie 3: VERL/vLLM optimalisaties en cloud deployment (Q3)

Status: concept — te verifiëren met het team en stakeholders.