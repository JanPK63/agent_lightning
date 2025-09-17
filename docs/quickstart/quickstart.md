# Quickstart / Runbook

Kort overzicht van stappen om de Agent Lightning MVP lokaal te draaien.

Vereisten
- Python 3.10+ geïnstalleerd
- Git
- Optioneel: Docker (voor containerized runs)

1) Maak virtuele omgeving

python -m venv .venv
source .venv/bin/activate

2) Installeer dependencies

pip install -r requirements.txt
pip install -r requirements-dev.txt

3) Pre-commit hooks

pre-commit install
pre-commit run --all-files

4) Run unit tests (smoke)

pytest --maxfail=1 --disable-warnings -q

5) Lint & type checks

black . --check
ruff .
mypy .

6) Start dev-server (optioneel)

uvicorn agent_server:app --reload

7) Memory Bank updates

Gebruik Kilo Code om Memory Bank-bestanden te (her)genereren en refresh:

Zie [`.kilocode/rules/memory-bank/tasks.md`](.kilocode/rules/memory-bank/tasks.md:1) voor commando's en [`.kilocode/rules/memory-bank/context.md`](.kilocode/rules/memory-bank/context.md:1) voor huidige context.

Aanvullende tips

- Houd zware dependency-installs (vLLM, VERL, PyTorch GPU) buiten CI; gebruik optionele requirements-gpu.txt
- Voor debugging: gebruik logs uit `monitoring_dashboard.py` en `influxdb_metrics.py`

Contact / maintainers

Agent Lightning team — zie [`.kilocode/rules/memory-bank/brief.md`](.kilocode/rules/memory-bank/brief.md:1)

Status: MVP quickstart (concept) — pas aan na wijzigingen