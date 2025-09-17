# Memory Bank — tasks.md

Documentatie van repetitieve taken en scripts voor Agent Lightning.

## Doel
Een centrale plek om vaak voorkomende taken en commando's te beschrijven zodat teamleden snel kunnen reproduceren.

## Taken
- setup: Maak virtuele omgeving en installeer dependencies
  - Command: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
- run-tests: Voer unit tests uit
  - Command: pytest --maxfail=1 --disable-warnings -q
- lint: Run linters en formatter checks
  - Commands:
    - black . --check
    - ruff .
    - mypy .
- pre-commit: Installeer pre-commit hooks
  - Command: pre-commit install
- update-memory-bank: Herbouw of update Memory Bank bestanden
  - Beschrijving: Gebruik Kilo Code om bestanden onder `.kilocode/rules/memory-bank/` te genereren of te verversen na grote wijzigingen.
- start-dev-server: Start lokale ontwikkelserver (indien van toepassing)
  - Voorbeeld: uvicorn agent_server:app --reload
- build-examples: Maak en test voorbeeld-workflows in `examples/`
  - Beschrijving: Zorg dat voorbeelden klein en reproduceerbaar zijn voor CI
- ci: Locally run CI checks
  - Beschrijving: Run linters, mypy, en pytest zoals in CI workflow

## Aanbevelingen
- Documenteer nieuwe repetitieve taken direct in deze file.
- Houd commando's idempotent en reproduceerbaar.

Last updated: 2025-09-10T15:51:00Z