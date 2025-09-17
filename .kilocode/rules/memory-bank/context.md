# Memory Bank — context.md

Overzicht van de huidige werkfocus en recente wijzigingen voor het Agent Lightning project.

Doel
Het doel van dit bestand is om korte, actuele context te geven over wat er nu aan de codebase en documentatie wordt gewerkt.

Huidige focus
- Initialiseren en verversen van de Memory Bank-bestanden (brief, product, architecture, tech, context).
- Stabiliseren van de MVP agent scaffold en testinfrastructuur.
- Verbeteren van developer experience: quickstart, pre-commit, linting en CI-smoke-tests.

Recent voltooide items
- Virtuele omgeving en dependencies: zie [`tasks.md`](.kilocode/rules/memory-bank/tasks.md:1)
- pre-commit hooks: zie [`.pre-commit-config.yaml`](.pre-commit-config.yaml:1)

Openstaande / volgende stappen
- Voltooi `update-memory-bank`: genereer en review alle files onder [`.kilocode/rules/memory-bank/`](.kilocode/rules/memory-bank/:1)
- Run unit tests en los failing tests op (pytest)
- Run linters (black / ruff / mypy) en automatiseer in CI
- Schrijf korte runbook / quickstart in [`README.md`](README.md:1) of [`docs/quickstart/`](docs/quickstart/:1)

Belangrijke aannames
- Development omgeving draait op Python 3.10+ (zie [`tech.md`](.kilocode/rules/memory-bank/tech.md:1))
- API keys en secrets staan in environment variables (zie [`.env.example`](.env.example:1))

Links
- Projectbrief: [`brief.md`](.kilocode/rules/memory-bank/brief.md:1)
- Productvisie: [`product.md`](.kilocode/rules/memory-bank/product.md:1)
- Architectuur: [`architecture.md`](.kilocode/rules/memory-bank/architecture.md:1)
- Technische details: [`tech.md`](.kilocode/rules/memory-bank/tech.md:1)
- Takenlijst: [`tasks.md`](.kilocode/rules/memory-bank/tasks.md:1)

Eigenaar
Agent Lightning team / Maintainers

Status
[Memory Bank: Active] — update in progress -> context.md gegenereerd

Laatst bijgewerkt: 2025-09-10T15:55:10.110Z