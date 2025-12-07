# MLArena Test Suite

- `unit/` – izolowane testy modułów; szybkie (<30s)
- `integration/` – przepływy wielomodułowe CLI
- `e2e/` – pełne pipeline na prawdziwych danych (domyślnie pomijane)

Uruchomienia:

```bash
# Szybka pętla
uv run pytest -m "unit or integration" --maxfail=3

# Cały zestaw bez e2e
uv run pytest -m "not e2e"

# E2E (wymaga MLA_E2E=1 i danych Titianica)
MLA_E2E=1 uv run pytest -m e2e -v
```
