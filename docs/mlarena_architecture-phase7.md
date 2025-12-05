# Phase 7 – Init Command Restoration

- **Zakres**: Przywrócono funkcję inicjalizacji projektu w MLArena jako komendę `mla init` (pełny odpowiednik legacy `init-project`): tworzenie struktury katalogów, kopiowanie szablonów kodu/konfigów, opcjonalny download danych z Kaggle.
- **Decyzje / odchylenia**: Komenda jest obsługiwana bez tworzenia eksperymentu; renderujemy `code/utils/config.py` z bezpiecznymi wartościami startowymi (`TARGET_COLUMN="target"`, `COMPETITION_NAME=<slug>`). README szablon renderowany tylko po kluczach, reszta placeholderów zostaje do uzupełnienia ręcznie.
- **Testy**: Ręcznie zweryfikowane (full run z pobraniem danych); scenariusz referencyjny: `uv run python scripts/mla.py init --project demo-init --force`.
- **Ryzyka / następne kroki**: Wymagane `kaggle` CLI; przy ponownym uruchomieniu bez `--force` istniejące pliki są chronione, ale nowe repozytoria mogą potrzebować dodatkowych presetów w `configs/` jeśli dodamy je w przyszłości.
- **Artefakty / PR**: `src/mlarena/cli/main.py`, `src/mlarena/utils/init_project.py`, `docs/mlarena_architecture.md`, `docs/mlarena_architecture-phase7.md`.
