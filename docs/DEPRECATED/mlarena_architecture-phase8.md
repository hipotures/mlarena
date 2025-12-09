# Phase 8 – Configuration Mapping

## Zakres

Phase 8 w oryginalnym planie zakładała:
1. Implementację aliasów template'ów dla backward compatibility (`dev-gpu` → `gpu-dev-5m`)
2. Mapowanie starych nazw na nowe w CLI
3. Standaryzację TemplateLoader (już zrealizowana w Phase 2-3)

**Realizacja:** Phase 8 zostaje **POMINIĘTA** jako niepotrzebna. Aliasy nie są wymagane.

## Decyzje / odchylenia

### 1. Brak aliasów template'ów
**Uzasadnienie:**
- System template'ów już używa nowej konwencji nazewnictwa w `config/templates/model.yaml`
- Backwards compatibility przez aliasy nie jest wymagana - brak legacy użytkowników systemu MLArena
- Dodatkowa warstwa abstrakcji (TEMPLATE_ALIASES) wprowadza niepotrzebną złożoność

### 2. Konwencja nazewnictwa template'ów (OSTATECZNA)
**Format:** `{compute}-{variant}-{time}[-{special}]`

**Przykłady:**
- `cpu-fast-1m` - CPU, fast preset, 1 minuta
- `gpu-dev-5m` - GPU, development preset, 5 minut
- `cpu-best-1h` - CPU, best preset, 1 godzina
- `gpu-extreme-24h` - GPU, extreme preset, 24 godziny
- `cpu-best-8h-av` - CPU, best preset, 8h, AutogluonVariant weights

**Nie używamy:**
- Starych nazw: `dev-gpu`, `fast-cpu`, `best-cpu` (deprecated)
- Aliasów w kodzie - tylko jedna nazwa per template

### 3. Problem: Hack `--model-template list`
**Obecny stan:**
```bash
# Brzydki hack - "list" jako wartość parametru
mla model --project X --model-template list
```

**Decyzja:** Hack zostaje TYMCZASOWO, do czasu implementacji meta-commands (zgodnie z `docs/TODO.md`).

**Docelowe rozwiązanie (TODO - osobny task):**
```bash
# Meta-komenda (nie moduł pipeline)
mla templates --project X [--type model|preprocess]
```

### 4. TemplateLoader - już zaimplementowany
System ładowania template'ów (TemplateLoader w `src/mlarena/core/config.py`) działa poprawnie:
- ✅ Merge global + local templates
- ✅ Local overrides global
- ✅ Lista template'ów z oznaczeniem źródła (🅶/🅻/🅶🅻)
- ✅ Zgodny z originalnym planem Phase 8.1

## Testy

Brak nowych testów - Phase 8 pominięta.

Istniejące testy TemplateLoader:
```bash
uv run pytest tests/unit/test_config.py -v
# PASSED - TemplateLoader działa zgodnie z planem
```

## Ryzyka / następne kroki

### Ryzyko 1: Niespójna dokumentacja
**Problem:** Dokumentacja (`CLAUDE.md`, `docs/*.md`) używa starych nazw (`dev-gpu`, `fast-cpu`).

**Plan naprawy (Task 1):**
- Przejść przez wszystkie pliki `.md` i zamienić stare nazwy na nowe
- Pliki do aktualizacji: `CLAUDE.md`, `README.md`, `docs/MLA_WORKFLOW_GUIDE.md`, `docs/MIGRATION_GUIDE.md`, `docs/mlarena_architecture.md`
- Skrypt: `grep -r "dev-gpu\|fast-cpu\|best-gpu" docs/ CLAUDE.md README.md`

### Ryzyko 2: Hack `--model-template list`
**Problem:** `--model-template list` to brzydkie obejście (special value zamiast dedykowanej flagi).

**Plan naprawy (Tasks 2-3 - zgodnie z TODO.md):**
1. Dodać meta-komendę `mla templates --project X [--type model]`
2. Usunąć hack z `src/mlarena/modules/model.py:268-299`
3. Zaktualizować dokumentację o nowe meta-commands

**Priorytet:** ŚREDNI (system działa, ale CLI jest nieintuicyjny)

### Następne kroki (poza Phase 8):

**TASK QUEUE:**
1. ✅ Phase 8 report (ten plik)
2. ✅ Task 1: Zunifikować nazewnictwo w dokumentacji (DONE)
3. ⏭️ Task 2: Dodać meta-komendę `templates` (30 min)
4. ⏭️ Task 3: Usunąć hack `--model-template list` (5 min)
5. ⏭️ Task 4: Dokumentacja meta-commands (10 min)

**Fazy 9-14:** Do przeglądu - większość zaawansowanych features (concurrency, caching) można pominąć jako "nice to have".

## Artefakty / PR

**Commits:**
- Phase 8 report + Task 1 (template naming unification)

**Pliki zmodyfikowane (Task 1 - Template Naming):**
- ✅ `CLAUDE.md` (→ `AGENTS.md`) - Wszystkie przykłady używają nowej konwencji
- ✅ `README.md` - Zaktualizowany przykład quick start
- ✅ `docs/configs.md` - Zaktualizowane przykłady i konwencja nazewnictwa
- ✅ `AGENTS.md` - Dodana sekcja "Template Naming Convention" z pełnym opisem
- ✅ `projects/kaggle/playground-series-s5e12/templates/model.yaml` - Poprawione nazwy (best-1h-* → cpu-best-1h-*)

**Pliki zmodyfikowane (Cleanup - AI Context Files):**
- ✅ `AGENTS.md` - Dodany disclaimer że to AI agent context, nie user docs
- ✅ `README.md` - Dodana sekcja "AI Agent Context" na końcu
- ✅ `docs/README.md` - Usunięty GEMINI.md jako "primary guide", MLA_WORKFLOW_GUIDE.md jest teraz primary
- ✅ `docs/configs.md` - Zamienione linki: CLAUDE.md → MLA_WORKFLOW_GUIDE.md, usunięte deprecated docs
- ✅ `docs/MLA_WORKFLOW_GUIDE.md` - Sekcja "See Also" zaktualizowana (CLAUDE.md → README.md)
- ✅ `docs/mlarena_architecture.md` - TODO: "Update CLAUDE.md" → "Update AGENTS.md"

**Pliki stworzone:**
- `docs/mlarena_architecture-phase8.md` (ten raport)

**Stan systemu po Phase 8:**
- TemplateLoader: ✅ Działa zgodnie z planem
- Konwencja nazewnictwa: ✅ Ustalona (`{compute}-{variant}-{time}`)
- Aliasy: ❌ Pominięte (niepotrzebne)
- Dokumentacja: ⚠️ Wymaga aktualizacji (Task 1)
- CLI meta-commands: ⚠️ TODO (Tasks 2-4)

---

**Status Phase 8:** CLOSED (pominięta jako niepotrzebna, uzgodnienia zapisane)

**Next:** Realizacja Tasks 1-4 lub przegląd Phases 9-14
