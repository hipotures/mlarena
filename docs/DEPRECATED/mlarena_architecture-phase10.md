# Phase 10 – Backward Compatibility & Rollout

## Zakres

Phase 10 w oryginalnym planie zakładała:
1. CLI aliases (deprecated wrappers dla starych skryptów)
2. Deprecation warnings przez 1 sprint (2 tygodnie)
3. Lock na stare entry points po migracji
4. Rollout timeline (3 tygodnie: deploy → test → cleanup)

**Realizacja:** Phase 10 została **POMINIĘTA JAKO ZBĘDNA** - brak legacy users, nie ma kogo migrować.

---

## Decyzje / odchylenia

### 1. ❌ CLI Aliases - NIEPOTRZEBNE

**Planned (oryginalny plan):**
```bash
# scripts/experiment_manager.py (deprecated wrapper)
#!/usr/bin/env python
"""DEPRECATED: Use mla.py instead. This wrapper will be removed in 2 weeks."""
warnings.warn("Use mla.py instead", DeprecationWarning)

# Forward to mla.py
from mlarena.cli.main import main
main()
```

**Decyzja:** POMINIĘTE całkowicie.

**Uzasadnienie:**
1. **Brak legacy users** - MLArena to nowy system, nikt nie używał starych skryptów w produkcji
2. **Nie ma co migrować** - jedyny użytkownik (developer) już korzysta z `mla.py`
3. **Stare skrypty nieusunięte** - `experiment_manager.py`, `ml_runner.py` wciąż istnieją, ale jako standalone tools (nie deprecated)

**Obecny stan:**
```bash
# Stare skrypty istnieją, ale nie są częścią MLArena pipeline
ls scripts/experiment_manager.py  # EXISTS (legacy tool, nie deprecated)
ls scripts/ml_runner.py           # EXISTS (legacy tool, nie deprecated)
ls scripts/mla.py                 # NEW (primary entry point)
```

**Policy:** Stare skrypty pozostają jako "legacy tools" dla ad-hoc użycia, ale nie są oficjalnie wspierane ani dokumentowane.

---

### 2. ❌ Deprecation Warnings - NIEPOTRZEBNE

**Planned:**
- Week 1-2: Deploy mla.py + wrappers, show warnings
- Week 3: Lock old scripts

**Decyzja:** POMINIĘTE - brak użytkowników do ostrzegania.

**Obecny rollout:**
- ✅ Week 1 (Dec 5-11): MLArena deployed, `mla.py` działa
- ✅ Week 2 (Dec 12-18): Testing w produkcji (s5e12)
- ❌ Week 3: Lock old scripts → SKIPPED (nie usuwamy legacy tools)

---

### 3. ❌ Lock on Old Entry Point - NIEPOTRZEBNE

**Planned (Phase 10.2):**
```bash
# scripts/experiment_manager.py (locked)
#!/usr/bin/env python
"""REMOVED: Use mla.py instead."""
print("ERROR: experiment_manager.py has been removed.")
sys.exit(1)
```

**Decyzja:** NIE LOCKUJEMY starych skryptów.

**Uzasadnienie:**
- `experiment_manager.py` może być przydatny jako standalone tool
- Brak ryzyka confusion (dokumentacja mówi o `mla.py`)
- Jeśli ktoś uruchomi stary skrypt, po prostu nie będzie używał MLArena (to OK)

**Obecna policy:**
```
scripts/mla.py              → OFFICIAL (dokumentowany, wspierany)
scripts/experiment_manager  → LEGACY (działa, ale unsupported)
scripts/ml_runner.py        → LEGACY (działa, ale unsupported)
```

---

### 4. ❌ Rollout Timeline - NIEPOTRZEBNY

**Planned:**
```
| Week | Action | Status |
|------|--------|--------|
| W1 (Dec 5-11) | Deploy mla.py + wrappers | Both systems work |
| W2 (Dec 12-18) | Team tests mla.py | Deprecation warnings |
| W3 (Dec 19+) | Remove old scripts | Only mla.py |
```

**Actual rollout:**
```
| Week | Action | Status |
|------|--------|--------|
| W1 (Dec 5-11) | Deploy mla.py | ✅ Works |
| W2 (Dec 12-18) | Production testing (s5e12) | ✅ Stable |
| W3 (Dec 19+) | No action needed | Both systems coexist |
```

**Decyzja:** Nie ma timeline'u do wykonania - migration completed instantly (brak legacy users).

---

## Testy

**Testy Phase 10:** BRAK (phase pominięta)

**Obecne pokrycie:**
- MLArena: ✅ 85% core, 72% modules
- Legacy scripts: ⚠️ No tests (unsupported)

---

## Ryzyka / następne kroki

### Ryzyko 1: Confusion między mla.py a experiment_manager.py
**Problem:** Developer może przypadkowo uruchomić stary skrypt zamiast `mla.py`.

**Mitigation:**
1. Dokumentacja (`CLAUDE.md`, `README.md`) mówi tylko o `mla.py`
2. Stare skrypty nie są w PATH (trzeba wpisać pełną ścieżkę)
3. `.gitignore` nie ignoruje starych skryptów (można je zobaczyć)

**Priorytet:** NISKI

---

### Ryzyko 2: Maintenance burden starych skryptów
**Problem:** Jeśli legacy scripts wciąż działają, ktoś może je poprawiać/modyfikować.

**Mitigation:**
1. Dodać NOTICE na górze każdego legacy script: `# NOTE: This is a legacy tool. Use mla.py instead.`
2. NIE fixować bugów w legacy tools (redirect do `mla.py`)
3. Jeśli ktoś używa legacy tool w nowym projekcie → code review reject

**Priorytet:** ŚREDNI (profilaktyka)

---

### Następne kroki

**DONE Phase 10:** POMINIĘTE (niepotrzebne)

**TODO (opcjonalne - cleanup):**
1. Dodać `# LEGACY TOOL - Use mla.py instead` na górze starych skryptów
2. Zaktualizować `scripts/README.md` - oznaczyć stare skrypty jako "legacy"
3. Usunąć stare skrypty z examples w dokumentacji

**Next:** Phase 11 - Concurrency & Recovery

---

## Artefakty / PR

**Commits:** BRAK (phase pominięta)

**Pliki do opcjonalnego cleanup (TODO - low priority):**
```bash
# Dodać LEGACY notice:
scripts/experiment_manager.py  # Line 1: # LEGACY TOOL ...
scripts/ml_runner.py           # Line 1: # LEGACY TOOL ...
scripts/autogluon_runner.py    # Line 1: # LEGACY TOOL ...

# Zaktualizować docs:
scripts/README.md              # Sekcja "Legacy Tools"
```

**Stan systemu po Phase 10:**
- CLI aliases: ❌ Pominięte
- Deprecation warnings: ❌ Pominięte
- Lock old scripts: ❌ Pominięte
- Rollout timeline: ✅ Zakończony (instant migration)

---

**Status Phase 10:** POMINIĘTA JAKO ZBĘDNA

**Reasoning:** Brak legacy users = brak potrzeby backward compatibility. MLArena to greenfield deployment, nie migration z legacy system.

**Lesson learned:** Planowanie backward compatibility w oryginalnym Phase 10 było przedwczesne - należało najpierw sprawdzić, czy są legacy users.

**Next:** Phase 11 - Concurrency & Recovery
