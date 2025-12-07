# Phase 12 – Enhanced Caching Strategy

## Zakres

Phase 12 w oryginalnym planie zakładała:
1. Composite cache key (config + git hash + data checksum)
2. Cache metadata (created_at, git_hash, artifacts info)
3. Cache invalidation (git/data change detection)
4. Fast data checksum (sample-based, not full file hash)

**Realizacja:** Phase 12 została **POMINIĘTA JAKO ZBĘDNA** - obecny file-based caching wystarczy.

---

## Decyzje / odchylenia

### 1. ❌ Composite Cache Key - POMINIĘTE

**Planned (oryginalny plan):**
```python
def compute_cache_key(
    config: Dict[str, Any],
    git_hash: str,
    data_checksum: Optional[str] = None
) -> str:
    """
    Key invalidated when:
    - Config changes (hyperparameters, template)
    - Code changes (git hash)
    - Data changes (checksum)
    """
    components = {
        "config": config,
        "git_hash": git_hash[:8],
        "data_checksum": data_checksum or "none",
    }
    return hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()[:16]
```

**Decyzja:** POMINIĘTE całkowicie.

**Uzasadnienie:**
1. **Experiment ID jest już cache key** - każdy experiment ma unikalny ID, nie potrzebujemy hashowania
2. **Git hash nie powinien invalidować cache** - jeśli użytkownik chce użyć cache z poprzedniego commita, to jego wybór
3. **Data checksum to overhead** - dane zmieniają się rzadko, manual invalidation (`--force`) wystarczy
4. **Composite key komplikuje debugging** - hash-based key jest nieczytelny, `exp-20251205-143000` jest self-documenting

**Obecny mechanizm:**
```python
# PreprocessModule zapisuje cache w experiment dir:
cache_dir = self.context.experiment_dir / "preprocess_cache"
cache_dir.mkdir(exist_ok=True)

train_fe_path = cache_dir / "train_fe.parquet"
test_fe_path = cache_dir / "test_fe.parquet"

train_fe.to_parquet(train_fe_path)
test_fe.to_parquet(test_fe_path)

# ModelModule czyta cache:
if train_fe_path.exists() and not force:
    train = pd.read_parquet(train_fe_path)
    test = pd.read_parquet(test_fe_path)
```

**Cache key:** `experiments/<experiment_id>/preprocess_cache/` - simple i czytelny.

**Invalidation:** Manual (`--force` flag) lub nowy experiment.

---

### 2. ❌ Cache Metadata - POMINIĘTE

**Planned:**
```json
{
  "cache_key": "a1b2c3d4e5f6g7h8",
  "created_at": "2025-12-05T14:30:00Z",
  "git_hash": "abc123de",
  "config": {...},
  "data_checksums": {
    "train": "checksum123",
    "test": "checksum456"
  },
  "artifacts": {
    "train_fe.parquet": {"size_mb": 12.5, "rows": 100000, "cols": 50},
    "test_fe.parquet": {"size_mb": 6.2, "rows": 50000, "cols": 49}
  }
}
```

**Decyzja:** POMINIĘTE - state.json już zawiera metadata.

**Uzasadnienie:**
1. **state.json jest już metadata store** - zawiera git hash, config, timestampy
2. **Artifact info jest redundant** - można sprawdzić `ls -lh` lub `pd.read_parquet().shape`
3. **Dodatkowy plik to overhead** - jeszcze jeden plik do synchronizacji z state.json

**Obecna metadata:**
```json
// experiments/<exp_id>/state.json
{
  "experiment_id": "exp-20251205-143000",
  "git": {
    "hash": "abc123de",
    "branch": "feature/template-redesign"
  },
  "modules": {
    "preprocess": {
      "status": "completed",
      "invocation": {
        "preprocess_template": "identity"
      },
      "payload": {
        "train_fe_path": "preprocess_cache/train_fe.parquet",
        "test_fe_path": "preprocess_cache/test_fe.parquet",
        "shape": {"train": [100000, 50], "test": [50000, 49]}
      }
    }
  }
}
```

**Wystarczające metadata:**
- ✅ Git hash
- ✅ Config (invocation params)
- ✅ Artifact paths
- ✅ Shape info (w payload)

---

### 3. ❌ Cache Invalidation Logic - POMINIĘTE

**Planned:**
```python
def get_cached_features(self, cache_key: str) -> Optional[CachedFeatures]:
    # Validate git hash matches
    current_git = GitSnapshot.get_info(REPO_ROOT)["hash"][:8]
    if meta["git_hash"] != current_git:
        console.print("[yellow]Cache invalidated: git hash changed[/yellow]")
        return None

    # Validate data checksums match
    current_train_checksum = compute_data_checksum(self.train_path)
    if meta["data_checksums"]["train"] != current_train_checksum:
        console.print("[yellow]Cache invalidated: train data changed[/yellow]")
        return None
```

**Decyzja:** POMINIĘTE - manual invalidation prostsze.

**Uzasadnienie:**
1. **Auto-invalidation może być unwanted** - użytkownik może chcieć użyć cache mimo zmian w kodzie
2. **Git hash change != cache invalid** - możemy chcieć porównać wyniki między commitami
3. **Data change detection to overhead** - dane zmieniają się rzadko, user może manualnie sprawdzić
4. **Explicit > Implicit** - `--force` jest jasny, auto-invalidation może zaskoczyć

**Obecny mechanizm:**
```bash
# Use cache (default)
mla model --project X --experiment-id exp-...
# → Używa cache jeśli istnieje

# Force recompute
mla model --project X --experiment-id exp-... --force
# → Ignoruje cache, recompute all

# New experiment (implicit invalidation)
mla model --project X
# → Nowy exp-id = nowy cache dir
```

**Invalidation triggers:**
1. `--force` flag (explicit)
2. Nowy experiment ID (implicit)
3. Usuń ręcznie cache dir (manual)

---

### 4. ❌ Data Checksum (Fast) - POMINIĘTE

**Planned:**
```python
def compute_data_checksum(data_path: Path, sample_size: int = 1000) -> str:
    """
    Fast checksum based on:
    - File size
    - First/last N rows
    - Column names + dtypes
    """
    df_head = pd.read_csv(data_path, nrows=sample_size)
    # ... hash head + tail + metadata
```

**Decyzja:** POMINIĘTE - niepotrzebne.

**Uzasadnienie:**
1. **Dane zmieniają się rzadko** - raz pobrane z Kaggle, nie modyfikowane
2. **Checksum nie wykryje subtle bugs** - np. zmiana jednej wartości w środku pliku
3. **File mtime wystarczy** - jeśli plik został modified, znaczy że coś się zmieniło
4. **User wie kiedy dane się zmieniły** - może wtedy manualnie użyć `--force`

**Obecna "validacja" danych:**
```python
# BRAK - zakładamy że dane są immutable po pobraniu
```

**Jeśli user zmienia dane:**
1. Nowy experiment (`mla model --project X`) → nowy cache
2. Lub `--force` → recompute

---

## Testy

**Testy Phase 12:** BRAK (phase pominięta całkowicie)

**Obecne testy cache:**
```bash
uv run pytest tests/unit/test_preprocess.py::test_cache_reuse -v
# ✅ PASSED - File-based cache działa

uv run pytest tests/unit/test_model.py::test_force_recompute -v
# ✅ PASSED - --force ignoruje cache
```

**Pokrycie:** Podstawowy caching (file-based) testowany, composite keys nie.

---

## Ryzyka / następne kroki

### Ryzyko 1: Cache używany mimo zmian w danych
**Problem:** User zmienia `train.csv`, cache nie jest invalidowany automatycznie.

**Likelihood:** NISKI (dane z Kaggle nie zmieniają się)

**Impact:** WYSOKI (złe predykcje, wrong submissions)

**Mitigation:**
1. ✅ Dokumentacja: "Jeśli zmieniłeś dane, użyj `--force` lub nowy experiment"
2. ⚠️ TODO: Dodać warning jeśli `train.csv` mtime > `cache/train_fe.parquet` mtime
3. ✅ Developer zazwyczaj wie kiedy zmienił dane

**Decyzja:** ACCEPT RISK - dokumentacja + potential warning w przyszłości

---

### Ryzyko 2: Cache używany mimo zmian w kodzie preprocessing
**Problem:** User zmienia `preprocess.yaml`, cache nie invalidowany.

**Likelihood:** ŚREDNI (preprocessing templates zmieniają się często podczas dev)

**Impact:** ŚREDNI (suboptimal features, wrong CV)

**Mitigation:**
1. ✅ `--force` flag (explicit recompute)
2. ✅ Nowy experiment (implicit new cache)
3. ⚠️ TODO: Detect template change → warning

**Decyzja:** ACCEPT RISK - developer zazwyczaj pamięta że zmienił template

---

### Ryzyko 3: Cache rosnący bez kontroli
**Problem:** Każdy experiment tworzy cache, dysk się zapełnia.

**Likelihood:** WYSOKI (10 experiments = 10x cache)

**Impact:** NISKI (dysk tani, cache można usunąć)

**Mitigation:**
1. ✅ Cache per-experiment (można usunąć stare experimenty)
2. ⚠️ TODO: `mla admin gc --older-than 7d` (cleanup command)
3. ✅ Developer może ręcznie `rm -rf experiments/exp-*/preprocess_cache/`

**Decyzja:** ACCEPT RISK - future cleanup command wystarczy

---

### Następne kroki

**DONE Phase 12:** POMINIĘTE całkowicie (obecny caching wystarczy)

**TODO (future enhancements - low priority):**
1. Warning jeśli data mtime > cache mtime
2. Warning jeśli template zmienił się od ostatniego cache
3. Cleanup command: `mla admin gc --older-than 7d`

**Next:** Phase 13 - Test Plan

---

## Artefakty / PR

**Commits:** BRAK (phase pominięta)

**Pliki obecnego cache:**
```
experiments/<exp_id>/
├── state.json                          # Metadata (git, config, payload)
└── preprocess_cache/                   # Cache artifacts
    ├── train_fe.parquet
    ├── test_fe.parquet
    └── preprocessing_state.pkl         # Optional: fitted transformers
```

**Stan systemu po Phase 12:**
- Composite cache key: ❌ Pominięte (exp_id wystarczy)
- Cache metadata: ❌ Pominięte (state.json wystarczy)
- Auto-invalidation: ❌ Pominięte (manual --force)
- Data checksum: ❌ Pominięte (mtime check potential future)

---

**Status Phase 12:** POMINIĘTA JAKO ZBĘDNA

**Reasoning:** File-based caching z experiment_id jako key jest prostszy, bardziej debugowalny, i wystarczający dla single-user workflow. Composite keys (config+git+data hash) to over-engineering - dodają complexity bez znaczących korzyści.

**Design principle:** **Simple file-based > Complex hash-based** dla developer tools.

**Lesson learned:** Caching w ML nie musi być sophisticated. Prostota (file paths + manual invalidation) wygrywa z automation (auto-detection + hash keys) dla developer workflow.

**Next:** Phase 13 - Test Plan
