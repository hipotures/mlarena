# Development TODO

> **Note:** This file tracks development tasks. Consider moving items to GitHub Issues for better project management.

## Performance Optimization

### Completed
- [x] Remove toplevel pandas imports from modules (moved to execute() functions) - **reduced from 4.5s to 2.5s**
- [x] Remove toplevel pandas import from utils/init/core.py
- [x] Skip ModuleRegistry.clear() on every run (use cached imports)

### Planned Optimizations (target: <1s for completed modules)
- [ ] **State caching** - Don't reload state.json when module status is already `completed` (estimated savings: ~0.5s)
  - Cache state in memory after first load
  - Only reload when file mtime changes
  - Skip JSON parsing for repeat calls

- [ ] **Lazy config loading** - Don't load project config.py for completed modules (estimated savings: ~0.3s)
  - Move `load_project_config()` inside module execute()
  - Only load when module actually runs
  - Completed modules don't need config

- [ ] **Skip pipeline loading** - Don't parse pipeline YAML for already-completed modules (estimated savings: ~0.2s)
  - Check module status before loading pipeline
  - Pipeline only needed when module executes

- [ ] **Parallel module discovery** - Import modules concurrently using ThreadPoolExecutor (estimated savings: ~0.2s)
  - Current: sequential import of 10 modules
  - Use threads to parallelize imports

- [ ] **Bytecode compilation** - Pre-compile modules to .pyc in production (estimated savings: ~0.1s)
  - Run `python -m compileall src/` during deployment
  - Ensure .pyc files are committed or generated at install time

**Expected total**: 4.5s → 2.5s (current) → <1s (with all optimizations)

## Experiment Manager Enhancements

- [ ] Add explicit `aborted` status to ExperimentManager modules (separate from `failed`) to mark user-interrupted runs without implying an error. Update state handling, list views, and restart logic accordingly.

## CLI meta-commands (non-flow)

- [ ] Dodać tryb pipeline bez podawania modułu: `mla --project X` uruchamia domyślny pipeline, `--from preprocess` startuje od preprocess, `--from model` od model itd.
- [ ] Dodać subkomendę `admin` (nie jako moduł) z akcjami:
  - `admin list --project X` (lista eksperymentów + statusy modułów z state.json)
  - `admin clean --project X [--artifacts|--models|--cache]` z potwierdzeniem
  - `admin gc --project X --older-than 7d` (sprzątanie starych eksperymentów)
  - `admin status --project X` (podsumowanie ostatnich runów)
- [ ] Utrzymać kompatybilność: najpierw obsłużyć meta-komendy, potem moduły, na końcu tryb pipeline.
- [ ] Zarezerwować przestrzeń nazw modułów (brak modułów `admin`, `pipeline`, `list`).
- [ ] Opcjonalny alias `mla list-experiments --project X` jako cienka nakładka na `experiment_logger.py`.
- [ ] W `admin clean` dodać ochronę przed przypadkowym usuwaniem: pokazanie targetów i prompt.

## Sample Weight Enhancements (Drift/AV optimization)

Based on AutoGluon best practices for covariate shift and importance weighting. See: [AutoGluon sample_weight docs](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html)

### T2 — Normalizacja wag (Weight Normalization)

**Cel:** Stabilność numeryczna i zgodność z innymi narzędziami. AutoGluon rekomenduje, żeby wagi sumowały się do liczby wierszy (średnia ≈ 1.0).

**Modyfikacje:**

1. **Plik:** `src/mlarena/defaults/preprocessing/adversarial_validation.py`
2. **Lokalizacja:** Po wyliczeniu wag (linia ~230-240), przed zapisem do CSV
3. **Kod:**
   ```python
   # Normalize weights to sum to N (AutoGluon recommendation)
   weights = weights * (len(weights) / weights.sum())
   ```
4. **Config parameter (opcjonalny):**
   ```yaml
   # In template config:
   normalize_weights: true  # default: true
   ```

**Benefit:** Poprawa stabilności numerycznej, lepsza zgodność z dokumentacją AG, porównywalne wyniki między różnymi metodami ważenia.

---

### T3 — Clipping ekstremów wag (Weight Clipping)

**Cel:** Ograniczenie wariancji importance-weighting. Przy silnym drifcie niektóre wagi mogą być ekstremalne (np. 50x), co destabilizuje trening.

**Modyfikacje:**

1. **Plik:** `src/mlarena/defaults/preprocessing/adversarial_validation.py`
2. **Lokalizacja:** Po normalizacji (T2), przed zapisem do CSV
3. **Kod:**
   ```python
   # Clip extreme weights
   if config.get("clip_weights"):
       clip_method = config.get("clip_method", "percentile")  # "percentile" | "fixed"

       if clip_method == "percentile":
           lower = config.get("clip_lower_percentile", 0.5)  # P0.5
           upper = config.get("clip_upper_percentile", 99.5)  # P99.5
           w_min, w_max = np.percentile(weights, [lower, upper])
       else:  # fixed
           w_min = config.get("clip_min", 0.2)
           w_max = config.get("clip_max", 5.0)

       weights = np.clip(weights, w_min, w_max)

       # Re-normalize after clipping
       weights = weights * (len(weights) / weights.sum())
   ```

4. **Config parameters:**
   ```yaml
   # In template config:
   clip_weights: true
   clip_method: "percentile"  # or "fixed"
   clip_lower_percentile: 0.5
   clip_upper_percentile: 99.5
   # OR for fixed:
   # clip_method: "fixed"
   # clip_min: 0.2
   # clip_max: 5.0
   ```

**Benefit:** Często poprawia public score kosztem minimalnie gorszego local CV. Szczególnie ważne przy silnym drifcie.

---

### T4 — Drift-aware bagging via groups (Group-based CV splits)

**Cel:** Bagging/stacking AutoGluon waliduje się "wzdłuż gradientu driftu", nie losowo. Poprawia korelację local→public przy covariate shift.

**Modyfikacje:**

#### 4a. Preprocessing: generowanie kolumny `__grp__`

1. **Plik:** `src/mlarena/defaults/preprocessing/adversarial_validation.py`
2. **Lokalizacja:** Po wyliczeniu `p_test` (prawdopodobieństwo "to test"), przed return
3. **Kod:**
   ```python
   # Generate drift groups for AutoGluon groups parameter
   if config.get("create_drift_groups"):
       n_groups = config.get("drift_groups_count", 5)  # 5 or 10
       train_df["__grp__"] = pd.qcut(
           av_predictions_train,  # p_test probabilities
           q=n_groups,
           labels=False,
           duplicates='drop'
       )
   ```

4. **Config parameters:**
   ```yaml
   # In preprocess template:
   create_drift_groups: true
   drift_groups_count: 5  # or 10 for more granular splits
   ```

5. **Output:** Kolumna `__grp__` w `train_processed.csv` (values: 0-4 or 0-9)

#### 4b. Model: przekazanie `groups` do TabularPredictor

1. **Plik:** `src/mlarena/defaults/models/autogluon_baseline.py`
2. **Lokalizacja:** W `TabularPredictor()` constructor (linia ~178)
3. **Kod:**
   ```python
   # Check if drift groups are available
   groups_column = config.dataset.groups_column  # New field in DatasetConfig
   if groups_column and groups_column in train_data.columns:
       # Don't drop groups column, AG uses it for CV splits
       pass  # groups stays in train_data
   else:
       groups_column = None

   predictor = TabularPredictor(
       label=target_column,
       path=str(config.system.model_path),
       eval_metric=config.dataset.metric,
       problem_type=config.dataset.problem_type,
       sample_weight=sample_weight_param,
       weight_evaluation=weight_evaluation_param,
       groups=groups_column,  # NEW: drift-aware bagging
       verbosity=2,
   )
   ```

4. **Config models (DatasetConfig):**
   ```python
   # In src/kaggle_tools/config_models.py:
   class DatasetConfig(ExtraModel):
       # ... existing fields ...
       groups_column: Optional[str] = None  # NEW: Column name for group-based CV
   ```

5. **Template usage:**
   ```yaml
   # In model template:
   config:
     groups_column: "__grp__"  # Use drift groups from preprocessing
   ```

**Benefit:** LeaveOneGroupOut bagging validation lepiej symuluje test set (który ma inny rozkład niż train). Poprawia generalizację przy drifcie.

---

### T5 — External dataset weight variants (3 strategie ważenia external data)

**Cel:** Sprawdzić optymalną strategię ważenia external dataset. External może być "less test-like" i psuć generalizację mimo podniesienia local CV.

**Modyfikacje:**

1. **Plik:** `src/mlarena/defaults/models/autogluon_baseline.py`
2. **Lokalizacja:** Sekcja merge train + orig (linie 144-151)
3. **Kod (zastąpić istniejący fragment):**
   ```python
   # If model merges train+orig, determine weights for external rows
   if merged_rows and len(weights) == base_train_rows:
       # Get external weighting strategy from config
       ext_weight_strategy = config.dataset.external_weight_strategy or "mean"

       if ext_weight_strategy == "neutral":
           # E1: External weight = 1.0 (neutral)
           fill_value = 1.0

       elif ext_weight_strategy == "mean":
           # E2: External weight = mean(kaggle_weights) [current default]
           fill_value = float(weights.mean()) if weights.notna().any() else 1.0

       elif ext_weight_strategy == "drift_based":
           # E3: External weight from same AV model (requires p_test for external)
           # This requires preprocessing to compute drift probabilities for external
           # Look for external_weights in artifacts
           ext_weights_artifact = artifacts.get("external_weight") if artifacts else None
           if ext_weights_artifact is not None:
               # Use pre-computed external weights from preprocessing
               fill_value = None  # Will use per-row weights
               ext_weights_series = pd.to_numeric(ext_weights_artifact.iloc[:, 0], errors="coerce")
               weights = pd.concat([weights, ext_weights_series.reset_index(drop=True)], ignore_index=True)
           else:
               # Fallback to mean
               print("[AutoGluon External Weights] WARNING: drift_based strategy requires external_weight artifact, falling back to mean")
               fill_value = float(weights.mean()) if weights.notna().any() else 1.0
       else:
           raise ValueError(f"Unknown external_weight_strategy: {ext_weight_strategy}")

       # Apply fill_value strategy (if not using per-row weights)
       if fill_value is not None:
           weights = pd.concat(
               [weights, pd.Series([fill_value] * merged_rows)],
               ignore_index=True,
           )

       print(f"[AutoGluon External Weights] Strategy: {ext_weight_strategy}, fill_value: {fill_value}")
   ```

4. **Config models (DatasetConfig):**
   ```python
   # In src/kaggle_tools/config_models.py:
   class DatasetConfig(ExtraModel):
       # ... existing fields ...
       external_weight_strategy: Optional[str] = "mean"  # "neutral", "mean", "drift_based"
   ```

5. **Template usage:**
   ```yaml
   # In model template:
   config:
     external_weight_strategy: "neutral"  # or "mean" or "drift_based"
   ```

6. **Preprocessing for drift_based (opcjonalne):**
   - Modify `adversarial_validation.py` to compute drift probabilities for `orig_df`
   - Save as separate artifact: `external_av_weights.csv`
   - Return in state: `external_weight_path`

**Benefit:** External dataset może pomagać lub szkodzić - dobra strategia ważenia maksymalizuje sygnał minimalizując szum.

---

### Implementation Priority

**Quick wins (template-only):**
- ✅ T1 (weight_evaluation) - Already implemented
- ✅ T6 (model types) - Already implemented

**High impact (code changes):**
1. **T2 (normalization)** - Easy, always beneficial (~10 LOC)
2. **T3 (clipping)** - Easy, often improves public (~20 LOC)
3. **T4 (groups)** - Medium, strong benefit for drift (~30 LOC preprocessing + 10 LOC model)
4. **T5 (external weights)** - Medium-hard, project-specific (~40 LOC)

**Recommended order:**
1. T2 → T3 (same file, can be done together)
2. T4a (preprocessing groups)
3. T4b (model groups parameter)
4. T5 (if using external dataset)
