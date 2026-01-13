# MCTS Preprocessing Search Design (MLA-native execution + Optuna-like SQLite storage)

## 0. Cel i twarde wymagania

Ten dokument opisuje, jak dodać **Monte Carlo Tree Search (MCTS)** jako alternatywę dla Optuna w komendzie **`mla pre tune`**.

### Must-have

- **Parzystość CLI**: działa jak `mla pre tune`, ale z flagą `--mcts`:
  - **nie uruchamia Optuna**
  - uruchamia **MCTS** jako strategię przeszukiwania.
- **Persystencja i query w SQLite (zamiast plików JSON)**:
  - jeden plik SQLite jako źródło prawdy dla tree/trials/parametrów
  - wspiera proste zapytania o aktualny `best_score` (w trakcie runu, read-only)
  - umożliwia bezpieczny resume bez dodatkowych snapshotów plikowych (źródłem prawdy jest SQLite)
  - resume jest dozwolone tylko przy zgodnym fingerprint (chain/search space/objective) dla danego `study_name`
- **Kontrola eksplozji rozgałęzień (branching factor)**:
  - MCTS używa **progressive widening** (nie rozwija “wszystkich dzieci” z węzła)
  - generator akcji tworzy propozycje “na żądanie” (gating + kolejność z `mla_super_chain.yaml`)
- **Cache + deduplikacja/transpozycje**:
  - pipeline ma kanoniczną reprezentację (`pipeline_signature`)
  - duplikaty nie są ponownie liczone (cache wyników, także dla multi-fidelity)
- **MLA jako jedyny “runner” eksperymentów**: system MCTS (Monte Carlo Research) ma wyłącznie:
  1) wybrać następną konfigurację (pipeline),
  2) zmaterializować ją do **template’ów MLA**,
  3) uruchomić gotowy proces MLA (**model + template paths** / TaskQueue).
  
  MCTS **nie implementuje** własnego wykonywania preprocess/model (poza minimalnym wrapperem do wywołania MLA).
- **Źródło prawdy dla chainowania i parametrów**: MCTS ma bazować na mechanizmach z tunera Optuna:
  - `mla_super_chain.yaml` jako **kanoniczna kolejność** i logika “co po czym może być” (wspólne dla: Optuna / MCTS / Random)
    - plik zawiera sekcje konfiguracyjne dedykowane algorytmom: `optuna:`, `mcts:`, `random:`; w tym dokumencie opisujemy wyłącznie `mcts:`
  - search spaces używane w `preprocess_tune.py` (np. `SEARCH_SPACE_DIR/*.yaml`) jako **specyfikacja parametrów i variantów**.
- **Root/baseline**: root to **czysty score** (bez “dodatkowego preprocessing’u” – tylko to, co robi model/AutoGluon).
  - Dopuszczalne są “fixed harness steps” (np. przyspieszające evaluację), ale:
    - nie liczą się do “głębokości transformacji”
    - nie są częścią “odkrytego” łańcucha transformacji.
- **Małe kroki + testy**: implementacja ma być rozbita na **10–20 etapów**, każdy weryfikowalny (preferowane unit testy).
- **Traceability**:
  - czytelne logi z korelacją (node_id, exp_id, nazwa template’u)
  - każdy runtime zostawia po sobie template’y uruchomieniowe
  - nazwy template’ów muszą pozwalać szybko znaleźć odpowiadający wpis w logach.

### Non-goals

- Przepisywanie pipeline MLA (init/eda/preprocess/model) – tylko minimalne hooki.
- Tworzenie nowego, równoległego “systemu uruchamiania eksperymentów” obok MLA.

---

## 1. Punkty odniesienia (co już jest)

- `preprocess_tune.py` zawiera:
  - ładowanie search space (`_load_search_spaces`)
  - logikę dopuszczalności kroków (EDA gating / heavy / problem_type)
  - merge base/override/fixed config
  - mechanizm zapisywania “best templates”
- `mla_super_chain.yaml` definiuje:
  - kolejność kroków preprocess
  - `meta.fixed`, heavy flags, timeouty, override’y itp.
  - sekcje konfiguracyjne per-algorytm: `optuna:`, `mcts:`, `random:`
- `generate_random_preprocess_experiments.py` pokazuje właściwy pattern:
  - generuje template’y (preprocess step + chain + model)
  - uruchamia standardowym poleceniem MLA (`mla.py model --model-template ...`)
  - opcjonalnie wrzuca do TaskQueue

Wdrożenie MCTS ma maksymalnie kopiować te mechanizmy (a nie budować nowe).

---

## 2. Architektura (high-level)

### 2.1 Komponenty

1) **MCTSRunner** (nowy)
- uruchamiany przez `PreprocessTuneModule` gdy `--mcts`
- zarządza:
  - drzewem i statystykami (UCT)
  - budżetem (liczbą ewaluacji)
  - persistence (SQLite: `experiments/db/mcts.db`)
  - logami
  - delegowaniem ewaluacji do executora

2) **SuperChainActionSpace** (nowy, reuse logiki z Optuna)
- ładuje `mla_super_chain.yaml` + search spaces
- generuje “akcje” (kolejne kroki) zgodnie z:
  - kolejnością super-chain
  - ograniczeniami grup
  - EDA gating / heavy gating
  - compat z problem_type

3) **TemplateMaterializer** (nowy)
- z PipelineState robi:
  - template’y kroków preprocess
  - chain template preprocess
  - model template wskazujący na chain
- nadaje deterministyczne nazwy i zapisuje na dysku

4) **ExperimentExecutor** (interfejs)
- jedyna granica pomiędzy MCTS a MLA
- uruchamia “gotowy proces” MLA i zwraca wynik (score) w ustrukturyzowanej postaci

5) **MCTSStorage (SQLite)** (nowy)
- zapisuje: run/study, węzły/triale, parametry (odpowiednik template’ów), wyniki, błędy, statystyki MCTS
- umożliwia proste zapytania SQL o `best_score` (w trakcie runu)
- rekomendacja: schemat “Optuna-like” (studies/trials/trial_params/attrs), żeby reuse’ować istniejące narzędzia do podglądu

---

## 3. Model uruchamiania: “natywny MLA”

### 3.1 Zasada

MCTS nie uruchamia preprocess/model “po swojemu”. Preferowana ścieżka:

- **Subprocess CLI**: `mla.py model --model-template <T> --exp-id <E> ...`
- (opcjonalnie) **TaskQueue**: enqueue tego samego polecenia

To jest najbliższe temu, co robisz manualnie: “MLA + model + ścieżki do template’ów”.

### 3.2 Executor – tryby

#### Tryb A: Subprocess CLI (rekomendowany na start)

`MlaCliExecutor`:
- buduje polecenie zgodne z MLA
- odpala je synchronicznie
- parsuje wynik i zapisuje do SQLite (JSON jest tylko transportem) – patrz §4

#### Tryb B: TaskQueue (opcjonalnie później)

- MCTS dodaje task do kolejki
- czeka/polluje aż task skończy
- trudniejsze do stabilnego testowania (zostawić na późniejsze etapy)

---

## 4. Persystencja i query: SQLite (Optuna-like) zamiast plików JSON

### 4.1 Cel

- **Jedno źródło prawdy** dla runu (tree/trials/parametry/wyniki) – bez snapshotów plikowych po stronie MCTS (wszystko w SQLite).
- **Możliwość odpytywania “na żywo”** o aktualny best wynik (np. polling dashboardu / skryptu).
- **Resume**: MCTS ma się dać wznowić tylko na podstawie bazy (bez dodatkowych snapshotów plikowych).
- **Traceability**: w bazie da się dojść od `trial_id/node_id` → `template_base` → `experiment_id` → artefakty MLA.

### 4.2 Lokalizacja i tryb pracy (WAL + read-only query)

- Domyślna lokalizacja bazy na projekt: `projects/kaggle/<slug>/experiments/db/mcts.db` (konfigurowalne przez `mcts.storage_url`).
- Rekomendacja: **SQLite WAL** (`PRAGMA journal_mode=WAL`) → wielu czytelników + 1 pisarz.
- Wzorzec “monitorowania jak w Optuna”:
  - po stronie czytającej: `file:<path>?mode=ro&cache=shared` + `PRAGMA query_only=1`
  - referencja: `scripts/optuna_live.py` (tryb read-only, polling)
- Jeden plik DB może przechowywać **wiele** niezależnych `study` (jak w Optunie).

### 4.3 Study: definicja, bezpieczne przerwanie, resume + ochrona przed zmianą konfiguracji

W tej architekturze `study` oznacza **jeden** proces przeszukiwania MCTS o stałej:
- przestrzeni akcji (kolejność kroków z `mla_super_chain.yaml` + search spaces),
- funkcji celu (metryka + `direction` + model ewaluacyjny),
- gatingu (EDA/heavy/problem_type/max_features_out).

**Różne study:** możesz trzymać wiele uruchomień w tej samej bazie; rozróżniasz je przez `study_name`.

**Bezpieczne przerwanie i kontynuacja tego samego study:**
- stan drzewa (węzły/triale + statystyki) jest zapisany w SQLite, więc proces można ubić w dowolnym momencie
- na starcie runner:
  - wczytuje istniejące `study` (jeśli istnieje) i kontynuuje przydzielanie kolejnych triali
  - wykrywa “osierocone” triale w stanie `RUNNING` (np. bez heartbeat / zbyt stare) i oznacza je jako `FAIL` albo przywraca do `WAITING` (policy w config)

**Wykrycie zmiany konfiguracji i blokada resume (strict, domyślnie):**
- przy tworzeniu `study` zapisujemy w `study_user_attributes` fingerprint wejścia, np.:
  - `mcts.super_chain_sha256` (hash treści `conf/preprocess/mla_super_chain.yaml`)
  - `mcts.search_spaces_sha256` (hash treści wszystkich plików search space)
  - `mcts.objective_fingerprint` (metric_name + direction + model_template/evaluation config)
  - `mcts.gating_fingerprint` (problem_type + allow_heavy_* + max_features_out + EDA fingerprint)
  - `mcts.mcts_config_fingerprint` (parametry algorytmu, jeśli chcemy je “zamrozić” w study)
- przy resume liczymy fingerprint ponownie i:
  - jeśli się nie zgadza → **przerywamy** z jasnym błędem (“study config mismatch”) i prosimy o nowy `study_name`
  - opcjonalnie (niezalecane): `resume_policy: force` pozwala wznowić mimo różnic

Ważne: liczba iteracji/budżet (`budget`) może się zwiększać między sesjami; to nie psuje study (tak jak w Optunie).

### 4.4 Co zapisujemy (minimum)

**Poziom study:**
- `run_id` (stabilny, do nazw plików), `project`, `study_name`, timestampy start/stop
- `direction` + `metric_name` (żeby best score miał jednoznaczny sens)
- snapshot metadanych: hash (i opcjonalnie treść) `mla_super_chain.yaml` + search spaces
- MCTS config (budżet, max_depth, exploration_weight, seed, gating flags)

**Poziom trial/node (per ewaluacja):**
- `trial_id/node_id`, `depth`, `pipeline_signature` (kanoniczna reprezentacja pipeline’u)
- relacje parent→child trzymamy jako krawędzie (np. `mcts_edges.parent_trial_id` / `mcts_edges.child_trial_id`);
  `parent_id` w logach oznacza rodzica dla danej iteracji/selekcji
- reprezentacja “akcji” (dodany krok/variant + wylosowane parametry)
- **parametry, które stanowią zawartość template’ów**:
  - preprocess chain (lista kroków w kolejności + warianty + parametry)
  - model template (jeśli jest stały – zapisujemy 1× per run; jeśli zmienny – per trial)
  - preferowane: zapis w formie *parametrów* (Optuna-like `trial_params`) + opcjonalnie snapshot YAML (TEXT) do 1:1 reprodukcji
- mapping do świata MLA: `template_base`, `experiment_id`, ścieżki do template plików (jeśli materializowane), `exit_code`, runtime, error info

**Poziom statystyk MCTS (do resume):**
- `n_visits`, `value_sum`, `value_best` / `value_mean` (plus ewentualnie cache UCT)

### 4.5 Model danych: “Optuna-like” (rekomendowane)

Żeby uprościć późniejsze dashboardy i zapytania, warto przyjąć schemat zbliżony do Optuna (nawet jeśli MCTS nie uruchamia Optuny jako algorytmu):

- `studies`: 1× study (stała konfiguracja, identyfikowana przez `study_name`)
- `trials`: 1× node (stan może być `WAITING/RUNNING/COMPLETE/FAIL`)
- `trial_params`: spłaszczone parametry odpowiadające template’om (łatwe filtrowanie / hash)
- `trial_user_attributes`: “grubsze” pola JSON/TEXT (np. `actions_json`, `template_base`, `experiment_id`, snapshot YAML)
- `study_user_attributes`: metadane runu (konfig, hash super-chain, itd.)

Dzięki temu:
- SQL o best score wygląda “tak jak w Optuna”.
- istniejące narzędzia (`scripts/optuna_live.py`, `scripts/optuna_dashboard.py`) da się reuse’ować lub łatwo dostosować.

**Uwagi implementacyjne (rekomendowane podejście):**
- schema jest “Optuna-like” (nazwy tabel/kolumn zgodne z typowym layoutem Optuny), ale:
  - wartości w `trial_params.param_value` są przechowywane w formacie czytelnym dla człowieka (np. JSON string), bo to ma odpowiadać zawartości template’ów
  - MCTS nadal steruje doborem kolejnych konfiguracji (Optuna nie jest używana jako algorytm optymalizacji)

**Konwencja nazw dla `trial_params` (żeby dało się łatwo filtrować SQL):**
- preprocess chain:
  - `preprocess.depth` = liczba searched transforms
  - `preprocess.step.00.name`, `preprocess.step.00.variant`, `preprocess.step.00.<param>`
  - `preprocess.step.01.name`, ...
- model template:
  - `model.template` (jeśli stały) lub `model.<param>`
- wartości “meta/fixed harness” (jeśli chcemy je widzieć w DB):
  - `fixed.step.00.name`, `fixed.step.00.<param>` (i nie wliczamy ich do `depth`)

**Jak z `trial_params` powstaje template (materializacja):**
- `TemplateMaterializer` pobiera wszystkie `trial_params` dla `trial_id`, parsuje `param_value` przez `json.loads`, grupuje klucze po indeksie kroku (`step.00`, `step.01`, …).
- Z tego buduje listę kroków preprocess (name/variant/params), następnie zapisuje:
  - step template’y (per krok)
  - chain template (referencje do kroków)
  - (opcjonalnie) model template wskazujący na chain
- SQLite przechowuje konfigurację (źródło prawdy); pliki YAML to tylko materializacja “runtime” potrzebna do uruchomienia MLA i/lub łatwego replay.
- (opcjonalnie, dla pełnej reprodukcji): zapisujemy snapshot wygenerowanych YAML jako `trial_user_attributes` (np. `mcts.preprocess_chain_yaml`, `mcts.preprocess_steps_yaml[]`).

**Przykład: 2 kroki preprocess w `trial_params` (wartości jako JSON w polu `param_value`):**

| param_name | param_value |
|---|---|
| `preprocess.depth` | `2` |
| `preprocess.step.00.name` | `"imputer"` |
| `preprocess.step.00.variant` | `"simple"` |
| `preprocess.step.00.strategy` | `"median"` |
| `preprocess.step.01.name` | `"scaler"` |
| `preprocess.step.01.variant` | `"standard"` |
| `preprocess.step.01.with_mean` | `true` |

#### Struktura tabel SQLite (minimalny kontrakt / DDL)

Poniżej minimalny schemat, który ma wystarczyć do:
- best-score query (polling jak w Optunie),
- resume + dedupe po `pipeline_signature`,
- odtworzenia template’ów na podstawie `trial_params` + `trial_user_attributes`.

Wartości w `*_user_attributes.value_json` i `trial_params.param_value` zapisujemy jako JSON (tekst), np. `json.dumps(value)`.

```sql
-- Studies (1 study = 1 stała konfiguracja MCTS)
CREATE TABLE IF NOT EXISTS studies (
  study_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  study_name TEXT NOT NULL UNIQUE
);

-- 0=MINIMIZE, 1=MAXIMIZE (single-objective; objective=0)
CREATE TABLE IF NOT EXISTS study_directions (
  study_id   INTEGER NOT NULL,
  objective  INTEGER NOT NULL DEFAULT 0,
  direction  INTEGER NOT NULL,
  PRIMARY KEY (study_id, objective),
  FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS study_user_attributes (
  study_id   INTEGER NOT NULL,
  key        TEXT NOT NULL,
  value_json TEXT NOT NULL,
  PRIMARY KEY (study_id, key),
  FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
);

-- Trials (1 trial = 1 node w drzewie MCTS)
-- state: 0=RUNNING, 1=COMPLETE, 2=PRUNED, 3=FAIL, 4=WAITING
CREATE TABLE IF NOT EXISTS trials (
  trial_id          INTEGER PRIMARY KEY AUTOINCREMENT,
  study_id          INTEGER NOT NULL,
  number            INTEGER NOT NULL,
  state             INTEGER NOT NULL,
  datetime_start    TEXT,
  datetime_complete TEXT,
  UNIQUE (study_id, number),
  FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trials_study_state  ON trials(study_id, state);
CREATE INDEX IF NOT EXISTS idx_trials_study_number ON trials(study_id, number);

-- Single objective value (table name zgodna z nowszą Optuną)
CREATE TABLE IF NOT EXISTS trial_values (
  trial_id   INTEGER NOT NULL,
  objective  INTEGER NOT NULL DEFAULT 0,
  value      REAL NOT NULL,
  PRIMARY KEY (trial_id, objective),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trial_values_value ON trial_values(value);

-- Params (spłaszczona reprezentacja “zawartości template’ów”)
CREATE TABLE IF NOT EXISTS trial_params (
  trial_id    INTEGER NOT NULL,
  param_name  TEXT NOT NULL,
  param_value TEXT NOT NULL,
  PRIMARY KEY (trial_id, param_name),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trial_params_name ON trial_params(param_name);

-- Attributes (większe JSON-y: template_base, experiment_id, actions_json, fingerprinty per-trial, itd.)
CREATE TABLE IF NOT EXISTS trial_user_attributes (
  trial_id   INTEGER NOT NULL,
  key        TEXT NOT NULL,
  value_json TEXT NOT NULL,
  PRIMARY KEY (trial_id, key),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);

-- MCTS-specific node state (statystyki węzłów; wspólne dla transpozycji)
CREATE TABLE IF NOT EXISTS mcts_nodes (
  trial_id          INTEGER PRIMARY KEY,
  depth             INTEGER NOT NULL,
  pipeline_signature TEXT NOT NULL UNIQUE,
  n_visits          INTEGER NOT NULL DEFAULT 0,
  value_sum         REAL NOT NULL DEFAULT 0.0,
  value_best        REAL,
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE,
  CHECK (depth >= 0)
);
CREATE INDEX IF NOT EXISTS idx_mcts_nodes_depth  ON mcts_nodes(depth);

-- Edges: per-parent statystyki potrzebne do UCT/PUCT (N(n,a), Q̄(n,a), P(n,a))
CREATE TABLE IF NOT EXISTS mcts_edges (
  parent_trial_id INTEGER NOT NULL,
  child_trial_id  INTEGER NOT NULL,
  action_signature TEXT NOT NULL,
  prior           REAL,
  n_visits        INTEGER NOT NULL DEFAULT 0,
  value_sum       REAL NOT NULL DEFAULT 0.0,
  value_best      REAL,
  PRIMARY KEY (parent_trial_id, child_trial_id),
  UNIQUE (parent_trial_id, action_signature),
  FOREIGN KEY (parent_trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE,
  FOREIGN KEY (child_trial_id)  REFERENCES trials(trial_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mcts_edges_parent ON mcts_edges(parent_trial_id);
CREATE INDEX IF NOT EXISTS idx_mcts_edges_child  ON mcts_edges(child_trial_id);

-- Multi-fidelity wyniki i koszty (F0/F1/F2)
CREATE TABLE IF NOT EXISTS mcts_evaluations (
  trial_id        INTEGER NOT NULL,
  fidelity        TEXT NOT NULL,
  status          TEXT NOT NULL,      -- "WAITING" | "RUNNING" | "COMPLETE" | "FAIL" | "PRUNED"
  value           REAL,
  metric_name     TEXT,
  duration_sec    REAL,
  n_rows          INTEGER,
  cv_folds        INTEGER,
  time_limit_sec  INTEGER,
  n_features_out  INTEGER,
  details_json    TEXT,              -- np. fold scores / stderr tail / paths
  PRIMARY KEY (trial_id, fidelity),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mcts_eval_fidelity_value ON mcts_evaluations(fidelity, value);
```

Przykładowe klucze w `study_user_attributes` (do resume/compat):
- `mcts.super_chain_sha256`, `mcts.search_spaces_sha256`, `mcts.objective_fingerprint`, `mcts.gating_fingerprint`
- `mcts.schema_version` (schema bazy, niezależne od `schema_version` w JSON z MLA)

Przykładowe klucze w `trial_user_attributes` (traceability):
- `mcts.template_base`, `mcts.experiment_id`, `mcts.actions_json`, `mcts.executor_cmd`

### 4.6 Przykładowe zapytania SQL (best score + najlepsze parametry)

Zakładamy, że `study_name` jednoznacznie identyfikuje study (np. `mcts_preprocess_<slug>_v1`).

**Aktualny best score (maximize):**

```sql
SELECT t.trial_id, tv.value
FROM trials t
JOIN trial_values tv ON tv.trial_id = t.trial_id AND tv.objective = 0
JOIN studies s ON s.study_id = t.study_id
WHERE s.study_name = :study_name
  AND (t.state = 1 OR UPPER(CAST(t.state AS TEXT)) = 'COMPLETE')
ORDER BY tv.value DESC
LIMIT 1;
```

**Aktualny best score (minimize):**

```sql
SELECT t.trial_id, tv.value
FROM trials t
JOIN trial_values tv ON tv.trial_id = t.trial_id AND tv.objective = 0
JOIN studies s ON s.study_id = t.study_id
WHERE s.study_name = :study_name
  AND (t.state = 1 OR UPPER(CAST(t.state AS TEXT)) = 'COMPLETE')
ORDER BY tv.value ASC
LIMIT 1;
```

**Parametry najlepszego triala (template contents jako params):**

```sql
SELECT param_name, param_value
FROM trial_params
WHERE trial_id = :trial_id
ORDER BY param_name;
```

**Powiązanie best trial → template/experiment (attrs):**

```sql
SELECT key, value_json
FROM trial_user_attributes
WHERE trial_id = :trial_id
  AND key IN ('mcts.template_base', 'mcts.experiment_id');
```

### 4.7 Transport wyniku z MLA (opcjonalnie `--json`)

SQLite jest persystencją, ale wciąż trzeba “złapać” score z uruchomionego MLA.

Najprościej (na start):
- dodać do MLA minimalny hook `--json` / `--json-output <path>` zwracający pojedynczy obiekt z metrykami
- `MlaCliExecutor` parsuje JSON ze stdout (lub pliku) i zapisuje wynik do SQLite

Ważne: ten JSON **nie jest elementem schematu SQLite** i nie jest “stanem MCTS”.
To tylko transport wyniku z subprocessa MLA → po parsowaniu trafia do DB (np. `trial_values.value`, `trial_user_attributes`).
Pole `paths.state_json` (jeśli używane) wskazuje na `state.json` eksperymentu MLA (artefakt MLA, nie persystencja MCTS).

Minimalne wymagania dla JSON:
- jednoznaczny obiekt (najlepiej single-line)
- stabilna wersja schematu: `schema_version`
- na failure: `status=error`, `error_type`, `error_message`

Przykładowy schemat:

```json
{
  "schema_version": 1,
  "status": "ok",
  "project": "<slug>",
  "experiment_id": "exp-mcts_..._n000123",
  "model_template": "mcts_..._n000123",
  "preprocess_template": "mcts_..._n000123",
  "metrics": {
    "local_cv_score": 0.81234,
    "metric_name": "roc_auc"
  },
  "paths": {
    "experiment_dir": ".../experiments/...",
    "state_json": ".../state.json"
  },
  "timings": {
    "total_sec": 123.4
  }
}
```

---

## 5. Przestrzeń przeszukiwania: mla_super_chain + search spaces

### 5.1 Porządek i poprawność

**`mla_super_chain.yaml`** to kanon kolejności. PipelineState dzieli kroki na:

* **fixed harness steps**: `meta.fixed: true` (np. przyspieszające/sanity)
* **searched transforms**: kroki, które MCTS dobiera (to liczymy jako “głębokość”)

**Inwariant kolejności (dla searched transforms)**:

* wybieramy kroki jako **rosnący podciąg** listy `preprocessors` z super-chain
* pierwszy krok może być dowolnym elementem super-chain (niekoniecznie pierwszym)
* kolejne kroki muszą być później w kolejce

To spełnia: “można wybrać dowolny kawałek z chaina, ale w kolejności z pliku”.

**Znaczenie “depth” w drzewie MCTS:**
- `depth == liczba searched transforms` w chain (fixed harness steps nie zwiększają depth)
- przejście `parent -> child` oznacza dołożenie **dokładnie jednego** kolejnego preprocessora (z wariantem + parametrami), więc kolejne poziomy drzewa odpowiadają kolejnym poziomom preprocessingu

### 5.2 Parametry i varianty

MCTS ma korzystać z tej samej specyfikacji parametrów co Optuna:

* load search spaces z tego samego miejsca co `preprocess_tune.py`
* sampling parametrów ma interpretować te same typy speców:

  * `choice`, `int_range`, `float_range`, subsety itd.
* różnica: zamiast `trial.suggest_*` użyć RNG-based sampler, ale zgodny semantycznie ze specem

### 5.3 Feasibility / gating

Obowiązkowo reuse filtrów:

* EDA gating (np. `filter_by_eda`)
* allow_heavy_steps / allow_heavy_variants
* compat z `problem_type`
* safety limity i wymagania

W efekcie MCTS nie generuje pipeline’ów, których Optuna tuner by nie zaakceptował.

### 5.4 Model problemu (state/action/objective)

**Stan (node)**: pipeline preprocessingu jako uporządkowana lista transformacji + parametry (czyli dokładnie to, co materializujemy do template’ów).
Opcjonalnie (jeśli dostępne bez dużego kosztu): meta-informacje o wyjściu pipeline’u, np.:
- typy cech (num/cat/sparse/dense),
- `n_features_out`, sparsity,
- informacja o “ciężkości” kroków.

**Akcja (edge)**:
- najczęściej: **dodaj nowy krok preprocessingu** na końcu (zgodnie z kolejnością w `mla_super_chain.yaml`)
- opcjonalnie: “dostrój parametry istniejącego kroku” jako osobny typ akcji (jeśli będzie potrzebne)

**Cel (reward)**:
- główna metryka (np. CV AUC / RMSE) z `direction`
- opcjonalne kary (czas, liczba cech, itp.), np.:
  - `reward = score - λ_features * log1p(n_features_out) - λ_time * duration_sec`

W SQLite przechowujemy zarówno `score`, jak i składowe kosztu (żeby dało się robić Pareto/top‑K).

### 5.5 Rosnący branching factor → Progressive Widening (PW)

W preprocessingu liczba możliwych transformacji rośnie z głębokością, więc klasyczne “rozwiń wszystkie dzieci” nie zadziała.

**Progressive Widening** ogranicza liczbę rozwiniętych dzieci węzła:

`m(n) = k * N(n)^α`

- `N(n)` = liczba wizyt węzła
- `m(n)` = maksymalna liczba rozwiniętych dzieci
- typowo: `α ∈ [0.3, 0.7]`, `k ∈ [1, 10]`

**Praktyka implementacyjna:**
- generator akcji może zwracać wiele kandydatów, ale PW decyduje, ile dzieci realnie dodamy do grafu
- na iterację MCTS dodajemy **co najwyżej jedno** nowe dziecko (aż do `m(n)`)

### 5.6 Selekcja: UCT vs PUCT + priory (P)

Minimalnie:
- **UCT**: wybór dziecka przez `Q̄ + c * sqrt(ln N / N_edge)`

Lepsze w praktyce:
- **PUCT** (AlphaZero‑style): `Q̄ + c * P(n,a) * sqrt(N(n)) / (1 + N(n,a))`

Skąd `P(n,a)`?
- startowo: **uniform**
- następnie (opcjonalnie): heurystyki domenowe (np. preferencja sekwencji imputacja→kodowanie→scaling)
- dalej (opcjonalnie): **surrogate/prior model** uczony na historii triali: (cechy stanu + opis akcji) → przewidywany zysk / P(improve)

W DEBUG logach warto emitować `q`, `n`, `p`, `uct/puct` dla kandydatów, żeby dało się analizować decyzje.

### 5.7 Cache, deduplikacja i transpozycje (krytyczne)

W preprocessingu wiele ścieżek prowadzi do “tego samego” pipeline’u (albo semantycznie równoważnego).

**Kanoniczna reprezentacja pipeline’u (`pipeline_signature`):**
- normalizacja parametrów (domyślne wartości jawnie, stabilne typy)
- stabilne sortowanie kluczy w parametrach
- jednoznaczna serializacja (np. JSON) + hash

**Cache wyników:**
- jeśli `pipeline_signature` już ma wynik dla danej wierności (F0/F1/F2), nie uruchamiamy ponownie MLA
- cache działa zarówno dla pełnej ewaluacji, jak i multi-fidelity

**Transposition table (opcjonalnie, ale bardzo wartościowe):**
- statystyki MCTS są współdzielone dla tego samego `pipeline_signature`, nawet jeśli pipeline został osiągnięty różnymi ścieżkami
- do tego potrzebujemy per‑edge statystyk (patrz DDL: `mcts_edges`)

### 5.8 Multi-fidelity / early stopping (żeby ~1 min uruchamiać tylko dla najlepszych)

Jeśli każda ewaluacja kosztuje ~1 min, budżet szybko się kończy. Dlatego:

**Multi-fidelity (2–3 poziomy):**
- `F0` (tani): mniejszy subset danych, 1 fold CV, krótszy limit czasu
- `F1` (średni): większy subset / 3 fold
- `F2` (pełny): docelowa ewaluacja (Twoje ~1 min)

**Promocja (successive halving / ASHA):**
- większość kandydatów odpada na `F0/F1`
- `F2` dostają tylko najlepsi (np. top‑fraction / top‑K per “rung”)

**Early pruning:**
- jeśli w trakcie CV (foldy sekwencyjnie) wynik jest wyraźnie gorszy niż incumbent (z marginesem), przerywamy

**Opcjonalnie: value model (surrogate do wyceny liścia)**
- zamiast zawsze odpalać trening można utrzymywać regresor przewidujący `score` z cech pipeline’u (+ meta‑cech zbioru)
- taki model może służyć jako “wartość liścia” przy rolloutach i ograniczać liczbę odpaleń `F2`

W SQLite zapisujemy wyniki na wszystkich poziomach (np. `mcts_evaluations`), a do `trial_values` trafia wartość “docelowa” (zwykle `F2`) używana do best‑score query.

### 5.9 Wynik końcowy (nie tylko best)

Poza “najlepszym pipeline” warto produkować:
- top‑K pipeline’ów (score + koszty + linki do template/experiment)
- (opcjonalnie) Pareto front, jeśli liczymy też koszt (czas/rozmiar)
- zapis listy transformacji (replay + interpretowalność)

### 5.10 Równoległość (opcjonalnie): virtual loss

Jeśli mamy wielu workerów (subprocessy/TaskQueue), MCTS powinno unikać wybierania tego samego liścia przez kilka procesów.
Standardowe rozwiązanie to **virtual loss**:
- przy selekcji rezerwujemy liść (tymczasowo zwiększamy jego “koszt”/licznik) zanim odpalimy ewaluację
- po zakończeniu ewaluacji zdejmujemy virtual loss i robimy backprop na prawdziwym reward

To jest logika runtime (nie musi być persystowana), ale powinna być widoczna w `mcts.debug.log`.

---

## 6. Materializacja template’ów i nazewnictwo

### 6.1 Dlaczego template per ewaluacja

* wszystko jest replayable w 1 komendzie:

  * `mla.py model --model-template <NAME>`
* debug jest identyczny jak w standardowym MLA (bo to ten sam entrypoint)

### 6.2 Deterministyczny schemat nazw

Każde `study` ma `run_id` (stabilny identyfikator do nazw plików), zapisywany w SQLite przy pierwszym uruchomieniu
i re-używany przy resume.

Przykład (propozycja):
* `run_id = mcts_s{study_id:04d}_{study_slug}`

Każdy węzeł ma `node_id` (zero-padded). Rekomendacja: `node_id == trial.number` w DB (łatwy resume i brak kolizji).

Bazowa nazwa template’u:

* `base = {run_id}_n{node_id:06d}_d{depth:02d}_{sig8}`

Gdzie:

* `depth` = liczba searched transforms (fixed harness steps nie liczymy)
* `sig8` = 8 znaków z hasha sygnatury (template+variant+config w kolejności)

Pliki:

* `templates/preprocess/{base}.yaml` – chain template
* `templates/preprocess/{base}__{k:02d}-{step}.yaml` – step template’y
* `templates/model/{base}.yaml` – model template wskazujący na preprocess template

Każda linia logu dla ewaluacji musi zawierać `{base}` i `experiment_id`.

### 6.3 Root/baseline

Dwie opcje konfigurowalne:

* **strict**: model template bez preprocess_template (czysty baseline)
* **harness_only**: tylko fixed harness chain, bez transformacji

Obie utrzymują `depth=0`.

---

## 7. Logi, persystencja, reprodukcja

### 7.1 Katalog study

Dla `--mcts` tworzymy:

```
experiments/mcts_runs/{run_id}/
  mcts.log        # INFO (minimalny, “status study”)
  mcts.debug.log  # DEBUG (pełna analiza działania MCTS)
```

Persystencja działa przez SQLite (domyślnie):

* `experiments/db/mcts.db` – baza SQLite z `studies/trials/params/attrs` (patrz §4)

W bazie trzymamy:
* deduplikację (unikatowy `pipeline_signature`)
* mapowanie `trial_id/node_id` → `template_base` → `experiment_id`
* wyniki i statystyki MCTS (resume bez dodatkowych snapshotów plikowych)

### 7.2 Minimalne wymagania logów (2 poziomy: INFO + DEBUG)

**INFO (minimalny zestaw, jedna linia/event):**
- START/RESUME: `study_name`, `run_id`, `storage_url`, `resume_policy`, fingerprinty (chain/search spaces/objective), budżet
- TRIAL_START: `trial_id/node_id`, `parent_id`, `depth`, `pipeline_signature`, `template_base` (jeśli materializujemy), `experiment_id` (jeśli już nadany)
- TRIAL_END: `trial_id`, `status` (`COMPLETE/FAIL`), `score/value`, `metric_name`, `duration_sec`, `exit_code`, `experiment_id`
- NEW_BEST: `best_value`, `best_trial_id`, `best_template_base`, `best_experiment_id`
- STOP/SUMMARY: liczba triali, czas całkowity, best_value + wskazanie najlepszego triala

**DEBUG (analiza działania MCTS; wystarczające do rekonstrukcji decyzji):**
- ACTION_SPACE: lista możliwych akcji dla danego stanu + powody odfiltrowania (EDA gating/heavy/problem_type/max_features_out)
- SELECTION: pełna ścieżka selekcji (root→leaf) + UCT/score komponenty dla kandydatów (np. `q`, `n`, `uct`, `exploration_weight`)
- EXPANSION: wybrana akcja + wylosowane parametry (w formacie zgodnym z `trial_params`) + seed/RNG
- DEDUPE: wykrycie duplikatu `pipeline_signature` + decyzja (skip/reuse)
- MATERIALIZE: ścieżki wygenerowanych template’ów + `sig8`/hash
- EXECUTOR_CMD: dokładna komenda MLA (lub task_id) + timeouty
- EXECUTOR_RESULT: sparsowany wynik (JSON → `value/metric_name/paths`) + surowy stderr/stdout tail przy błędzie
- BACKPROP: reward/value + aktualizacje statystyk (`n_visits`, `value_sum`, `value_mean`) dla węzłów na ścieżce
- DB_STATE: przejścia stanów triali (`WAITING→RUNNING→COMPLETE/FAIL`) + obsługa “stale RUNNING” (fail/requeue)

**Wspólne wymagania (INFO i DEBUG):**
- każde zdarzenie ma klucze korelacyjne: `study_name`, `run_id`, `trial_id/node_id`, `parent_id`, `depth`, `pipeline_signature`, `template_base`, `experiment_id`
- format logu ma być “grep-friendly” (1 event = 1 linia; dopuszczalne: key=value albo JSON)

### 7.3 Raportowanie nowego best score

MCTS powinien (podobnie jak Optuna) monitorować i zgłaszać znalezienie nowej najlepszej wartości funkcji celu (best score).

**Wymagania:**
* Wykrycie poprawy wyniku globalnego względem dotychczasowego `best_value`.
* Logowanie zdarzenia w sposób wyróżniający się (np. `[NEW BEST] ...`).
* Opcjonalnie: powiadomienia (dźwięk, komunikator).

**Referencja implementacyjna:**
Przykładowa implementacja monitorowania bazy (polling) i wysyłania powiadomień (Telegram + bell) znajduje się w skrypcie `scripts/optuna_live.py`. Kluczowe elementy:
* Cykliczne odpytywanie bazy SQLite (tryb read-only).
* Porównywanie `current_best` z `last_best_val`.
* Obsługa powiadomień zewnętrznych.

---

## 8. Konfiguracja

Preferowane: dopisać `mcts:` do istniejącej konfiguracji tunera (np. w `mla_super_chain.yaml`),
żeby nie wprowadzać drugiego systemu konfiguracyjnego.

Preferowana nazwa pliku super-chain (uniwersalna): `conf/preprocess/mla_super_chain.yaml`.

Obecnie `conf/preprocess/mla_super_chain.yaml` może być symlinkiem do legacy nazwy
`conf/preprocess/super_chain_optuna.yaml` (compat), ale dokumentacja i nowe komponenty powinny używać
`mla_super_chain.yaml` jako kanonicznej ścieżki.

**Konwencja zawartości `mla_super_chain.yaml`:**
- część wspólna (używana przez wszystkie algorytmy): `preprocessors`, `evaluation`, `meta.*`
- sekcje dedykowane algorytmom: `optuna:`, `mcts:`, `random:`
- ten dokument dotyczy wyłącznie `mcts:`

Przykład:

```yaml
preprocessors: [...]
evaluation: {...}

optuna: {...}
random: {...}

mcts:
  storage_url: "sqlite:///experiments/db/mcts.db"
  study_name: "mcts_preprocess_{project}_v1" # stała nazwa dla resume; zmiana chain/search space => nowy study_name
  direction: "maximize"          # albo "minimize" – musi być spójne z metryką
  resume_policy: "strict"        # "strict" | "force"
  stale_running_trials: "fail"   # "fail" | "requeue"

  budget: 80
  max_depth: 9
  selection_policy: "puct"       # "uct" | "puct"
  exploration_weight: 1.414      # c w UCT/PUCT
  prior_policy: "uniform"        # "uniform" | "heuristic" | "surrogate"

  # Progressive widening: m(n)=k*N(n)^alpha
  expansion_width: 2             # k
  expansion_alpha: 0.5           # alpha
  seed: 42

  root_mode: "harness_only"      # "no_preprocess" | "harness_only"
  executor: "cli"                # "cli" | "task_queue"
  cli_json: true                 # używaj --json w mla.py (transport wyniku → zapis do SQLite)

  allow_heavy_steps: true
  allow_heavy_variants: true

  multi_fidelity:
    enable: true
    levels:
      - name: "F0"
        sample_frac: 0.2
        cv_folds: 1
        time_limit_sec: 15
      - name: "F2"
        sample_frac: 1.0
        cv_folds: 3
        time_limit_sec: 60
    promotion:
      strategy: "asha"           # "none" | "successive_halving" | "asha"
      top_fraction: 0.25

  pruning:
    enable: true
    incumbent_margin: 0.0        # np. 0.002 dla AUC

  penalties:
    features_lambda: 0.0
    time_lambda: 0.0

  parallelism:
    workers: 1
    virtual_loss: 1.0

  dedupe:
    enable: true
    strategy: "unique_signature" # unikat w SQLite po `pipeline_signature`
```

---

## 9. Plan wdrożenia w małych krokach (z testami)

> Zasada: wszystko, co odpala realny trening, idzie przez `ExperimentExecutor`, żeby unit testy używały `FakeExecutor`.

**Zasada testowania iteracyjnego:**
- każdy krok dodaje/rozszerza testy dla dostarczonej funkcjonalności (preferowane unit)
- po implementacji kroku `N` uruchamiamy testy z kroków `1..N` (żeby nie regresować fundamentów)

| Krok | Deliverable                    | Zmiany                                          | Testy (preferowane unit)                             |
| ---: | ------------------------------ | ----------------------------------------------- | ---------------------------------------------------- |
|    1 | Routing `--mcts` (stub)        | flaga + wybór MCTSRunner                        | test parsowania CLI / wyboru ścieżki                 |
|    2 | `MCTSConfig` + walidacja       | load config, defaults, bounds                   | invalid config -> wyjątek; defaults działają         |
|    3 | Loader super-chain             | czytanie `mla_super_chain.yaml` do struktury    | zachowanie kolejności, odczyt `meta.fixed`           |
|    4 | Loader search spaces           | reuse `_load_search_spaces` jako shared util    | ładowanie yaml; czytelny error przy brakach          |
|    5 | Model `PipelineState`          | fixed vs searched, sygnatura                    | sygnatura stabilna; kolejność wpływa na hash         |
|    6 | Generator akcji                | `next_actions(state)` (kolejność + grupy)       | akcje zawsze “później” w chain; brak duplikatów grup |
|    7 | Sampler parametrów             | RNG-based zgodny ze specami                     | dla każdego typu spec: value in-domain               |
|    8 | Materializer template’ów       | zapis plików + naming scheme                    | pliki powstają; chain wskazuje na kroki              |
|    9 | `ExperimentExecutor` interface | `FakeExecutor` + Result schema                  | MCTS loop działa bez MLA; wyniki wracają             |
|   10 | Core MCTS (PUCT + PW)          | selection/expansion/backprop + progressive widening | test z FakeExecutor: wybiera lepszą gałąź; PW nie eksploduje |
|   11 | Cache/transpozycje             | pipeline_signature + cache wyników + (opc.) transposition table | duplikaty nie odpalają executora                     |
|   12 | SQLite storage                 | schema (nodes/edges/evaluations) + resume       | restart kontynuuje; resume_policy=strict działa      |
|   13 | NEW BEST + notifier            | event + (opc.) Telegram/Bell                    | caplog: NEW BEST event; notifier nie blokuje runu    |
|   14 | `MlaCliExecutor`               | składanie komendy + timeouty                    | test budowania command line; failure mapping         |
|   15 | `--json` w MLA                 | minimalny hook w runnerze                       | test schematu JSON; status=error na wyjątku          |
|   16 | `--dry-run`                    | generuje template’y i komendy bez uruchamiania  | pliki + wpisy w SQLite powstają                      |
|   17 | Multi-fidelity + pruning       | F0/F1/F2 + ASHA/successive halving              | promocje zgodne z regułami; PRUNED działa            |
|   18 | Real run budżet=1–2            | end-to-end na małej próbie                      | score złapany, artefakty istnieją                    |
|   19 | (opcjonalnie) równoległość     | virtual loss / worker pool                      | limit workerów, brak kolizji na liściach             |
|   20 | Wyniki (top-K + best)          | export top-K + promocja best template           | top-K stabilne; best aktualizuje się tylko przy poprawie |
