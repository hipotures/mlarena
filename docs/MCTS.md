# MCTS Preprocessing Search Design (MLA-native execution + generator z Optuna chain)

## 0. Cel i twarde wymagania

Ten dokument opisuje, jak dodać **Monte Carlo Tree Search (MCTS)** jako alternatywę dla Optuna w komendzie **`mla pre tune`**.

### Must-have

- **Parzystość CLI**: działa jak `mla pre tune`, ale z flagą `--mcts`:
  - **nie uruchamia Optuna**
  - uruchamia **MCTS** jako strategię przeszukiwania.
- **MLA jako jedyny “runner” eksperymentów**: system MCTS (Monte Carlo Research) ma wyłącznie:
  1) wybrać następną konfigurację (pipeline)
  2) zmaterializować ją do **template’ów MLA**,
  3) uruchomić gotowy proces MLA (**model + template paths** / TaskQueue).
  
  MCTS **nie implementuje** własnego wykonywania preprocess/model (poza minimalnym wrapperem do wywołania MLA).
- **Źródło prawdy dla chainowania i parametrów**: MCTS ma bazować na mechanizmach z tunera Optuna:
  - `mla_super_chain.yaml` jako **kanoniczna kolejność** i logika “co po czym może być”
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
  - persistence (tree.json, registry)
  - logami
  - delegowaniem ewaluacji do executora

2) **OptunaChainActionSpace** (nowy, reuse logiki z Optuna)
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

---

## 3. Model uruchamiania: “natywny MLA”

### 3.1 Zasada

MCTS nie uruchamia preprocess/model “po swojemu”. Preferowana ścieżka:

- **Subprocess CLI**: `mla.py model --model-template <T> --exp-id mcts-<E> ...` (prefix id mcts)
- (opcjonalnie) **TaskQueue**: enqueue tego samego polecenia

To jest najbliższe temu, co robisz manualnie: “MLA + model + ścieżki do template’ów”.

### 3.2 Executor – tryby

#### Tryb A: Subprocess CLI (rekomendowany na start)

`MlaCliExecutor`:
- buduje polecenie zgodne z MLA
- odpala je synchronicznie
- parsuje wynik (najlepiej z JSON – patrz §4)

#### Tryb B: TaskQueue (opcjonalnie później)

- MCTS dodaje task do kolejki
- czeka/polluje aż task skończy
- trudniejsze do stabilnego testowania (zostawić na późniejsze etapy)

---

## 4. Opcjonalna flaga `--json` (ułatwienie parsowania wyników)

### 4.1 Cel

Dodać minimalny hook do MLA:

- `--json` (bool) i/lub `--json-output <path>`
- po zakończeniu modułu (np. `model`) MLA wypisuje / zapisuje **pojedynczy JSON** z metrykami i ścieżkami do artefaktów

To pozwala MCTS nie grzebać w katalogach, tylko brać “reward” z jednego źródła.

### 4.2 Wymagania dla JSON

- jednoznaczny obiekt (najlepiej single-line)
- stabilna wersja schematu: `schema_version`
- na failure: `status=error`, `error_type`, `error_message`

### 4.3 Proponowany schemat

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
B
B
B

