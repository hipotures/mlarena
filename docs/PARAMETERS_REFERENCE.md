# MLArena Parameters Reference

This document provides a comprehensive list of dotted-path parameters that can be used via the CLI (e.g., `model.time_limit=300`) or in YAML templates.

---

## 🌍 Global & Flow Controls
These parameters control the overall execution of the pipeline.

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `project` | string | (Required) | The competition slug (e.g., `titanic`). |
| `experiment_id` | string | None | Unique ID for the experiment. If not set, a timestamped ID is generated. |
| `profile` | string | None | Name of a config profile to load (e.g., `smoke`, `dev`). |
| `model_template` | string | `baseline` | The model YAML template to use. |
| `preprocess_template`| string | None | The preprocessing YAML template or chain name. |
| `wait_seconds` | int | `30` | Delay before fetching the score from Kaggle. |
| `skip_submit` | bool | `false` | If true, stops after prediction (no Kaggle upload). |
| `skip_git` | bool | `false` | If true, skips the automatic git commit after success. |
| `force` | bool | `false` | Re-run modules even if they are marked as completed. |
| `lock` | bool | `false` | Create `overwrite.lock` after success to prevent future re-runs. |
| `skip_deps` | bool | `false` | Run only the specified module, ignoring its dependencies. |
| `show_payload` | bool | `false` | Display the output payload of modules in the console. |

---

## 🛠️ Common Section (`common.*`)
Parameters used as fallback values by multiple modules (Model, Tune, Preprocess).

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `common.seed` | int | `42` | Global random seed for reproducibility. |
| `common.time_limit` | int | None | Global training time limit in seconds. |
| `common.use_gpu` | bool | `false` | Globally enable/disable GPU acceleration. |
| `common.preset` | string | `medium` | AutoGluon quality preset (`best`, `high`, `medium`, `good`, `fast`). |

---

## 📦 Module Specific Parameters

### 1. Model Training (`model.*`)
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `model.mla_retention` | bool | `false` | Cleanup AutoGluon artifacts, keeping only the best model. |
| `model.time_limit` | int | (from common) | Override time limit for the model module. |
| `model.preset` | string | (from common) | Override AutoGluon preset. |
| `model.use_gpu` | bool | (from common) | Override GPU usage. |
| `model.hpo_preset` | string | None | Name of an HPO preset (e.g., `hpo_boost_medium`). |
| `model.hyperparameters`| dict | {} | Nested model parameters (e.g., `model.hyperparameters.GBM.max_depth=5`). |
| `model.quiet_model_panel` | bool | `false` | Suppress MODEL header/footer panels. |
| `model.verbosity` | int | `2` | AutoGluon logging verbosity (lower = quieter). |

### 2. Preprocessing (`preprocess.*`)
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `preprocess.cache` | bool | `true` | Use cached artifacts if input data/config hasn't changed. |
| `preprocess.quiet_preprocess_panel` | bool | `false` | Suppress PREPROCESS header/footer panels. |

### 2b. Optuna (`optuna.*`)
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `optuna.study_name` | string | `optuna_preprocess` | Optuna study name. |
| `optuna.direction` | string | `minimize` | Optuna direction (`maximize` or `minimize`). |
| `optuna.n_trials` | int | `10` | Number of trials to run. |
| `optuna.optuna_workers` | int | `1` | Parallel Optuna workers (`n_jobs`). |
| `optuna.storage_url` | string | `sqlite:///...` | Optuna storage URL. |
| `optuna.optuna_storage_timeout` | int | (optional) | SQLite connection timeout (seconds). |
| `optuna.optuna_live` | bool | `false` | Enable live Optuna dashboard integration. |

### 2c. Preprocess Tune (`preprocess_tune.*`)
Optuna settings live under `optuna.*` (legacy `preprocess_tune.*` names are still accepted).

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `preprocess_tune.super_chain` | string | `conf/preprocess/super_chain_optuna.yaml` | Super-chain YAML used for tuning. |
| `preprocess_tune.max_trial_sec` | int | `1800` | Hard timeout per trial (seconds). |
| `preprocess_tune.model_cleanup` | bool | `false` | Remove `optuna_model` and `*_processed.csv.gz` files after each trial. |
| `preprocess_tune.ag_cleanup` | bool | `false` | Alias for `model_cleanup`. |
| `preprocess_tune.cleanup_processed` | bool | `false` | Remove `*_processed.csv.gz` after each trial (implied by `model_cleanup`). |
| `preprocess_tune.allow_heavy_steps` | bool | `false` | Allow heavy steps in the super-chain. |
| `preprocess_tune.allow_heavy_variants` | bool | `false` | Allow heavy variants within steps. |
| `preprocess_tune.max_features_out` | int | `50000` | Hard cap on output features. |
| `preprocess_tune.model_template` | string | (optional) | FAST model template for evaluation (fallback: ``evaluation.model`` in super-chain). |
| `preprocess_tune.seed` | int | `common.seed` | Trial seed. |
| `preprocess_tune.quiet_preprocess_panel` | bool | `false` | Suppress PREPROCESS panels during tuning. |
| `preprocess_tune.quiet_model_panel` | bool | `false` | Suppress MODEL panels during tuning. |
| `preprocess_tune.model_verbosity` | int | `model.verbosity` | AutoGluon logging verbosity override for tuning. |

### 3. Submission (`submit.*`)
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `submit.confirm_timeout`| int | `60` | Seconds to wait for manual confirmation before auto-submitting. |
| `submit.queue_submit` | bool | `false` | Add the submission to the local queue instead of uploading immediately. |

### 4. Fetch Score (`fetch_score.*`)
| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `fetch_score.wait_seconds`| int | (from global)| Seconds to wait for Kaggle to process the submission. |

---

## CLI Overrides

MLArena accepts only `key=value` overrides on the command line. Use dotted paths for nested config values (e.g., `model.time_limit=600`, `common.seed=123`).
