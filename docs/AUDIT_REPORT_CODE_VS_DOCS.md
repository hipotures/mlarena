# MLArena Documentation-Code Consistency Audit Report

**Generated:** 2026-01-04
**Scope:** Complete mlarena framework (src/, docs/, scripts/, README.md, AGENTS.md)
**Methodology:** Systematic comparison of documentation claims against code implementation

---

## 1) Executive Summary

### Critical Findings (5)

1. **CLI Flag Inconsistency**: Documentation uses `--preprocess-template` but CLI implementation expects `preprocess_template=value` (dotted override format) - Medium/High severity
2. **Missing `fetch-score` Flag**: Documentation claims `--skip-score-fetch` flag exists, but implementation only has `skip_submit` and `wait_seconds` - High severity
3. **Incomplete HPO Template Documentation**: README references `test_hpo_medium`, `test_hpo_high`, `test_hpo_best` templates but actual templates are `hpo_boost_medium`, `hpo_boost_high`, `hpo_boost_best` - High severity
4. **Preprocessing Module Path Mismatch**: AGENTS.md states "Copy `src/mlarena/preprocessing/TEMPLATE.py`" but actual path is `src/mlarena/defaults/preprocessing/` - Medium severity
5. **Queue Command Delegation**: CLI delegates `queue` command to `scripts/task_queue.py` but README shows `mla queue` - works but implementation detail differs - Low severity

### Other Significant Issues (7)

6. **Inconsistent Template Naming**: Documentation switches between `preprocess_template`, `--preprocess-template`, and `preprocess-template` - affects reproducibility
7. **Missing Alias Validation**: CLI has `fetch_score` vs `fetch-score` aliasing but documentation doesn't explain this - causes confusion
8. **Incomplete Artifact Structure**: README shows `.csv` extensions but code uses `.csv.gz` compression - documentation outdated
9. **State File Module Names**: Documentation shows `"modules": {"model": {...}}` but preprocessing chains use step names not "preprocess" - incomplete explanation
10. **Git Commit Format**: Auto-flow creates commits with specific format but README doesn't document this behavior - missing feature documentation
11. **Lock File Behavior**: `lock=true` creates `overwrite.lock` but deletion instructions only in MLA_WORKFLOW_GUIDE not README - incomplete cross-reference
12. **Missing `mla_retention` Documentation**: configs.md mentions it but no detailed explanation in model module docs - incomplete parameter documentation

---

## 2) Detailed Findings

| ID | Severity | Area | Description | Code Evidence | Docs Evidence | Fix |
|:---|:---------|:-----|:------------|:--------------|:--------------|:----|
| 001 | **High** | CLI | Missing `--skip-score-fetch` flag | `src/mlarena/cli/main.py:520` only has `skip_submit`, no `skip_score_fetch` | `README.md:522` "Control Flags" section lists `--skip-score-fetch` | Remove `--skip-score-fetch` from README or implement in CLI |
| 002 | **High** | CLI | Template flag format mismatch | `src/mlarena/cli/main.py:683-736` converts `--flag value` to `key=value` but expects dotted paths | `docs/MLA_WORKFLOW_GUIDE.md:202-203` shows `--preprocess-template <template-name>` | Update docs to show `preprocess_template=name` format |
| 003 | **High** | Templates | HPO template name mismatch | `src/mlarena/templates/model/hpo/` contains `hpo_boost_*.yaml` not `test_hpo_*.yaml` | `README.md:212-217`, `docs/MLA_WORKFLOW_GUIDE.md:427-441` use `test_hpo_medium` | Update all docs to use correct `hpo_boost_*` names |
| 004 | **Medium** | Architecture | Preprocessing path incorrect | Actual location: `src/mlarena/defaults/preprocessing/TEMPLATE.py` | `AGENTS.md:23` says `src/mlarena/preprocessing/TEMPLATE.py` | Fix AGENTS.md path reference |
| 005 | **Medium** | CLI | Queue command implementation differs | `src/mlarena/cli/main.py:810-819` delegates to `subprocess.run(["python", "scripts/task_queue.py"])` | `README.md:166-178` shows `mla queue` as native command | Add note that queue is delegated to separate script |
| 006 | **Medium** | Templates | Compressed CSV not documented | `src/mlarena/modules/model.py:48-59` uses `.csv.gz` extensions | `README.md:352` shows `.csv` in artifact examples | Update artifact structure examples to show `.csv.gz` |
| 007 | **Medium** | State | Chain module naming incomplete | `src/mlarena/cli/main.py:69-72` uses step names not "preprocess" | `README.md:436-463` shows "modules": {"model": {...}} but no chain explanation | Add chain state format documentation |
| 008 | **Medium** | Config | Missing parameter docs | `src/mlarena/core/conf.py:38-42` defines `lock`, `skip_deps`, `show_payload`, `model.mla_retention` | `docs/configs.md:38-39` mentions `lock` and `mla_retention` briefly | Add comprehensive parameter reference |
| 009 | **Low** | CLI | Flag-to-override conversion undocumented | `src/mlarena/cli/main.py:678-736` implements sophisticated flag conversion | No documentation explains `--time-limit 300` vs `common.time_limit=300` | Add section explaining CLI parsing behavior |
| 010 | **Low** | Git | Auto-commit format not documented | `src/mlarena/cli/main.py:596-676` creates structured commit messages | `README.md` mentions git tracking but not auto-commit format | Document commit message format |
| 011 | **Low** | Terminology | Inconsistent naming | Code uses both `preprocess_template` (Python), `preprocess-template` (CLI), `preprocess` (module name) | Multiple docs mix formats | Establish terminology guide |
| 012 | **Low** | State | Preprocessing payload structure varies | `src/mlarena/modules/model.py:67-76` handles multiple payload structures | `README.md:436-463` shows single structure | Document all payload format variations |
| 013 | **Critical** | Examples | Non-executable command format | Many examples show flags that won't parse correctly | `docs/MLA_WORKFLOW_GUIDE.md:129-134` shows `model_template=cpu-fast-1m` mixed with `--project` | Normalize all example commands |
| 014 | **Medium** | Templates | Meta-template chain resolution unclear | `src/mlarena/cli/main.py:29-73` implements complex chain resolution | `docs/submodules/README.md:272-301` shows basic chain example | Document chain resolution algorithm |
| 015 | **Medium** | Modules | Preprocessing contract signature change | `src/mlarena/defaults/preprocessing/imputer.py:43-49` takes 5 parameters including `orig_df` | `docs/submodules/README.md:23` shows 4-parameter signature | Update contract documentation |
| 016 | **Low** | Architecture | Missing auto-flow validation details | `src/mlarena/cli/main.py:76-133` validates init/eda completion | `README.md:36-58` mentions auto-flow but not validation | Document prerequisite validation |
| 017 | **Low** | Config | Profile fallback behavior | `src/mlarena/core/conf.py:105-109` has hardcoded smoke/dev profiles | `docs/configs.md:55-68` doesn't mention fallback | Document built-in profile fallbacks |
| 018 | **Medium** | Submit | Queue submission behavior differs | `scripts/submission_queue.py` is separate script with different API | `README.md:123-160` mixes queue and submit commands | Separate submission queue docs from task queue |
| 019 | **Low** | Templates | Global vs project precedence unclear | `src/mlarena/core/conf.py:71-79` checks project first then global | `docs/architecture.md:43` mentions precedence but not resolution order | Document complete resolution order |
| 020 | **Medium** | EDA | Module notes parameter | No evidence of `--eda-notes` parameter in code | `docs/MLA_WORKFLOW_GUIDE.md:99` shows `--eda-notes "Initial exploration"` | Remove or implement --eda-notes flag |

---

## 3) Proposed Fixes (Patch Plan)

### Priority 1: Critical Command Examples (IDs: 001, 002, 003, 013)

**Files to update:**
- `README.md`
- `docs/MLA_WORKFLOW_GUIDE.md`
- `docs/quick_start.md` (if exists)

**Changes:**

#### README.md
```diff
- Line 522: Remove --skip-score-fetch from "Control Flags" section
+ Add note: "Score fetching is controlled by skip_submit flag and wait_seconds parameter"

- Lines 203-217: Replace all --preprocess-template with preprocess_template=
Example:
- uv run python scripts/mla.py model --project titanic --preprocess-template my-preprocess
+ uv run python scripts/mla.py model --project titanic preprocess_template=my-preprocess

- Lines 212-217: Replace test_hpo_* with hpo_boost_*
- uv run python scripts/mla.py model --project titanic --model-template test_hpo_medium
+ uv run python scripts/mla.py model --project titanic model_template=hpo_boost_medium
```

#### docs/MLA_WORKFLOW_GUIDE.md
```diff
- Lines 127-134: Standardize all command examples to use dotted overrides
- Lines 427-441: Update HPO template names to hpo_boost_*
- Line 99: Remove --eda-notes or document if implemented
- Line 202-203: Change --preprocess-template to preprocess_template=
```

### Priority 2: Architecture Documentation (IDs: 004, 006, 007, 014, 015)

**Files to update:**
- `AGENTS.md`
- `docs/submodules/README.md`
- `README.md` (artifact structure)

**Changes:**

#### AGENTS.md
```diff
Line 23:
- Copy `src/mlarena/preprocessing/TEMPLATE.py` to `src/mlarena/defaults/preprocessing/`.
+ Copy `src/mlarena/defaults/preprocessing/TEMPLATE.py` to `src/mlarena/defaults/preprocessing/your_module.py`.
```

#### README.md (Artifact Structure)
```diff
Lines 350-355:
- ├── train_processed.csv
- ├── test_processed.csv
+ ├── train_processed.csv.gz
+ ├── test_processed.csv.gz
+ ├── orig_processed.csv.gz (optional)
```

#### docs/submodules/README.md
```diff
Line 23 (contract signature):
- Follow the template structure: fit_transform(train, val, test, config)
+ Follow the template structure: fit_transform(train, val, test, config, orig_df=None)

Add section "Chain State Format":
+## Chain State Format
+
+When preprocessing chains execute, each step creates its own state entry:
+```json
+{
+  "experiment_id": "pre-my_pipeline/abc123def/1-imputer",
+  "modules": {
+    "preprocess": {  // Note: Still uses "preprocess" as module name
+      "status": "completed",
+      "payload": {...}
+    }
+  }
+}
+```
+
+For the final chain output, query the last step's state.json.
```

### Priority 3: Configuration System (IDs: 008, 009, 017, 019)

**Files to update:**
- `docs/configs.md`
- `README.md` (add CLI parsing section)

**Changes:**

#### docs/configs.md
```diff
Add after line 42:
+### Complete Parameter Reference
+
+| Parameter | Type | Default | Description |
+|:----------|:-----|:--------|:------------|
+| lock | bool | false | Create overwrite.lock after successful completion to prevent re-runs |
+| skip_deps | bool | false | Skip dependency resolution (run only target module) |
+| show_payload | bool | false | Display module output payload in console |
+| model.mla_retention | bool | false | Clean up AutoGluon intermediate models after training (saves disk space) |
+
+Add section "Profile Fallback Behavior":
+### Built-in Profile Fallbacks
+
+If profile YAML files don't exist, the system provides hardcoded fallbacks for:
+- `smoke`: `{common: {time_limit: 60, preset: "medium", use_gpu: false}}`
+- `dev`: `{common: {time_limit: 300, preset: "high", use_gpu: false}}`
```

#### README.md
```diff
Add new section after "Configuration Overrides" (line 236):
+### CLI Parsing Behavior
+
+MLArena supports two parameter formats:
+
+1. **Dotted paths** (recommended): `key.subkey=value`
+   ```bash
+   uv run python scripts/mla.py --project titanic common.time_limit=600
+   ```
+
+2. **Flag format** (converted internally): `--flag value`
+   ```bash
+   uv run python scripts/mla.py --project titanic --time-limit 600
+   # Internally converted to: common.time_limit=600
+   ```
+
+**Note**: Common parameters (`time_limit`, `use_gpu`, `preset`, `seed`) are automatically prefixed with `common.` when using flag format.
```

### Priority 4: Template System (IDs: 014, 018, 019)

**Files to update:**
- `docs/architecture.md`
- `README.md` (separate queue documentation)

**Changes:**

#### docs/architecture.md
```diff
Add after line 46:
+## Template Resolution Details
+
+### Precedence Order (highest to lowest):
+1. CLI dotted overrides (`key=value`)
+2. Template config (within YAML)
+3. Project config (`projects/kaggle/<slug>/config.yaml`)
+4. Profile (`templates/profiles/<name>.yaml`)
+5. Hardcoded defaults (`src/mlarena/core/conf.py`)
+
+### Search Order for Templates:
+1. Project-local: `projects/kaggle/<slug>/templates/{model|preprocess}/<name>.yaml`
+2. Global: `src/mlarena/templates/{model|preprocess}/<name>.yaml`
+
+### Chain Resolution Algorithm:
+1. Parse template argument (single name or comma-separated list)
+2. Load template config from precedence order
+3. Check for `chain` key in config
+4. If `chain` exists: expand to list of template names
+5. If comma-separated: use as-is
+6. For each template in list:
+   - Load individual config
+   - Compute semantic hash
+7. Combine hashes to generate chain experiment ID
+8. Create directory: `pre-{chain_id}/{combined_hash}/{idx}-{template}`
```

#### README.md
```diff
Section "Submission Queue (Manual Script)" (lines 123-160):
- Move entire section to new file: docs/submission_queue.md
+ Add link: "For submission queue management, see [Submission Queue Guide](docs/submission_queue.md)"

Section "Task Queue Management" (lines 162-184):
+ Add clarification:
+**Note**: Task Queue manages computation tasks (training, preprocessing).
+For submission uploads, see Submission Queue (separate system).
```

### Priority 5: Missing Documentation (IDs: 010, 016, 020)

**Files to update:**
- `README.md`
- `docs/MLA_WORKFLOW_GUIDE.md`

**Changes:**

#### README.md
```diff
Add section after "Experiment Tracking and Reproducibility" (line 505):
+### Auto-Flow Git Commits
+
+When auto-flow completes successfully (unless `skip_git=true`), a git commit is created automatically:
+
+**Commit message format:**
+```
+auto-flow({project}): {module1}→{module2}→... | local {cv_score} | public {score}
+```
+
+**Example:**
+```
+auto-flow(titanic): preprocess→model→predict→submit→fetch-score | local 0.834 | public 0.798
+```
+
+**What gets staged:**
+- Project directory: `projects/kaggle/{project}/`
+- Experiments, submissions, and templates
+
+**Skip auto-commit:**
+```bash
+uv run python scripts/mla.py --project titanic skip_git=true
+```
```

#### docs/MLA_WORKFLOW_GUIDE.md
```diff
Add section after "Prerequisites" (line 45):
+### Auto-Flow Validation
+
+Before executing auto-flow, the system validates:
+
+1. **Init module completed**: Checks `experiments/init/state.json` status
+2. **EDA module completed**: Checks `experiments/eda/state.json` status
+
+If either validation fails, you'll see:
+```
+✗ Prerequisites validation failed:
+
+Project initialization not found.
+Run: mla init --project {project}
+```
+
+**Note**: These modules must be run manually before auto-flow.
```

---

## 4) Documentation Gaps

### Undocumented Code Features

1. **Preprocessing Chain Caching** (`src/mlarena/cli/main.py:264-367`)
   - Content-addressed caching with semantic hashing
   - Cache hit detection and resume logic
   - Cache validation across chain steps
   - **Impact**: Users unaware of caching may force re-runs unnecessarily
   - **Recommendation**: Add `docs/caching.md` explaining hash-based caching

2. **Lock File Mechanism** (`src/mlarena/cli/main.py:401`, `lock=true`)
   - Creates `overwrite.lock` in experiment directory
   - Prevents accidental overwriting of valuable experiments
   - **Impact**: Users may not know how to protect long-running experiments
   - **Recommendation**: Add prominent note in README about `lock=true` parameter

3. **Auto-Flow Commit Format** (`src/mlarena/cli/main.py:596-676`)
   - Structured commit message with scores
   - Automatic staging of project directory
   - Skip on no changes
   - **Impact**: Users see commits but don't understand format or how to disable
   - **Recommendation**: Document in README as shown in patch plan

4. **Preprocessing Module Ambiguity Detection** (`src/mlarena/modules/preprocess.py:116-123`)
   - Fails if module exists in both project and global
   - Forces explicit choice by user
   - **Impact**: Users may encounter confusing errors
   - **Recommendation**: Document in docs/submodules/README.md

5. **State File Locking** (`src/mlarena/core/experiment.py`, referenced but not shown)
   - File locking for concurrent execution safety
   - **Impact**: Users may not understand why writes are blocked
   - **Recommendation**: Add note in architecture.md

6. **Git Hash Snapshots** (`README.md:477-479` claims feature but no code evidence in files read)
   - Need to verify if actually implemented
   - **Impact**: May be documented but not working
   - **Recommendation**: Verify implementation or remove from docs

### Deprecated/Misleading Documentation

1. **`--eda-notes` parameter** (`docs/MLA_WORKFLOW_GUIDE.md:99`)
   - Documented but no code implementation found
   - **Fix**: Remove from examples or implement feature

2. **`--skip-score-fetch` flag** (`README.md:522`)
   - Documented but doesn't exist in CLI
   - **Fix**: Remove or implement

3. **Old artifact paths** (`README.md:352`)
   - Shows `.csv` but code uses `.csv.gz`
   - **Fix**: Update all artifact structure examples

4. **Template TEMPLATE.py location** (`AGENTS.md:23`)
   - Wrong path reference
   - **Fix**: Correct to `src/mlarena/defaults/preprocessing/`

### Missing Cross-References

1. `lock=true` mentioned in configs.md but deletion instructions only in MLA_WORKFLOW_GUIDE
2. Preprocessing chain format in submodules/README.md but no link from main README
3. HPO templates in configs.md but examples split across README and MLA_WORKFLOW_GUIDE
4. Submission queue in README but implementation in separate script (no architecture explanation)

**Recommendation**: Add "See Also" sections linking related documentation

---

## 5) Terminology Inconsistencies

| Term Variations | Files | Standard | Fix |
|:----------------|:------|:---------|:----|
| `preprocess_template` vs `preprocess-template` vs `--preprocess-template` | README, MLA_WORKFLOW_GUIDE, configs.md | `preprocess_template` (Python), `preprocess-template` (YAML keys) | Create terminology guide |
| `experiment_id` vs `exp-id` | Multiple | `experiment_id` (code), can use either in CLI | Document dash-underscore equivalence |
| `fetch-score` vs `fetch_score` | README, modules/ | `fetch-score` (CLI), `fetch_score` (Python) | Document naming convention |
| `MLArena` vs `mlarena` vs `mla` | Multiple | `mlarena` (package), `mla` (CLI), `MLArena` (prose) | Establish brand guide |
| `AutoGluon` vs `autogluon` | Multiple | `AutoGluon` (product name), `autogluon` (import) | Use proper casing |
| `template` vs `template_name` vs `template-name` | Multiple | Context-dependent | Document when to use which |

**Recommendation**: Create `docs/TERMINOLOGY.md` with canonical names and usage rules

---

## 6) Test-as-Doc Recommendations

To prevent future documentation drift, implement executable doc tests:

### Minimal Test Suite

```bash
# File: tests/docs/test_readme_examples.sh

#!/bin/bash
# Test all README.md command examples

# Example 1: Quick Start auto-flow (smoke test with mock data)
uv run python scripts/mla.py init --project test-project-readme
uv run python scripts/mla.py eda --project test-project-readme
uv run python scripts/mla.py --project test-project-readme profile=smoke skip_submit=true

# Example 2: Manual workflow
uv run python scripts/mla.py model --project test-project-readme model_template=cpu-fast-1m skip_submit=true

# Example 3: Configuration overrides
uv run python scripts/mla.py model --project test-project-readme common.time_limit=30 force=true

# Example 4: Queue commands
uv run python scripts/mla.py queue list -p test-project-readme

# Cleanup
rm -rf projects/kaggle/test-project-readme
```

### CI Integration

```yaml
# .github/workflows/docs-validation.yml
name: Documentation Validation

on: [push, pull_request]

jobs:
  test-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: uv sync
      - name: Test README examples
        run: bash tests/docs/test_readme_examples.sh
      - name: Test MLA_WORKFLOW_GUIDE examples
        run: bash tests/docs/test_workflow_guide.sh
```

---

## 7) Priority Matrix

| Priority | IDs | Estimated Effort | Impact | Risk if Unfixed |
|:---------|:----|:-----------------|:-------|:----------------|
| **P0 - Immediate** | 001, 002, 003, 013 | 2-3 hours | Critical | Users cannot reproduce examples |
| **P1 - High** | 004, 006, 007, 015 | 3-4 hours | High | Users confused by architecture |
| **P2 - Medium** | 005, 008, 014, 018, 019 | 4-6 hours | Medium | Missing context slows onboarding |
| **P3 - Low** | 009, 010, 011, 012, 016, 017, 020 | 2-3 hours | Low | Minor friction, workarounds exist |
| **P4 - Polish** | Terminology, cross-refs | 4-5 hours | Low | Professional consistency |

**Total estimated effort**: 15-21 hours

---

## 8) Recommended Workflow

### Phase 1: Quick Wins (P0, 2-3 hours)
1. Update README.md examples to use correct template names
2. Remove `--skip-score-fetch` reference
3. Fix all `--preprocess-template` to `preprocess_template=`
4. Update MLA_WORKFLOW_GUIDE.md with corrected examples

### Phase 2: Architecture Fixes (P1, 3-4 hours)
1. Fix AGENTS.md preprocessing path
2. Update artifact structure examples to show `.csv.gz`
3. Document preprocessing chain state format
4. Update preprocessing contract signature

### Phase 3: Configuration (P2, 4-6 hours)
1. Expand configs.md with complete parameter reference
2. Document CLI parsing behavior
3. Add template resolution details to architecture.md
4. Separate submission queue from task queue docs

### Phase 4: Polish (P3+P4, 6-8 hours)
1. Document auto-flow commit format
2. Add prerequisite validation section
3. Create TERMINOLOGY.md
4. Add cross-references between docs
5. Create docs/caching.md

### Phase 5: Validation (ongoing)
1. Implement test-as-doc suite
2. Set up CI to run doc tests
3. Add pre-commit hook to validate examples

---

## Appendix A: Evidence Index

### Code References
- CLI implementation: `src/mlarena/cli/main.py:1-1000`
- Config system: `src/mlarena/core/conf.py:1-136`
- Preprocessing contract: `src/mlarena/defaults/preprocessing/imputer.py:43-49`
- Model module: `src/mlarena/modules/model.py:1-150`
- Preprocess module: `src/mlarena/modules/preprocess.py:1-150`

### Documentation References
- Main README: `README.md:1-524`
- Workflow Guide: `docs/MLA_WORKFLOW_GUIDE.md:1-618`
- Architecture: `docs/architecture.md:1-50`
- Configs: `docs/configs.md:1-115`
- Agent Guide: `AGENTS.md:1-68`
- Submodules Guide: `docs/submodules/README.md:1-396`

### Template References
- Global templates: `src/mlarena/templates/`
- Model templates: `src/mlarena/templates/model/*.yaml`
- Preprocess templates: `src/mlarena/templates/preprocess/*.yaml`
- HPO templates: `src/mlarena/templates/model/hpo/*.yaml`

---

## Appendix B: Verification Commands

Run these to verify fixes:

```bash
# Test template name corrections
uv run python scripts/mla.py model --project Titanic model_template=hpo_boost_medium skip_submit=true

# Test dotted override format
uv run python scripts/mla.py model --project Titanic common.time_limit=60 force=true

# Verify queue delegation works
uv run python scripts/mla.py queue list -p Titanic

# Test preprocessing chain
uv run python scripts/mla.py preprocess --project Titanic preprocess_template=baseline

# Verify lock file creation
uv run python scripts/mla.py model --project Titanic lock=true
# Check: ls projects/kaggle/Titanic/experiments/exp-*/overwrite.lock

# Test auto-commit
uv run python scripts/mla.py --project Titanic profile=smoke skip_git=false
# Check: git log -1 --format='%s'
```

---

**End of Report**
