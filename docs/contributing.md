# Contributing to MLArena

Thank you for your interest in contributing to MLArena! This guide will help you get started.

**For documentation standards, see:** [Terminology Guide](TERMINOLOGY.md)

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Adding New Features](#adding-new-features)

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Git
- Kaggle API credentials (for testing)

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/hipotures/mlarena.git
cd mlarena

# Install dependencies
uv sync

# Install development tools
uv run playwright install chromium

# Verify installation
uv run python scripts/mla.py --help
```

---

## Development Workflow

### Branch Strategy

```bash
# Create feature branch
git checkout -b feat/your-feature-name

# Create bugfix branch
git checkout -b fix/issue-description

# Create docs branch
git checkout -b docs/what-you-document
```

### Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples:**
```bash
feat(model): add XGBoost ensemble support
fix(submit): handle missing sample_submission.csv
docs(preprocessing): update chain resolution algorithm
refactor(pipeline): simplify dependency resolution
test(eda): add integration tests for ydata-profiling
```

---

## Coding Standards

### Python Style

Follow **PEP 8** with these specifics:

1. **Line length**: 100 characters (not 80)
2. **Imports**: Organized in groups (stdlib, third-party, local)
3. **Type hints**: Required for all public functions
4. **Docstrings**: Google-style for all modules, classes, and public functions

**Example:**
```python
"""Module docstring describing what this module does."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from rich.console import Console

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry


logger = logging.getLogger(__name__)
console = Console()


@ModuleRegistry.register
class MyModule(BaseModule):
    """One-line summary of the module.

    Longer description if needed, explaining what this module does,
    when to use it, and any important behavior.

    Attributes:
        name: Module identifier used in CLI
        description: Human-readable description
    """

    name = "my-module"
    description = "Brief description of what this module does"

    def execute(self) -> ModuleResult:
        """Execute the module logic.

        Performs the main work of this module, including validation,
        processing, and artifact creation.

        Returns:
            ModuleResult containing execution status and outputs

        Raises:
            ValueError: If required configuration is missing
            RuntimeError: If execution fails

        Examples:
            >>> module = MyModule(context, config)
            >>> result = module.execute()
            >>> print(result.status)
            'completed'
        """
        # Implementation
        pass
```

### Naming Conventions

Follow [Terminology Guide](TERMINOLOGY.md) standards:

- **Python code**: `snake_case` for variables, functions
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Module files**: `snake_case.py`
- **YAML keys**: `kebab-case`

**Examples:**
```python
# Good
def load_processed_data(experiment_id: str) -> pd.DataFrame:
    TARGET_COLUMN = "Survived"
    class PreprocessModule(BaseModule):
        pass

# Bad
def LoadProcessedData(experimentId):  # Wrong casing
    targetColumn = "Survived"  # Should be constant
    class preprocess_module:  # Should be PascalCase
        pass
```

---

## Testing

### Running Tests

```bash
# Run all tests
uv run python -m pytest

# Run with coverage
uv run python -m pytest --cov=src/mlarena --cov-report=html

# Run specific test file
uv run python -m pytest tests/test_pipeline.py

# Run tests matching pattern
uv run python -m pytest -k "test_preprocessing"
```

### Integration Testing

Test with a real project using smoke mode:

```bash
# Quick integration test (~2 minutes)
uv run python scripts/mla.py project=Titanic profile=smoke skip_submit=true

# Full pipeline test
uv run python scripts/mla.py project=Titanic skip_submit=true
```

### Writing Tests

**Test file structure:**
```python
"""Tests for preprocessing module."""

import pytest
from pathlib import Path
import pandas as pd

from mlarena.modules.preprocess import PreprocessModule


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "feature": [10, 20, 30],
        "target": [0, 1, 0]
    })


def test_preprocess_basic(sample_data):
    """Test basic preprocessing execution."""
    # Arrange
    config = {"_artifact_dir": Path("/tmp/test")}

    # Act
    result = preprocess_function(sample_data, config)

    # Assert
    assert result is not None
    assert len(result) == len(sample_data)


def test_preprocess_missing_config():
    """Test that missing config raises ValueError."""
    with pytest.raises(ValueError, match="Missing required config"):
        preprocess_function(pd.DataFrame(), {})
```

---

## Documentation

### Documentation Standards

1. **All new modules**: Must have corresponding `.md` file in `docs/`
2. **All preprocessing submodules**: Must have entry in `docs/submodules/README.md`
3. **All parameters**: Must be documented in `docs/configs.md` if global
4. **All templates**: Must include YAML comments explaining parameters

### Updating Documentation

**When adding a feature:**
```bash
# 1. Update relevant docs
vim docs/MLA_WORKFLOW_GUIDE.md  # Add workflow example
vim docs/configs.md              # If new parameters
vim README.md                    # If major feature

# 2. Add to Documentation Index
vim README.md  # Update "Documentation Index" section

# 3. Verify links work
grep -r "your-new-doc.md" docs/  # Check cross-references
```

**Documentation checklist:**
- [ ] Feature documented in appropriate guide
- [ ] Examples are executable (test them!)
- [ ] Parameters explained with types and defaults
- [ ] Cross-references added where relevant
- [ ] Terminology follows [Terminology Guide](TERMINOLOGY.md)
- [ ] Code examples use correct naming conventions

### Writing Examples

**Good example:**
```markdown
### Using Custom Preprocessing

Create a custom preprocessing module:

1. Create file: `projects/kaggle/titanic/code/preprocessing/my_step.py`
2. Implement interface:
   ```python
   def fit_transform(train_df, val_df, test_df, config, orig_df=None):
       # Your logic here
       return train_df, val_df, test_df, orig_df, state_dict
   ```
3. Run preprocessing:
   ```bash
   uv run python scripts/mla.py preprocess project=titanic preprocess_template=my_step
   ```

**See:** [Preprocessing Submodules Guide](submodules/README.md)
```

**Bad example:**
```markdown
You can create custom preprocessing by making a file and running it.
```

---

## Submitting Changes

### Before Submitting

**Checklist:**
- [ ] Code follows [coding standards](#coding-standards)
- [ ] All tests pass (`uv run python -m pytest`)
- [ ] New features have tests
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No secrets or credentials in code
- [ ] `.gitignore` updated if needed

### Pull Request Process

1. **Push your branch:**
   ```bash
   git push origin feat/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title following commit convention
   - Description explaining what and why
   - Link to related issues
   - Screenshots/examples if UI-related

3. **PR Description Template:**
   ```markdown
   ## Summary
   Brief description of changes

   ## Changes
   - Added feature X
   - Fixed bug Y
   - Updated docs for Z

   ## Testing
   - [ ] All tests pass
   - [ ] Integration test with Titanic project
   - [ ] Manually verified feature works

   ## Documentation
   - [ ] Updated relevant docs
   - [ ] Added examples
   - [ ] Updated CHANGELOG (if applicable)

   ## Breaking Changes
   None / List any breaking changes

   Fixes #123
   ```

4. **Respond to review** - address feedback promptly

5. **After merge** - delete your feature branch

---

## Adding New Features

### Adding a New Module

1. **Create module file:**
   ```python
   # src/mlarena/modules/my_module.py

   from mlarena.core.module import BaseModule, ModuleResult
   from mlarena.core.registry import ModuleRegistry

   @ModuleRegistry.register
   class MyModule(BaseModule):
       name = "my-module"
       description = "What this module does"

       def can_run(self) -> tuple[bool, str]:
           """Validate before execution."""
           # Check prerequisites
           return True, ""

       def execute(self) -> ModuleResult:
           """Perform module work."""
           # Implementation
           return ModuleResult(status="completed", payload={})
   ```

2. **Add module tests:**
   ```python
   # tests/test_my_module.py

   def test_my_module_execution():
       # Test module works
       pass
   ```

3. **Document module:**
   ```markdown
   # docs/modules/my_module.md

   # My Module

   ## Overview
   ## Usage
   ## Parameters
   ## Examples
   ```

4. **Add to Documentation Index** in README.md

---

### Adding a Preprocessing Submodule

1. **Use template:**
   ```bash
   cp src/mlarena/defaults/preprocessing/TEMPLATE.py \
      src/mlarena/defaults/preprocessing/my_submodule.py
   ```

2. **Implement `fit_transform`:**
   ```python
   def fit_transform(
       train_df: pd.DataFrame,
       val_df: pd.DataFrame | None,
       test_df: pd.DataFrame,
       config: Dict[str, Any],
       orig_df: pd.DataFrame | None = None
   ) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
       # Implementation
       return train_df, val_df, test_df, orig_df, state_dict
   ```

3. **Create template:**
   ```yaml
   # src/mlarena/templates/preprocess/my_submodule.yaml
   module: my_submodule
   cache: true
   config:
     param1: default_value
     param2: default_value
   ```

4. **Document in `docs/submodules/my_submodule.md`:**
   - Parameters with types and defaults
   - Examples with YAML configs
   - Artifacts generated
   - Edge cases and recommendations

5. **Add to** `docs/submodules/README.md` index

**See:** [Preprocessing Submodules Guide](submodules/README.md#creating-a-new-sub-module)

---

### Adding a Model Template

1. **Create YAML file:**
   ```yaml
   # src/mlarena/templates/model/my_template.yaml
   model: autogluon_baseline  # or custom model name
   preprocess-template: baseline  # optional default preprocessing
   config:
     time_limit: 600
     preset: medium_quality
     hyperparameters:
       GBM:
         num_boost_round: 100
   ```

2. **If custom model, implement trainer:**
   ```python
   # src/mlarena/defaults/models/my_model.py

   def train(train_df, target_col, config, **kwargs):
       """Train custom model."""
       # Implementation
       return predictor, metrics
   ```

3. **Document in** `docs/model_templates.md`

4. **Test template:**
   ```bash
   uv run python scripts/mla.py model project=Titanic model_template=my_template skip_submit=true
   ```

---

## Code Review Guidelines

### For Contributors

**Before requesting review:**
- Self-review your changes
- Ensure all checks pass
- Provide context in PR description
- Link related issues/docs

### For Reviewers

**Focus on:**
1. **Correctness**: Does it work? Are edge cases handled?
2. **Design**: Is the approach sound? Could it be simpler?
3. **Testing**: Are tests adequate?
4. **Documentation**: Is it documented? Are examples clear?
5. **Style**: Does it follow conventions?
6. **Breaking changes**: Will this break existing code?

**Be constructive:**
- ✅ "Consider using X instead of Y because..."
- ✅ "This could be simplified by..."
- ✅ "Good approach! Minor suggestion: ..."
- ❌ "This is wrong"
- ❌ "Why didn't you use X?"

---

## Common Patterns

### Accessing Project Config

```python
from mlarena.utils.project import load_project_config

config = load_project_config(project_root)
target_column = config.get("TARGET_COLUMN")
id_column = config.get("ID_COLUMN", "id")
```

### Saving Artifacts

```python
from pathlib import Path

artifact_dir = Path(self.context.artifact_dir) / self.name
artifact_dir.mkdir(parents=True, exist_ok=True)

output_file = artifact_dir / "output.csv"
df.to_csv(output_file, index=False)
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Starting preprocessing")
logger.warning("Missing column: %s", col_name)
logger.error("Failed to process: %s", error)
```

### Rich Console Output

```python
from rich.console import Console
from rich.table import Table

console = Console()

table = Table(title="Results")
table.add_column("Metric")
table.add_column("Value")
table.add_row("CV Score", f"{cv_score:.4f}")

console.print(table)
```

---

## Avoiding Common Pitfalls

### ❌ Don't

```python
# Don't hardcode paths
output_path = "/home/user/mlarena/experiments/..."

# Don't modify global state
import sys
sys.path.append("/some/path")  # Use project imports instead

# Don't ignore errors silently
try:
    risky_operation()
except Exception:
    pass  # BAD: Swallows all errors

# Don't leak train/test information
test_df["new_feature"] = train_df["feature"].mean()  # Leakage!

# Don't use legacy flag naming
# Use: preprocess_template=
```

### ✅ Do

```python
# Use context-provided paths
output_path = self.context.artifact_dir / "output.csv"

# Use proper imports
from mlarena.utils.project import load_project_config

# Handle errors appropriately
try:
    risky_operation()
except ValueError as e:
    logger.error("Operation failed: %s", e)
    raise

# Compute features separately
train_mean = train_df["feature"].mean()
train_df["new_feature"] = train_df["feature"] - train_mean
test_df["new_feature"] = test_df["feature"] - train_mean

# Use correct conventions
preprocess_template=baseline
```

---

## Getting Help

**Questions?**
- Review existing code in `src/mlarena/`
- Check [Documentation Index](../README.md#documentation-index)
- Read [AGENTS.md](../AGENTS.md) for codebase navigation
- Ask in GitHub Discussions

**Found a bug?**
- Search existing issues
- Create detailed bug report with reproducible example
- Include Python version, OS, and error messages

**Want to discuss a feature?**
- Open a GitHub Discussion first
- Get feedback before implementing
- Propose API and design

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (check LICENSE file).

---

## Thank You!

Every contribution helps make MLArena better. We appreciate your time and effort! 🙏
