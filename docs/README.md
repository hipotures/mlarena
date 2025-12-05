# Documentation Index

Complete guide to all documentation in the Kaggle competitions repository.

## Quick Navigation

### For New Users
Start here to get up and running quickly:
1. **[Main README](../README.md)** - Repository overview and quick start
2. **[CLAUDE.md](../CLAUDE.md)** - Comprehensive guide for Claude Code (architecture, commands, workflows)
3. **[scripts/README.md](../scripts/README.md)** - All available scripts and their usage

### For Specific Tasks
- **Running experiments:** [CLAUDE.md](../CLAUDE.md) → Modern Workflow section
- **Hyperparameter tuning:** [OPTUNA_GUIDE.md](OPTUNA_GUIDE.md)
- **Template configuration:** [configs.md](configs.md)
- **Troubleshooting:** [CLAUDE.md](../CLAUDE.md) → Common Pitfalls section

## Primary Documentation

### Main Guides

#### [CLAUDE.md](../CLAUDE.md) (22 KB)
**The primary comprehensive guide.** Covers:
- Repository architecture (four-layer system)
- Common commands and workflows
- Modular experiment pipeline
- Project creation and configuration
- Automated submission & score fetching
- Development guidelines and best practices
- **Audience:** All users, especially those using Claude Code

#### [README.md](../README.md) (18 KB)
**Repository overview and quick start.**
- Project structure
- Quick start commands
- Optuna system overview
- Submission tracking
- Active competitions
- **Audience:** First-time users, quick reference

### Specialized Guides

#### [OPTUNA_GUIDE.md](OPTUNA_GUIDE.md) (796 lines)
**Complete hyperparameter tuning system guide.**
- Feature engineering with data leakage protection
- Hyperparameter tuning (XGBoost, LightGBM, CatBoost)
- Model ensembling and stacking
- Complete workflow examples
- Configuration reference
- **Audience:** Users doing hyperparameter optimization

#### [configs.md](configs.md) (397 lines)
**Template configuration reference.**
- `templates/model.yaml` structure
- `templates/preprocess.yaml` structure
- Hyperparameter configuration
- Feature engineering templates
- Global vs project template resolution
- **Audience:** Users creating custom templates

## Scripts Documentation

### [scripts/README.md](../scripts/README.md)
**All scripts overview with usage examples.**
- User-facing scripts (experiment_manager, autogluon_runner, etc.)
- Infrastructure scripts (template_loader, pipeline_loader, etc.)
- Complete script categories
- **Audience:** Users running experiments from command line

### [scripts/README_KAGGLE.md](../scripts/README_KAGGLE.md)
**Kaggle scraper automation guide.**
- Chrome CDP setup
- Score fetching automation
- Troubleshooting scraper issues
- **Audience:** Users automating Kaggle interactions

## Design Documents

These documents explain the architecture and design decisions:

### [ml_code_separation_design_v2.md](ml_code_separation_design_v2.md) (188 lines)
**Current ML code architecture.**
- Separation of ML code from infrastructure
- Clean model interfaces
- Template-based configuration
- **Status:** Current version

### [template_system_redesign.md](template_system_redesign.md) (85 lines)
**Template system design and roadmap.**
- Dual template system (model + preprocess)
- Module contracts
- Planned enhancements
- **Status:** Partially implemented

### [template_merge_guidelines.md](template_merge_guidelines.md) (46 lines)
**Template resolution guidelines.**
- Global vs project template merging
- Override behavior
- Warning policy
- **Audience:** Developers working with templates

## Development Files

### [TODO.md](TODO.md)
**Active development tasks.**
- Experiment manager enhancements
- Planned features
- **Note:** Consider moving to GitHub Issues

## Archived Documentation

### [archive/](archive/)
Documentation that has been superseded or is no longer actively maintained:
- `ml_code_separation_design_v1.md` - Original design (superseded by v2)
- `codex1.md` - AI feedback on v1 design
- `gemini1.md` - AI feedback on v1 design

See [archive/README.md](archive/README.md) for details.

## Documentation by Use Case

### "I want to run my first experiment"
1. [README.md](../README.md) - Quick Start section
2. [CLAUDE.md](../CLAUDE.md) - Modern Workflow section
3. [scripts/README.md](../scripts/README.md) - experiment_manager.py

### "I want to tune hyperparameters"
1. [OPTUNA_GUIDE.md](OPTUNA_GUIDE.md) - Quick Start section
2. [configs.md](configs.md) - Template configuration
3. [CLAUDE.md](../CLAUDE.md) - Optuna system overview

### "I want to create custom templates"
1. [configs.md](configs.md) - Complete template reference
2. [template_system_redesign.md](template_system_redesign.md) - Design overview
3. [template_merge_guidelines.md](template_merge_guidelines.md) - Merge behavior

### "I want to understand the architecture"
1. [CLAUDE.md](../CLAUDE.md) - Four-layer system
2. [ml_code_separation_design_v2.md](ml_code_separation_design_v2.md) - ML code patterns
3. [template_system_redesign.md](template_system_redesign.md) - Template architecture

### "I'm having problems"
1. [CLAUDE.md](../CLAUDE.md) - Common Pitfalls section
2. [OPTUNA_GUIDE.md](OPTUNA_GUIDE.md) - Troubleshooting section
3. [scripts/README_KAGGLE.md](../scripts/README_KAGGLE.md) - Scraper troubleshooting

## File Organization

```
/mnt/ml/kaggle/
├── README.md                          # Repository overview
├── CLAUDE.md (→ AGENTS.md)            # Main comprehensive guide
├── AGENTS.md                          # Primary documentation (CLAUDE.md is symlink)
├── docs/
│   ├── README.md                      # This index file
│   ├── OPTUNA_GUIDE.md                # Hyperparameter tuning guide
│   ├── configs.md                     # Template configuration
│   ├── ml_code_separation_design_v2.md # Architecture
│   ├── template_system_redesign.md    # Design doc
│   ├── template_merge_guidelines.md   # Guidelines
│   ├── TODO.md                        # Development tasks
│   └── archive/                       # Archived documentation
│       ├── README.md
│       ├── ml_code_separation_design_v1.md
│       ├── codex1.md
│       └── gemini1.md
└── scripts/
    ├── README.md                      # Scripts overview
    └── README_KAGGLE.md               # Kaggle scraper guide
```

## Contributing to Documentation

When updating documentation:
1. **Keep README.md and CLAUDE.md in sync** - They share common content
2. **Update this index** if adding new documentation files
3. **Add cross-references** between related documents
4. **Follow the style** - Use clear headers, code blocks, and examples
5. **Test examples** - Ensure all command examples are accurate

## Documentation Standards

- **Use markdown** for all documentation
- **Include table of contents** for files >500 lines
- **Use code blocks** with language tags for syntax highlighting
- **Add cross-references** to related documentation
- **Update timestamps** when making significant changes
- **Keep examples current** with actual code structure

---

**Last updated:** Documentation reorganization completed December 2025

**Maintainers:** See main [README.md](../README.md) for contact information
