# Thermophysical Property: Melting Point

## Competition Overview
**Competition URL:** https://www.kaggle.com/competitions/melting-point
**Type:** Community Prediction Competition
**Host:** John Hedengren
**Course:** [Machine Learning for Engineers](https://apmonitor.com/pds) - [Thermophysical Properties](https://apmonitor.com/pds/index.php/Main/ThermophysicalProperties)

### Objective
Build ML models that predict melting point (Kelvin) for organic compounds given molecular descriptors and group contribution features.

### Evaluation Metric
**MAE (Mean Absolute Error)** - Lower is better

### Dataset Description
**Challenge:** Predict melting points from group contribution features (subgroup counts representing functional groups within molecules).

**Size:** 2.93 MB (3 CSV files)
**Total compounds:** 3,328
**Training samples:** 2,662 (80%)
**Test samples:** 666 (20%)
**License:** MIT

**Features (854 columns):**
- `id` - Unique compound identifier
- `SMILES` - Molecular string representation
- `Group 1` through `Group 424` - Molecular descriptor features (group contribution counts)

**Target:**
- `Tm` - Melting point in Kelvin (train only)

**Domain:** Chemistry and chemical engineering - critical for drug design, material selection, and process safety.

**Tags:** Mean Absolute Error

## Progress Tracking

### Best Submissions
| Date | Score | Model | Notes |
|------|-------|-------|-------|
| - | - | - | - |

### Experiments Log
| ID | Date | Configuration | Score | Status |
|----|------|---------------|-------|--------|
| - | - | - | - | - |

## Setup

### Download Data
```bash
cd data/
kaggle competitions download -c melting-point
unzip melting-point.zip
```

### Environment
```bash
# Add required packages to pyproject.toml
```

## Competition Stats
- **Entrants:** 2,116
- **Participants:** 633
- **Teams:** 605
- **Submissions:** 3,630+
- **Prizes:** Kudos (no points/medals)

## Submission Format
CSV file with columns: `id`, `Tm`

Example:
```csv
id,Tm
210,350.1
211,583.8
...
```

## Key Challenge
Build models that:
- Capture complex, nonlinear relationships between molecular structure and melting behavior
- Generalize across diverse chemical families
- Push limits of data-driven property prediction

## Notes
- Competition accepted: 2025-11-16
- Melting point measurements are often costly, time-consuming, or unavailable
- Group contribution method: functional groups within molecules represented as features

## TODO
- [x] Download and explore dataset
- [ ] Baseline model
- [ ] Feature engineering
- [ ] Model optimization
- [ ] Ensemble methods
