# Playground Series S5E11 - Predicting Loan Payback

## Competition Overview
**Competition URL:** https://www.kaggle.com/competitions/playground-series-s5e11
**Type:** Playground Prediction Competition (Kaggle)
**Host:** Kaggle
**Deadline:** 2025-11-30 23:59 UTC

### Objective
Predict the probability that a borrower will pay back their loan based on financial and demographic features.

### Evaluation Metric
**ROC AUC Score** - Area under the ROC curve between predicted probability and observed target.

### Dataset Description
**Source:** Synthetically generated from [Loan Prediction dataset](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025/data)
**Size:** 81.3 MB (3 CSV files)
**Training samples:** 593,994 rows
**Test samples:** ~198,000 rows (estimated from sample_submission)

**Features (12 predictors):**
- `annual_income` - Annual income of borrower
- `debt_to_income_ratio` - Debt to income ratio
- `credit_score` - Credit score
- `loan_amount` - Amount of loan requested
- `interest_rate` - Interest rate on loan
- `gender` - Gender (categorical)
- `marital_status` - Marital status (categorical)
- `education_level` - Education level (categorical: High School, Master's, etc.)
- `employment_status` - Employment status (categorical: Employed, Self-employed, etc.)
- `loan_purpose` - Purpose of loan (categorical: Debt consolidation, Other, etc.)
- `grade_subgrade` - Loan grade/subgrade (categorical: C3, D3, etc.)

**Target:**
- `loan_paid_back` - Binary (0/1) indicating if loan was paid back

**Tags:** Beginner, Tabular

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
kaggle competitions download -c playground-series-s5e11
unzip playground-series-s5e11.zip
```

### Environment
```bash
# Add required packages to pyproject.toml
```

## Competition Stats
- **Entrants:** 8,573
- **Participants:** 2,170
- **Teams:** 2,104
- **Submissions:** 15,225+
- **Prizes:** Swag (no points/medals)

## Submission Format
CSV file with columns: `id`, `loan_paid_back`

Example:
```csv
id,loan_paid_back
593994,0.5
593995,0.2
593996,0.1
```

## Notes
- Competition accepted: 2025-11-16
- Start date: 2025-11-01
- Dataset synthetically generated - feature distributions close to but not exactly the same as original
- Can use original dataset for training to improve performance

## TODO
- [x] Download and explore dataset
- [ ] Baseline model
- [ ] Feature engineering
- [ ] Model optimization
- [ ] Ensemble methods
