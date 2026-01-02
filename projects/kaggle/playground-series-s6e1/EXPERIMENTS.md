# DOE Experiments for playground-series-s6e1

**Generated**: 2026-01-01
**Total experiments**: 36
**Model**: GBM + XGB + CAT (BOOST only)
**Time limit**: 1 hour (3600s)
**HPO trials**: 50
**CPU-only**: Yes

---

## Experiment Table

| exp_id | parent_exp_id | model_template | preprocess_template | single_change | eda_rationale | expected_effect | risk_notes |
|--------|---------------|----------------|---------------------|---------------|---------------|-----------------|------------|
| E00 | - | 20260101120000_baseline | 20260101120000_baseline | Baseline: sanity_check only | Establish clean baseline with minimal preprocessing | RMSE ~8.6-8.7 (current baseline) | None - reference point |
| E01 | E00 | 20260101120100_onehot | 20260101120100_onehot | Encoding: ordinal → one-hot | Low cardinality (2-8 values) makes one-hot viable | +0.0 to -0.1 RMSE | 30+ new features from 8 categoricals |
| E02 | E00 | 20260101120200_target_enc | 20260101120200_target_enc | Encoding: ordinal → target_mean (5-fold OOF) | Capture categorical-target relationships | -0.1 to -0.2 RMSE | Overfitting risk if folds leak |
| E03 | E00 | 20260101120300_custom_feat | 20260101120300_custom_feat | Add: custom features (5 interactions) | Domain features (study_attendance_score, etc.) | -0.1 to -0.3 RMSE | May introduce noise if weak |
| E04 | E00 | 20260101120400_poly2 | 20260101120400_poly2 | Add: polynomial features (degree=2, numeric only) | Large dataset supports complexity | -0.1 to -0.2 RMSE | Feature explosion: 3→9 features |
| E05 | E00 | 20260101120500_standard_scale | 20260101120500_standard_scale | Add: standard scaling (numeric) | Test if scaling helps tree models | ±0.0 RMSE (control test) | Should have no effect |
| E06 | E00 | 20260101120600_robust_scale | 20260101120600_robust_scale | Add: robust scaling (numeric) | Test robust vs standard scaler | ±0.0 RMSE (control test) | Should have no effect |
| E07 | E00 | 20260101120700_log1p_target | 20260101120700_log1p_target | Add: log1p target transform | Stabilize variance for RMSE | -0.1 to -0.3 RMSE | Must inverse transform |
| E08 | E00 | 20260101120800_sqrt_target | 20260101120800_sqrt_target | Add: sqrt target transform | Milder variance stabilization | -0.1 to -0.2 RMSE | Must inverse transform |
| E09 | E00 | 20260101120900_rare_001 | 20260101120900_rare_001 | Add: rare category handler (0.01) | Edge case protection | ±0.0 to -0.05 RMSE | May bucket important categories |
| E10 | E00 | 20260101121000_rare_0005 | 20260101121000_rare_0005 | Add: rare category handler (0.005) | More aggressive bucketing | ±0.0 to -0.05 RMSE | May over-bucket |
| E11 | E00 | 20260101121100_var_thresh | 20260101121100_var_thresh | Add: variance threshold (0.01) | Remove low-variance features | ±0.0 RMSE | May remove valid features |
| E12 | E00 | 20260101121200_rfe_20 | 20260101121200_rfe_20 | Add: RFE feature selection (k=20) | Reduce overfitting | -0.05 to -0.15 RMSE | May discard useful features |
| E13 | E00 | 20260101121300_rfe_15 | 20260101121300_rfe_15 | Add: RFE feature selection (k=15) | More aggressive reduction | -0.05 to -0.1 RMSE | Higher discard risk |
| E14 | E00 | 20260101121400_feat_interact | 20260101121400_feat_interact | Feature interactions | Test specific feature combinations | -0.1 to -0.2 RMSE | May add noise |
| E15 | E00 | 20260101121500_datetime | 20260101121500_datetime | Datetime handler | Sanity check for temporal patterns | ±0.0 RMSE | Should be no-op |
| E16 | E00 | 20260101121600_outlier_iqr | 20260101121600_outlier_iqr | Outlier handler (IQR) | Test outlier capping | ±0.0 to -0.05 RMSE | May cap valid extremes |
| E17 | E00 | 20260101121700_target_smooth | 20260101121700_target_smooth | Target encoding with smoothing | Reduce overfitting | -0.1 to -0.2 RMSE | Smoothing may dilute signal |
| E18 | E00 | 20260101121800_custom_only | 20260101121800_custom_only | Custom features only | Test if engineered suffice | -0.2 to +0.1 RMSE | Risky - may lose signal |
| E19 | E00 | 20260101121900_poly3 | 20260101121900_poly3 | Polynomial degree=3 | Higher-order polynomials | -0.1 to +0.1 RMSE | Extreme explosion: 3→19 |
| E20 | E00 | 20260101122000_catboost_enc | 20260101122000_catboost_enc | CatBoost encoding | Alternative target encoding | -0.1 to -0.2 RMSE | Different regularization |
| E21 | E02 | 20260101122100_chain_a1 | 20260101122100_chain_a1 | Chain A: target_enc + custom_feat | Best encoding + domain features | -0.2 to -0.4 RMSE | Feature correlation |
| E22 | E21 | 20260101122200_chain_a2 | 20260101122200_chain_a2 | Chain A2: +log1p_target | Add target transform to A1 | -0.3 to -0.5 RMSE | Three changes total |
| E23 | E22 | 20260101122300_chain_a3 | 20260101122300_chain_a3 | Chain A3: +rfe_20 | Add feature selection to A2 | -0.3 to -0.5 RMSE | May remove engineered |
| E24 | E02 | 20260101122400_chain_b1 | 20260101122400_chain_b1 | Chain B1: target_enc+poly2 | Encoding + polynomial features | -0.2 to -0.4 RMSE | Many features |
| E25 | E24 | 20260101122500_chain_b2 | 20260101122500_chain_b2 | Chain B2: +feat_interact | Add interactions to B1 | -0.3 to -0.5 RMSE | High feature count |
| E26 | E25 | 20260101122600_chain_b3 | 20260101122600_chain_b3 | Chain B3: +rfe_20 | Add selection to B2 | -0.3 to -0.5 RMSE | Critical for overfitting |
| E27 | E03 | 20260101122700_chain_c1 | 20260101122700_chain_c1 | Chain C1: custom+target_enc | Features first, then encoding | -0.2 to -0.4 RMSE | Order sensitivity test |
| E28 | E27 | 20260101122800_chain_c2 | 20260101122800_chain_c2 | Chain C2: +sqrt_target | Add sqrt transform to C1 | -0.2 to -0.4 RMSE | Compare vs log (E22) |
| E29 | E28 | 20260101122900_chain_c3 | 20260101122900_chain_c3 | Chain C3: +rare_001 | Add rare bucketing to C2 | -0.2 to -0.4 RMSE | Edge case protection |
| E30 | E07 | 20260101123000_chain_d1 | 20260101123000_chain_d1 | Chain D1: log1p+target_enc | Transform-first approach | -0.2 to -0.4 RMSE | Transform→encode order |
| E31 | E30 | 20260101123100_chain_d2 | 20260101123100_chain_d2 | Chain D2: +custom_feat | Add features after transform+encode | -0.3 to -0.5 RMSE | Late feature engineering |
| E32 | E31 | 20260101123200_chain_d3 | 20260101123200_chain_d3 | Chain D3: +poly2 | Add polynomials to D2 | -0.3 to -0.5 RMSE | High complexity |
| E33 | E31 | 20260101123300_chain_d4 | 20260101123300_chain_d4 | Chain D4: +rfe_15 (vs poly) | Selection instead of poly | -0.3 to -0.5 RMSE | Compare vs E32 |
| E34 | E02 | 20260101123400_ultra | 20260101123400_ultra | Kitchen sink: all features | All best performers combined | -0.4 to -0.6 RMSE | High overfitting risk |
| E35 | E34 | 20260101123500_ultra_v2 | 20260101123500_ultra_v2 | Ultra v2: +rare_0005 | Final safety net | -0.4 to -0.6 RMSE | Maximum complexity |

---

## Execution Commands

### Baseline (E00)
```bash
uv run python scripts/mla.py --project playground-series-s6e1 --model-template 20260101120000_baseline skip_submit=true
```

### Single-Step Experiments (E01-E20)

All can run in parallel:

```bash
# Encoding variations
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101120100_onehot skip_submit=true &
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101120200_target_enc skip_submit=true &
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101120300_custom_feat skip_submit=true &

# Feature engineering
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101120400_poly2 skip_submit=true &

# Scaling (control tests)
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101120500_standard_scale skip_submit=true &
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101120600_robust_scale skip_submit=true &

# Target transforms
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101120700_log1p_target skip_submit=true &
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101120800_sqrt_target skip_submit=true &

# Rare categories
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101120900_rare_001 skip_submit=true &
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101121000_rare_0005 skip_submit=true &

# Feature selection
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101121100_var_thresh skip_submit=true &
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101121200_rfe_20 skip_submit=true &
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101121300_rfe_15 skip_submit=true &

# ... (E14-E20)

wait  # Wait for all parallel jobs to complete
```

### Chain Experiments (E21-E35)

Run sequentially within each chain:

**Chain A (E21-E23)**:
```bash
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101122100_chain_a1 skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101122200_chain_a2 skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101122300_chain_a3 skip_submit=true
```

**Chain B (E24-E26)**:
```bash
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101122400_chain_b1 skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101122500_chain_b2 skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101122600_chain_b3 skip_submit=true
```

**Chain C (E27-E29)**:
```bash
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101122700_chain_c1 skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101122800_chain_c2 skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101122900_chain_c3 skip_submit=true
```

**Chain D (E30-E33)**:
```bash
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101123000_chain_d1 skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101123100_chain_d2 skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101123200_chain_d3 skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101123300_chain_d4 skip_submit=true
```

**Kitchen Sink (E34-E35)**:
```bash
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101123400_ultra skip_submit=true
uv run python scripts/mla.py -p playground-series-s6e1 --model-template 20260101123500_ultra_v2 skip_submit=true
```

---

## Results Tracking

Create `results.csv` to track experiment outcomes:

```csv
exp_id,rmse_cv,rmse_cv_std,n_features,train_time_s,status,notes
E00,,,,,pending,
E01,,,,,pending,
E02,,,,,pending,
...
```

Update after each experiment completes.

---

## DAG Structure

```
E00 (baseline) ← ROOT
├── E01-E20 (single-step variations)
│   ├── E02 (target_enc) ← BEST SINGLE-STEP CANDIDATE
│   │   ├── E21 (Chain A1)
│   │   │   ├── E22 (Chain A2)
│   │   │   └── E23 (Chain A3)
│   │   ├── E24 (Chain B1)
│   │   │   ├── E25 (Chain B2)
│   │   │   └── E26 (Chain B3)
│   │   └── E34 (Kitchen Sink)
│   │       └── E35 (Ultra)
│   ├── E03 (custom_feat)
│   │   └── E27 (Chain C1)
│   │       ├── E28 (Chain C2)
│   │       └── E29 (Chain C3)
│   └── E07 (log1p_target)
│       └── E30 (Chain D1)
│           └── E31 (Chain D2)
│               ├── E32 (Chain D3)
│               └── E33 (Chain D4)
```

---

**Generated files**: 123 total
- 36 model templates
- 36 chain templates
- 51 module templates

**Total runtime estimate**: ~36 hours (sequential) or ~9-12 hours (4-way parallel)
