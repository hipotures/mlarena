# dae_embeddings

## Overview
Trains a small denoising autoencoder (MLPRegressor) on numeric features and appends
hidden-layer embeddings to the dataset. Noise can be swap or gaussian. The model is
fit on train or train+test depending on `fit_on`.

## Parameters
- `include_cols` (list[str] | null): Explicit numeric columns to use. Defaults to all numeric.
- `exclude_cols` (list[str]): Columns to exclude.
- `use_original_features_only` (bool): Restrict to original features.
- `embedding_dim` (int): Hidden size when `hidden_layers` is not provided.
- `hidden_layers` (list[int] | null): Custom hidden layer sizes.
- `activation` (str): `relu`, `tanh`, `logistic`, `identity`.
- `max_iter` (int): MLP training iterations.
- `batch_size` (int): MLP batch size.
- `learning_rate_init` (float): MLP learning rate.
- `alpha` (float): L2 regularization.
- `early_stopping` (bool): Enable early stopping.
- `validation_fraction` (float): Fraction for early stopping.
- `random_state` (int): RNG seed.
- `noise_type` (str): `swap` or `gaussian`.
- `swap_prob` (float): Swap noise probability per cell.
- `gaussian_sigma` (float): Gaussian noise sigma.
- `gaussian_scale_by_std` (bool): Scale noise by column std.
- `fit_on` (str): `train`, `train_val`, `train_test`, `train_val_test`, `all`.
- `max_rows` (int | null): Optional subsample limit for fitting.
- `scale` (bool): Standardize numeric columns before training.
- `missing_strategy` (str): `mean`, `median`, `zero`.
- `add_original` (bool): Keep original numeric columns.
- `drop_original` (bool): Drop original numeric columns.
- `prefix` (str): Prefix for embedding columns.

## Example
```yaml
module: dae_embeddings
config:
  embedding_dim: 16
  noise_type: swap
  swap_prob: 0.15
  fit_on: train_test
  scale: true
```

## Artifacts
- `dae_model.pkl`: Trained MLPRegressor.
- `scaler.pkl`: Optional StandardScaler.
- `summary.json`: Transformation summary.

## Notes
- Requires numeric input; imputed values are used for missing entries.
- Use after encoding/scaling in the pipeline for best stability.
- This step can be heavy on large feature sets.
