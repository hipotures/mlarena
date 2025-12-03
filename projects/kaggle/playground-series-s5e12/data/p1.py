"""
Adversarial validation between train and test for playground-series-s5e12.

Steps:
- Load train/test from this directory.
- Drop target/id from features.
- Build combined dataset with `is_test` flag.
- 5-fold CV AUC to measure shift; report top feature importances.
- Adversarial filtering: drop lowest-quantile train rows (most unlike test) and
  show new AUC after filtering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score

# LightGBM optional import guard
try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "LightGBM is required for adversarial validation. "
        "Install with: uv run pip install lightgbm"
    ) from exc


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(__file__).resolve().parent
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_adv_dataset(train: pd.DataFrame, test: pd.DataFrame):
    # Drop target/id appropriately
    train_adv = train.drop(columns=["diagnosed_diabetes", "id"], errors="ignore").copy()
    test_adv = test.drop(columns=["id"], errors="ignore").copy()

    train_adv["is_test"] = 0
    test_adv["is_test"] = 1

    adv_data = pd.concat([train_adv, test_adv], axis=0).reset_index(drop=True)
    X_adv = adv_data.drop(columns=["is_test"])
    y_adv = adv_data["is_test"]

    # Cast object columns to categorical for LightGBM
    for col in X_adv.select_dtypes(include="object").columns:
        X_adv[col] = X_adv[col].astype("category")

    return X_adv, y_adv, len(train_adv)


def run_adversarial_validation(X: pd.DataFrame, y: pd.Series):
    model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return scores, model


def main():
    train_df, test_df = load_data()
    X_adv, y_adv, train_len = build_adv_dataset(train_df, test_df)

    scores, model = run_adversarial_validation(X_adv, y_adv)
    print(f"Adversarial Validation AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    model.fit(X_adv, y_adv)
    feat_imp = (
        pd.DataFrame({"feature": X_adv.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 15 features distinguishing Train vs Test:")
    print(feat_imp.head(15).to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feat_imp.head(15))
    plt.title("Top features distinguishing Train from Test")
    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "adv_feature_importance.png"
    plt.savefig(out_path, dpi=200)
    print(f"\nFeature importance plot saved to: {out_path}")

    # --- Adversarial filtering: drop lowest-quantile train rows (most train-like) ---
    train_probs = model.predict_proba(X_adv.iloc[:train_len])[:, 1]
    train_with_probs = train_df.copy()
    train_with_probs["adv_val_prob"] = train_probs
    cutoff = 0.05  # remove bottom 5% (most unlike test)
    threshold = train_with_probs["adv_val_prob"].quantile(cutoff)

    keep_mask = train_with_probs["adv_val_prob"] > threshold
    train_aligned = train_with_probs[keep_mask].drop(columns=["adv_val_prob"])
    removed = len(train_with_probs) - len(train_aligned)
    print(f"\nAdversarial filtering: removed {removed} rows ({cutoff*100:.1f}% cutoff).")
    print(f"Train size: {len(train_df)} -> {len(train_aligned)}")

    # Optional: re-run adversarial validation on filtered train to see drift drop
    X_adv_f, y_adv_f, _ = build_adv_dataset(train_aligned, test_df)
    scores_f, _ = run_adversarial_validation(X_adv_f, y_adv_f)
    print(f"Post-filter Adversarial AUC: {scores_f.mean():.4f} (+/- {scores_f.std():.4f})")


if __name__ == "__main__":
    main()
