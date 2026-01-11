#!/usr/bin/env python3
import time
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from pathlib import Path
import warnings
from functools import partial

# UI Imports
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.tree import Tree
from rich import box

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "projects/kaggle/playground-series-s6e1/data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TARGET_COL = "exam_score"
ID_COL = "id"
OUTPUT_DIR = Path("standalone_output")
OUTPUT_DIR.mkdir(exist_ok=True)

console = Console()

# --- HELPER CLASSES ---

class SanityCheck(BaseEstimator, TransformerMixin):
    def __init__(self, drop_duplicates=False, max_missing_fraction=0.97):
        self.drop_duplicates = drop_duplicates
        self.max_missing_fraction = max_missing_fraction
        self.cols_to_drop = []
    def fit(self, X, y=None):
        missing_frac = X.isnull().mean()
        self.cols_to_drop = missing_frac[missing_frac > self.max_missing_fraction].index.tolist()
        return self
    def transform(self, X):
        X = X.copy()
        if self.drop_duplicates: X = X.drop_duplicates()
        if self.cols_to_drop: X = X.drop(columns=self.cols_to_drop, errors='ignore')
        return X

class MissingnessFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, add_row_stats=True, cap_row_missing_count=25):
        self.add_row_stats = add_row_stats
        self.cap_row_missing_count = cap_row_missing_count
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if self.add_row_stats:
            missing_count = X.isnull().sum(axis=1)
            if self.cap_row_missing_count is not None:
                missing_count = missing_count.clip(upper=self.cap_row_missing_count)
            X['row_missing_count'] = missing_count.astype(float)
            X['row_missing_ratio'] = X.isnull().mean(axis=1).astype(float)
        return X

class RareCategoryHandler(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=19, min_freq_ratio=0.015, top_k=10, rare_label="__RARE__"):
        self.min_freq = min_freq
        self.min_freq_ratio = min_freq_ratio
        self.top_k = top_k
        self.rare_label = rare_label
        self.top_categories = {}
    def fit(self, X, y=None):
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        n_rows = len(X)
        for col in cat_cols:
            counts = X[col].value_counts()
            if self.top_k is not None:
                keep = counts.index[:int(self.top_k)].tolist()
            else:
                valid_mask = (counts >= self.min_freq) & ((counts / n_rows) >= self.min_freq_ratio)
                keep = counts[valid_mask].index.tolist()
            self.top_categories[col] = set(keep)
        return self
    def transform(self, X):
        X = X.copy()
        for col, cats in self.top_categories.items():
            if col in X.columns:
                X[col] = X[col].apply(lambda x: x if x in cats or pd.isna(x) else self.rare_label)
        return X

class NumericBinner(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=25, strategy='kmeans'):
        self.n_bins = int(n_bins)
        self.strategy = strategy
        self.binners = {}
    def fit(self, X, y=None):
        from sklearn.preprocessing import KBinsDiscretizer
        for col in X.columns:
            est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
            try:
                est.fit(X[[col]].dropna())
                self.binners[col] = est
            except: pass 
        return self
    def transform(self, X):
        X = X.copy()
        for col, binner in self.binners.items():
            if col in X.columns:
                mask = ~X[col].isna()
                if mask.any():
                    X.loc[mask, f"{col}_bin"] = binner.transform(X.loc[mask, [col]]).flatten()
        return X

class FeaturePolynomial(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, max_features=25):
        self.degree = degree
        self.max_features = int(max_features)
        self.poly = None
        self.new_feature_names = []
    def fit(self, X, y=None):
        X_num = X.fillna(0)
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.poly.fit(X_num)
        all_names = self.poly.get_feature_names_out(X.columns)
        new_names_pool = all_names[len(X.columns):]
        if len(new_names_pool) == 0: return self
        self.new_feature_names = list(new_names_pool)[:self.max_features]
        return self
    def transform(self, X):
        if self.poly and self.new_feature_names and not X.empty and X.shape[1] > 0:
            X_num = X.fillna(0)
            X_poly = self.poly.transform(X_num)
            X_new = X_poly[:, len(X.columns):]
            X_selected = X_new[:, :len(self.new_feature_names)]
            return pd.DataFrame(X_selected, columns=self.new_feature_names, index=X.index)
        return pd.DataFrame(index=X.index)

class ClusteringFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=45):
        self.n_clusters = int(n_clusters)
        self.kmeans = None
    def fit(self, X, y=None):
        if X.empty or X.shape[1] == 0: return self
        X_fill = X.fillna(0)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        self.kmeans.fit(X_fill)
        return self
    def transform(self, X):
        if self.kmeans is not None and not X.empty and X.shape[1] > 0:
            X_fill = X.fillna(0)
            res = pd.DataFrame(index=X.index)
            res['cluster_id'] = self.kmeans.predict(X_fill).astype(float)
            return res
        return pd.DataFrame(index=X.index)

class DriftDetector(BaseEstimator, TransformerMixin):
    def __init__(self, max_psi=0.2, max_drop_fraction=0.1):
        self.max_psi = max_psi
        self.max_drop_fraction = max_drop_fraction
        self.dropped_features = []
    def fit_with_test(self, X_train, X_test):
        n_features = X_train.shape[1]
        max_drop = int(n_features * self.max_drop_fraction)
        psi_scores = []
        for col in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                train_vals, test_vals = X_train[col].dropna(), X_test[col].dropna()
                try:
                    bins = np.histogram_bin_edges(train_vals, bins=10)
                    t_h, _ = np.histogram(train_vals, bins=bins)
                    s_h, _ = np.histogram(test_vals, bins=bins)
                    t_p, s_p = t_h/len(train_vals), s_h/len(test_vals)
                    eps = 1e-6
                    t_p, s_p = np.where(t_p==0, eps, t_p), np.where(s_p==0, eps, s_p)
                    psi = np.sum((t_p - s_p) * np.log(t_p / s_p))
                    psi_scores.append((col, psi))
                except: psi_scores.append((col, 0.0))
            else: psi_scores.append((col, 0.0))
        high_drift = sorted([x for x in psi_scores if x[1] > self.max_psi], key=lambda x: x[1], reverse=True)
        self.dropped_features = [x[0] for x in high_drift[:max_drop]]
        return self
    def transform(self, X):
        return X.drop(columns=self.dropped_features, errors='ignore')

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=0.5):
        self.n_features = n_features
        self.selected_cols = []
    def fit(self, X, y=None):
        if y is None: return self
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_num = X[num_cols].fillna(0)
        k = int(len(num_cols) * self.n_features) if isinstance(self.n_features, float) else int(self.n_features)
        mi_func = partial(mutual_info_regression, random_state=42, n_jobs=-1)
        selector = SelectKBest(mi_func, k=k)
        selector.fit(X_num, y)
        self.selected_cols = [num_cols[i] for i in selector.get_support(indices=True)]
        self.selected_cols += X.select_dtypes(exclude=['number']).columns.tolist()
        return self
    def transform(self, X): return X[self.selected_cols]

# --- UI HELPER ---

def print_step_flow(name, duration, datasets):
    """Renders a panel similar to trace_trial_flow.py with timing and shapes."""
    tree = Tree(f"[bold cyan]{name} ({duration:.2f}s)[/]", guide_style="dim green")
    for d_name, df in datasets.items():
        if df is not None:
            rows, cols = df.shape
            tree.add(f"📄 [white]{d_name:6}[/] [dim]([yellow]{rows:,}[/] × [bold green]{cols}[/])")
    console.print(Align.center(Panel(tree, box=box.ROUNDED, expand=False, border_style="blue")))
    console.print(Align.center(Text("↓", style="bold yellow")))

# --- MAIN PIPELINE ---

def run_pipeline():
    console.print(Panel("[bold magenta]STANDALONE STANDALONE PIPELINE SIMULATOR[/]", box=box.DOUBLE, expand=False, border_style="magenta"))
    console.print(Align.center(Text("↓", style="bold yellow")))

    # 1. Loading
    t_load = time.time()
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    d_load = time.time() - t_load
    print_step_flow("Data Loading", d_load, {"train": train_raw, "test": test_raw})

    original_features = ["age", "gender", "course", "study_hours", "class_attendance", "internet_access", "sleep_hours", "sleep_quality", "study_method", "facility_rating", "exam_difficulty"]

    # Step 0
    t = time.time()
    sc = SanityCheck(max_missing_fraction=0.97)
    train_raw = sc.fit_transform(train_raw)
    test_raw = sc.transform(test_raw)
    print_step_flow("0-sanity_check", time.time()-t, {"train": train_raw, "test": test_raw})

    # Step 1
    t = time.time()
    shuffled = train_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    n_tr, n_tu, n_ev = int(n*0.6), int(n*0.1), int(n*0.1)
    train_df = shuffled.iloc[:n_tr].reset_index(drop=True)
    tuning_df = shuffled.iloc[n_tr : n_tr+n_tu].reset_index(drop=True)
    eval_df = shuffled.iloc[n_tr+n_tu : n_tr+n_tu+n_ev].reset_index(drop=True)
    for df in [train_df, tuning_df, eval_df, test_raw]:
        if ID_COL in df.columns: df.drop(columns=[ID_COL], inplace=True)
    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_tuning, X_eval, X_test = tuning_df.drop(columns=[TARGET_COL]), eval_df.drop(columns=[TARGET_COL]), test_raw
    print_step_flow("1-train_fraction", time.time()-t, {"train": X_train, "tuning": X_tuning, "eval": X_eval, "test": X_test})

    def apply_step(name, transformer, fit_y=False, use_orig_only=False):
        nonlocal X_train, X_tuning, X_eval, X_test
        t_s = time.time()
        current_cols = X_train.columns.tolist()
        input_cols = [c for c in original_features if c in current_cols] if use_orig_only else current_cols
        
        if fit_y: transformer.fit(X_train[input_cols], y_train)
        else: transformer.fit(X_train[input_cols])
        
        def _do(df):
            transformed = transformer.transform(df[input_cols])
            if name.split("-")[-1] == "feature_selector": return transformed
            new_cols = [c for c in transformed.columns if c not in input_cols]
            if new_cols: return pd.concat([df, transformed[new_cols]], axis=1)
            res = df.copy()
            res[input_cols] = transformed.values
            return res

        X_train, X_tuning, X_eval, X_test = _do(X_train), _do(X_tuning), _do(X_eval), _do(X_test)
        print_step_flow(name, time.time()-t_s, {"train": X_train, "tuning": X_tuning, "eval": X_eval, "test": X_test})

    # Step 2
    print_step_flow("2-target_transformer", 0.0, {"train": X_train})

    # Step 3
    apply_step("3-missingness_features", MissingnessFeatures(), use_orig_only=True)

    # Step 4
    t = time.time()
    num_cols, cat_cols = X_train.select_dtypes(include=['number']).columns, X_train.select_dtypes(include=['object']).columns
    imp_num = SimpleImputer(strategy='most_frequent').fit(X_train[num_cols])
    imp_cat = SimpleImputer(strategy='most_frequent').fit(X_train[cat_cols])
    for df in [X_train, X_tuning, X_eval, X_test]:
        df[num_cols], df[cat_cols] = imp_num.transform(df[num_cols]), imp_cat.transform(df[cat_cols])
    print_step_flow("4-imputer", time.time()-t, {"train": X_train, "test": X_test})

    # Step 5-6
    apply_step("5-rare_category_handler", RareCategoryHandler(top_k=10), use_orig_only=True)
    apply_step("6-numeric_binner", NumericBinner(), use_orig_only=True)

    # Step 7
    t = time.time()
    cat_cols_enc = [c for c in X_train.select_dtypes(include=['object']).columns if c in original_features]
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int8).fit(X_train[cat_cols_enc])
    f_names = ohe.get_feature_names_out(cat_cols_enc)
    def _do_ohe(df):
        enc = pd.DataFrame(ohe.transform(df[cat_cols_enc]), columns=f_names, index=df.index)
        return pd.concat([df.drop(columns=cat_cols_enc), enc], axis=1)
    X_train, X_tuning, X_eval, X_test = _do_ohe(X_train), _do_ohe(X_tuning), _do_ohe(X_eval), _do_ohe(X_test)
    print_step_flow("7-encoder", time.time()-t, {"train": X_train, "test": X_test})

    # Step 8
    t = time.time()
    scaler_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=800, random_state=42).fit(X_train[scaler_cols])
    def _do_scale(df):
        res = df.copy()
        res[scaler_cols] = scaler.transform(df[scaler_cols])
        return res
    X_train, X_tuning, X_eval, X_test = _do_scale(X_train), _do_scale(X_tuning), _do_scale(X_eval), _do_scale(X_test)
    print_step_flow("8-scaler", time.time()-t, {"train": X_train, "test": X_test})

    # Step 9-10
    apply_step("9-feature_polynomial", FeaturePolynomial(), use_orig_only=True)
    apply_step("10-clustering_features", ClusteringFeatures(), use_orig_only=True)

    # Step 11
    t = time.time()
    dd = DriftDetector().fit_with_test(X_train, X_test)
    X_train, X_tuning, X_eval, X_test = dd.transform(X_train), dd.transform(X_tuning), dd.transform(X_eval), dd.transform(X_test)
    print_step_flow("11-drift_detector", time.time()-t, {"train": X_train, "test": X_test})

    # Step 12
    apply_step("12-feature_selector", CustomFeatureSelector(n_features=0.5), fit_y=True)

    console.print(Panel(f"[bold green]PIPELINE COMPLETE[/]\nFinal Shape: {X_train.shape}", box=box.DOUBLE, expand=False, border_style="green"))
    X_train.to_parquet(OUTPUT_DIR / "train_processed.parquet")
    X_test.to_parquet(OUTPUT_DIR / "test_processed.parquet")

if __name__ == "__main__":
    run_pipeline()