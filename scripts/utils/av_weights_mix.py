#!/usr/bin/env python
"""
Adversarial weights with optional merge of original dataset.

Two modes:
- align (default): keep tylko kolumny konkursowe; dodatkowe z oryginału są odrzucane,
  brakujące uzupełniane NA. Wagi zwracane tylko dla oryginalnego train (clip).
- union: bierzemy sumę zbioru kolumn; brakujące uzupełniane NA; opcjonalnie dodajemy
  kolumnę źródła z flagą (syntetyk=0, oryginał=1), by zaznaczyć brakujące pola.

Nie modyfikuje istniejącego pipeline; to standalone helper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="playground-series-s5e12", help="Slug projektu w projects/kaggle/")
    ap.add_argument("--orig-path", default="data/diabetes_dataset.csv", help="Ścieżka do oryginalnego zbioru (relatywna do project root lub absolutna)")
    ap.add_argument("--mode", choices=["align", "union"], default="align", help="align=kolumny konkursu; union=pełna suma kolumn")
    ap.add_argument("--source-flag", default=None, help="Opcjonalna kolumna z flagą źródła (0=synth train/test, 1=orig)")
    ap.add_argument("--output", default="data/train_av_weights_mix.csv", help="Dokąd zapisać wagi (rel. do project root lub ścieżka abs.)")
    ap.add_argument("--model-dir", default="preprocess_cache/av_model_mix", help="Katalog na model AV (rel. lub abs.)")
    ap.add_argument("--time-limit", type=int, default=300, help="Limit czasu dla AV (s)")
    ap.add_argument("--presets", default="medium_quality_faster_train", help="Preset AutoGluon")
    ap.add_argument("--included-model-types", nargs="*", default=None, help="Opcjonalne typy modeli (GBM/CAT/XGB/...)")
    ap.add_argument("--id-column", default="id")
    ap.add_argument("--target-column", default="diagnosed_diabetes")
    ap.add_argument("--drop-columns", nargs="*", default=None, help="Dodatkowe kolumny do drop przed AV")
    ap.add_argument("--keep-extra-weights", action="store_true", help="Domyślnie wagi przycinane do oryginalnego train; ta flaga zachowuje wagi także dla rekordów z oryginału.")
    return ap.parse_args()


def resolve_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p)


def resolve_data_file(data_dir: Path, filename: str) -> Path:
    """Resolve data file with .csv.gz fallback for backward compatibility."""
    gz_path = data_dir / f"{filename}.gz"
    if gz_path.exists():
        return gz_path
    csv_path = data_dir / filename
    if csv_path.exists():
        return csv_path
    # Return .csv.gz as default (will be created if needed)
    return gz_path


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    project_root = repo_root / "projects" / "kaggle" / args.project
    data_dir = project_root / "data"

    # Paths with .csv.gz fallback support
    train_path = resolve_data_file(data_dir, "train.csv")
    test_path = resolve_data_file(data_dir, "test.csv")
    orig_path = resolve_path(args.orig_path, project_root)
    output_path = resolve_path(args.output, project_root)
    model_dir = resolve_path(args.model_dir, project_root)

    # Imports
    sys.path.insert(0, str(project_root / "code"))
    from adversarial_validation import compute_adversarial_weights  # type: ignore

    print(f"[info] Project: {project_root}")
    print(f"[info] Mode: {args.mode}")
    print(f"[info] Train: {train_path}")
    print(f"[info] Test:  {test_path}")
    print(f"[info] Orig:  {orig_path}")

    train_df = pd.read_csv(train_path, compression='infer')
    test_df = pd.read_csv(test_path, compression='infer')
    orig_df = pd.read_csv(orig_path, compression='infer')

    id_col = args.id_column
    target_col = args.target_column

    # Build extra_df with IDs
    if id_col not in orig_df.columns:
        start_id = train_df[id_col].max() + 1 if id_col in train_df.columns else 0
        orig_df[id_col] = range(start_id, start_id + len(orig_df))

    if args.mode == "align":
        keep_cols: List[str] = [c for c in train_df.columns if c != target_col or target_col in orig_df.columns]
        extra_df = orig_df.reindex(columns=keep_cols, fill_value=pd.NA)
    else:  # union
        union_cols = sorted(set(train_df.columns) | set(orig_df.columns) | set(test_df.columns))
        train_df = train_df.reindex(columns=union_cols, fill_value=pd.NA)
        test_df = test_df.reindex(columns=union_cols, fill_value=pd.NA)
        extra_df = orig_df.reindex(columns=union_cols, fill_value=pd.NA)

    def add_source(df: pd.DataFrame, flag: int) -> pd.DataFrame:
        if args.source_flag:
            df = df.copy()
            df[args.source_flag] = flag
        return df

    train_len = len(train_df)
    train_concat = pd.concat([add_source(train_df, 0), add_source(extra_df, 1)], ignore_index=True)
    test_df = add_source(test_df, 0)

    drop_cols = args.drop_columns or []
    if id_col not in drop_cols:
        drop_cols = drop_cols + [id_col]

    model_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    av_df = compute_adversarial_weights(
        train_df=train_concat,
        test_df=test_df,
        id_column=id_col,
        target_column=target_col,
        drop_columns=drop_cols,
        presets=args.presets,
        time_limit=args.time_limit,
        included_model_types=args.included_model_types,
        output_dir=model_dir,
    )

    if not args.keep_extra_weights:
        av_df = av_df.iloc[:train_len].reset_index(drop=True)

    av_df.to_csv(output_path, index=False, compression='infer')
    print(f"[done] Saved AV weights to {output_path}")
    print(f"[info] Rows: {len(av_df)}, Columns: {list(av_df.columns)}")


if __name__ == "__main__":
    main()
