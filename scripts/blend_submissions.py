#!/usr/bin/env python3
"""
Blend top-N Kaggle submissions using public scores as weights.

Default flow:
1) Parse Kaggle CLI output from /tmp/sub.txt
2) Select top-N by public score (tie-breaker: newer date)
3) Map Kaggle filenames to local submissions in /mnt/mlarena
4) Blend predictions with weighted average
5) Save blended submission + manifest

Example:
python scripts/blend_submissions.py \
  --project playground-series-s5e12 \
  --kaggle-output /tmp/sub.txt \
  --top-n 5 \
  --weighting public \
  --output-name submission-blend-top5-public.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


DATE_FMT = "%Y-%m-%d %H:%M:%S.%f"


def load_weighted_blender(repo_root: Path):
    blender_path = repo_root / "src" / "kaggle_tools" / "stacking" / "blender.py"
    if not blender_path.exists():
        raise FileNotFoundError(f"Missing blender module at {blender_path}")
    spec = importlib.util.spec_from_file_location("kaggle_tools.stacking.blender", blender_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load blender module from {blender_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WeightedBlender


def parse_kaggle_output(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Kaggle output file: {path}")

    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        lower = line.lower()
        if lower.startswith("filename") or line.startswith("---"):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 5:
            continue
        filename, date_str, _desc, _status, public_score = parts[:5]
        try:
            public_score_val = float(public_score)
        except ValueError:
            continue
        try:
            date = datetime.strptime(date_str, DATE_FMT)
        except ValueError:
            date = None
        rows.append(
            {
                "filename": filename,
                "public_score": public_score_val,
                "date": date,
            }
        )

    if not rows:
        raise RuntimeError(f"No submission rows with public scores found in {path}")
    return rows


def select_top_n(rows: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["public_score"], r["date"] or datetime.min),
        reverse=True,
    )
    return rows_sorted[:top_n]


def resolve_local_file(submissions_dir: Path, kaggle_filename: str) -> Path:
    direct = submissions_dir / kaggle_filename
    if direct.exists():
        return direct
    stem = Path(kaggle_filename).stem
    matches = sorted(submissions_dir.glob(f"{stem}-*.csv"))
    if not matches:
        raise FileNotFoundError(f"No local submission found for {kaggle_filename}")
    if len(matches) > 1:
        print(f"Warning: multiple matches for {kaggle_filename}, using {matches[0].name}")
    return matches[0]


def compute_weights(
    scores: List[float],
    mode: str,
    power: float,
    temp: float,
    eps: float,
    manual: List[float] | None,
) -> List[float]:
    n = len(scores)
    if n == 0:
        raise ValueError("No scores provided for weight computation")

    if mode == "public":
        raw = scores[:]
    elif mode == "rank":
        raw = [float(n - i) for i in range(n)]
    elif mode == "power":
        raw = [float(s) ** power for s in scores]
    elif mode == "softmax":
        if temp <= 0:
            raise ValueError("temp must be > 0 for softmax weighting")
        raw = [math.exp(float(s) / temp) for s in scores]
    elif mode == "offset":
        min_score = min(scores)
        raw = [max(float(s) - min_score + eps, 0.0) for s in scores]
    elif mode == "uniform":
        raw = [1.0] * n
    elif mode == "manual":
        if not manual or len(manual) != n:
            raise ValueError("manual weights must be provided and match top-n")
        raw = manual[:]
    else:
        raise ValueError(f"Unknown weighting mode: {mode}")

    total = sum(raw)
    if total <= 0:
        raise ValueError("Weights sum to zero; adjust weighting settings")
    return [float(w) / total for w in raw]


def blend_predictions(
    model_files: List[Path],
    weights: List[float],
    id_column: str | None,
    target_column: str | None,
) -> pd.DataFrame:
    first_df = pd.read_csv(model_files[0])
    id_col = id_column or first_df.columns[0]
    target_col = target_column or first_df.columns[-1]
    base_ids = first_df[id_col]

    preds = []
    for path in model_files:
        df = pd.read_csv(path)
        if id_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Missing columns in {path.name}: expected {id_col}, {target_col}")
        if df[id_col].equals(base_ids):
            series = df[target_col]
        else:
            aligned = df.set_index(id_col).reindex(base_ids)
            if aligned[target_col].isna().any():
                raise ValueError(f"ID mismatch when aligning {path.name}")
            series = aligned[target_col].reset_index(drop=True)
        preds.append(series)

    WeightedBlender = load_weighted_blender(Path(__file__).resolve().parents[1])
    blender = WeightedBlender()
    ensemble = blender.blend(preds, weights)
    return pd.DataFrame({id_col: base_ids, target_col: ensemble})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend top-N Kaggle submissions.")
    parser.add_argument("--project", required=True, help="Kaggle project slug (e.g., playground-series-s5e12)")
    parser.add_argument("--kaggle-output", default="/tmp/sub.txt", help="Path to Kaggle CLI output file")
    parser.add_argument(
        "--submissions-dir",
        default=None,
        help="Override submissions directory (defaults to /mnt/mlarena/projects/kaggle/<project>/submissions)",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of top submissions to blend")
    parser.add_argument(
        "--weighting",
        choices=["public", "rank", "power", "softmax", "offset", "uniform", "manual"],
        default="public",
        help="Weighting strategy for blending",
    )
    parser.add_argument("--power", type=float, default=2.0, help="Power for power weighting")
    parser.add_argument("--temp", type=float, default=0.001, help="Temperature for softmax weighting")
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon for offset weighting")
    parser.add_argument("--weights", nargs="+", type=float, help="Manual weights (only for --weighting manual)")
    parser.add_argument("--output-name", default=None, help="Output CSV filename")
    parser.add_argument("--id-column", default=None, help="ID column name override")
    parser.add_argument("--target-column", default=None, help="Target column name override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    submissions_dir = (
        Path(args.submissions_dir)
        if args.submissions_dir
        else Path("/mnt/mlarena/projects/kaggle") / args.project / "submissions"
    )
    if not submissions_dir.exists():
        raise FileNotFoundError(f"Submissions dir not found: {submissions_dir}")

    rows = parse_kaggle_output(Path(args.kaggle_output))
    selected = select_top_n(rows, args.top_n)
    if not selected:
        raise RuntimeError("No submissions selected for blending")

    model_files = [resolve_local_file(submissions_dir, r["filename"]) for r in selected]
    scores = [r["public_score"] for r in selected]
    weights = compute_weights(scores, args.weighting, args.power, args.temp, args.eps, args.weights)

    output_name = args.output_name or f"submission-blend-top{len(model_files)}-{args.weighting}.csv"
    output_path = submissions_dir / output_name

    submission = blend_predictions(model_files, weights, args.id_column, args.target_column)
    submission.to_csv(output_path, index=False)

    manifest = {
        "project": args.project,
        "kaggle_output": str(Path(args.kaggle_output)),
        "submissions_dir": str(submissions_dir),
        "top_n": len(model_files),
        "weighting": args.weighting,
        "power": args.power,
        "temp": args.temp,
        "eps": args.eps,
        "weights": weights,
        "models": [
            {
                "kaggle_filename": r["filename"],
                "local_filename": f.name,
                "public_score": r["public_score"],
                "date": r["date"].strftime("%Y-%m-%d %H:%M:%S") if r["date"] else None,
            }
            for r, f in zip(selected, model_files)
        ],
        "output": output_path.name,
    }
    manifest_path = Path(str(output_path) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("Top submissions:")
    for r, f, w in zip(selected, model_files, weights):
        date = r["date"].strftime("%Y-%m-%d %H:%M:%S") if r["date"] else "-"
        print(f"- {r['filename']} | public={r['public_score']:.5f} | date={date} | local={f.name} | weight={w:.6f}")
    print(f"\nSaved blended submission: {output_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
