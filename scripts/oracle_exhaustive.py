#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import warnings
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor
from rich.console import Console
from rich.table import Table

from mlarena.modules.mcts.node import PipelineState
from mlarena.modules.mcts.space import SuperChainActionSpace

console = Console()
warnings.filterwarnings("ignore")


def info(message: str) -> None:
    console.print(message)


def warn(message: str) -> None:
    console.print(f"[yellow]Warning:[/yellow] {message}")


def err(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")


def flatten_config(action_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    group = action_dict.get("group_name", "unknown")
    variant = action_dict.get("variant", "unknown")
    flat[f"{prefix}action_group"] = group
    flat[f"{prefix}action_variant"] = variant
    config = action_dict.get("config", {})
    for k, v in config.items():
        key = f"{prefix}{group}_{k}"
        if isinstance(v, (list, dict)):
            flat[key] = json.dumps(v, sort_keys=True)
        else:
            flat[key] = v
    return flat


def parse_action_full(action_json_str: Optional[str], prefix: str = "") -> Dict[str, Any]:
    if not action_json_str or pd.isna(action_json_str):
        return {}
    try:
        data = json.loads(action_json_str)
        return flatten_config(data, prefix=prefix)
    except Exception:
        return {}


def _load_best_parent(conn: sqlite3.Connection, study_name: str) -> Optional[Tuple[int, float, int]]:
    row = conn.execute(
        "SELECT study_id FROM studies WHERE study_name = ?",
        (study_name,),
    ).fetchone()
    if not row:
        return None
    study_id = row[0]

    query = """
    SELECT e.trial_id, e.value, n.depth
    FROM mcts_evaluations e
    JOIN mcts_nodes n ON e.trial_id = n.trial_id
    WHERE n.study_id = ? AND e.status = 'COMPLETE' AND e.value IS NOT NULL
    ORDER BY e.value DESC
    LIMIT 1
    """
    return conn.execute(query, (study_id,)).fetchone()


def _load_best_parent_at_depth(
    conn: sqlite3.Connection,
    study_name: str,
    depth: int,
) -> Optional[Tuple[int, float, int]]:
    row = conn.execute(
        "SELECT study_id FROM studies WHERE study_name = ?",
        (study_name,),
    ).fetchone()
    if not row:
        return None
    study_id = row[0]

    query = """
    SELECT e.trial_id, e.value, n.depth
    FROM mcts_evaluations e
    JOIN mcts_nodes n ON e.trial_id = n.trial_id
    WHERE n.study_id = ? AND n.depth = ? AND e.status = 'COMPLETE' AND e.value IS NOT NULL
    ORDER BY e.value DESC
    LIMIT 1
    """
    return conn.execute(query, (study_id, depth)).fetchone()


def _load_parent_by_id(conn: sqlite3.Connection, trial_id: int) -> Optional[Tuple[int, float, int]]:
    query = """
    SELECT e.trial_id, e.value, n.depth
    FROM mcts_evaluations e
    JOIN mcts_nodes n ON e.trial_id = n.trial_id
    WHERE e.trial_id = ? AND e.status = 'COMPLETE' AND e.value IS NOT NULL
    LIMIT 1
    """
    return conn.execute(query, (trial_id,)).fetchone()


def _reconstruct_state(conn: sqlite3.Connection, trial_id: int) -> Tuple[Dict[str, str], int, Optional[str]]:
    used_groups: Dict[str, str] = {}
    last_step_index = -1
    prev_action_json = None
    curr_id = trial_id

    while True:
        row = conn.execute(
            "SELECT parent_trial_id, action_json FROM mcts_edges WHERE child_trial_id = ?",
            (curr_id,),
        ).fetchone()
        if not row:
            break
        parent_id, action_json = row
        if curr_id == trial_id:
            prev_action_json = action_json
        try:
            act = json.loads(action_json)
            group = act.get("group_name") or act.get("group")
            step = act.get("step_name") or act.get("step")
            if group:
                used_groups[group] = step
            s_idx = int(act.get("searched_index", -1))
            if s_idx > last_step_index:
                last_step_index = s_idx
        except Exception:
            pass
        curr_id = parent_id

    return used_groups, last_step_index, prev_action_json


def _coerce_float(value: float, round_to: Optional[int]) -> float:
    if round_to is not None:
        try:
            return round(value, int(round_to))
        except Exception:
            return float(value)
    return float(value)


def _float_grid(
    spec: Dict[str, Any],
    *,
    samples: int,
    strategy: str,
    rng: random.Random,
) -> List[float]:
    min_val = float(spec.get("min", 0.0))
    max_val = float(spec.get("max", 0.0))
    if max_val < min_val:
        return []

    round_to = spec.get("round")
    log = bool(spec.get("log", False))
    if samples <= 0:
        return []

    if log:
        if min_val <= 0:
            min_val = 1e-10
        if strategy == "random":
            values = [math.exp(rng.uniform(math.log(min_val), math.log(max_val))) for _ in range(samples)]
        else:
            if samples == 1:
                values = [min_val]
            else:
                step = (math.log(max_val) - math.log(min_val)) / (samples - 1)
                values = [math.exp(math.log(min_val) + i * step) for i in range(samples)]
    else:
        if strategy == "random":
            values = [rng.uniform(min_val, max_val) for _ in range(samples)]
        else:
            if samples == 1:
                values = [min_val]
            else:
                step = (max_val - min_val) / (samples - 1)
                values = [min_val + i * step for i in range(samples)]

    return [_coerce_float(v, round_to) for v in values]


def _param_values(
    spec: Dict[str, Any],
    *,
    float_samples: int,
    float_strategy: str,
    rng: random.Random,
) -> List[Any]:
    ptype = spec.get("type")
    if ptype == "choice":
        return list(spec.get("values", []) or [])
    if ptype == "int_range":
        mn = int(spec.get("min", 0))
        mx = int(spec.get("max", 0))
        step = int(spec.get("step", 1))
        if step <= 0 or mx < mn:
            return []
        return list(range(mn, mx + 1, step))
    if ptype == "float_range":
        step = spec.get("step")
        if step is not None:
            mn = float(spec.get("min", 0.0))
            mx = float(spec.get("max", 0.0))
            step = float(step)
            if step <= 0 or mx < mn:
                return []
            count = int(math.floor((mx - mn) / step))
            values = [mn + i * step for i in range(count + 1)]
            return [_coerce_float(v, spec.get("round")) for v in values]
        return _float_grid(spec, samples=float_samples, strategy=float_strategy, rng=rng)
    if ptype == "bool":
        return [True, False]
    if ptype == "fixed":
        return [spec.get("value")]
    if ptype == "subset":
        values = spec.get("values", []) or []
        n = len(values)
        if n == 0:
            return [[]]
        min_items = int(spec.get("min_items", 1))
        max_items = int(spec.get("max_items", n))
        max_items = min(max_items, n)
        min_items = max(min_items, 0)
        if min_items > max_items:
            return []
        combos: List[List[Any]] = []
        for k in range(min_items, max_items + 1):
            for combo in combinations(values, k):
                combo_list = list(combo)
                if spec.get("sort", False):
                    try:
                        combo_list.sort()
                    except Exception:
                        pass
                combos.append(combo_list)
        return combos
    return [None]


def _iter_param_grid(
    params: Dict[str, Any],
    *,
    float_samples: int,
    float_strategy: str,
    rng: random.Random,
) -> Iterable[Dict[str, Any]]:
    if not params:
        yield {}
        return

    keys = sorted(params.keys())
    values_list: List[List[Any]] = []
    for key in keys:
        vals = _param_values(
            params[key],
            float_samples=float_samples,
            float_strategy=float_strategy,
            rng=rng,
        )
        if not vals:
            return
        values_list.append(vals)

    for combo in product(*values_list):
        yield {k: v for k, v in zip(keys, combo)}


def _param_count(spec: Dict[str, Any], *, float_samples: int) -> int:
    ptype = spec.get("type")
    if ptype == "choice":
        return len(spec.get("values", []) or [])
    if ptype == "int_range":
        mn = int(spec.get("min", 0))
        mx = int(spec.get("max", 0))
        step = int(spec.get("step", 1))
        if step <= 0 or mx < mn:
            return 0
        return ((mx - mn) // step) + 1
    if ptype == "float_range":
        step = spec.get("step")
        if step is not None:
            mn = float(spec.get("min", 0.0))
            mx = float(spec.get("max", 0.0))
            step = float(step)
            if step <= 0 or mx < mn:
                return 0
            return int(math.floor((mx - mn) / step)) + 1
        return max(int(float_samples), 0)
    if ptype == "bool":
        return 2
    if ptype == "fixed":
        return 1
    if ptype == "subset":
        values = spec.get("values", []) or []
        n = len(values)
        if n == 0:
            return 1
        min_items = int(spec.get("min_items", 1))
        max_items = int(spec.get("max_items", n))
        max_items = min(max_items, n)
        min_items = max(min_items, 0)
        if min_items > max_items:
            return 0
        total = 0
        for k in range(min_items, max_items + 1):
            total += math.comb(n, k)
        return total
    return 1


def _count_param_grid(
    params: Dict[str, Any],
    *,
    float_samples: int,
) -> int:
    if not params:
        return 1
    total = 1
    for key in sorted(params.keys()):
        count = _param_count(params[key], float_samples=float_samples)
        if count == 0:
            return 0
        total *= count
    return total


def _find_variant(space: Dict[str, Any], variant_name: str) -> Dict[str, Any]:
    variants = space.get("variants", []) or []
    for variant in variants:
        if variant.get("name") == variant_name:
            return variant
    return {"params": {}}


def _load_expected_cols(model_dir: Path) -> List[str]:
    oracle_csv = model_dir / "mcts_oracle.csv"
    if oracle_csv.exists():
        return pd.read_csv(oracle_csv, nrows=0).columns.tolist()
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1")
    parser.add_argument("--study", default="s6e1_008_xgb_gpu")
    parser.add_argument("--parent-id", type=int, default=None, help="Trial id to extend (defaults to best)")
    parser.add_argument("--out", default="/tmp/oracle_exhaustive.csv")
    parser.add_argument("--float-samples", type=int, default=5)
    parser.add_argument("--float-strategy", choices=["linspace", "random"], default="linspace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--lookahead", type=int, default=0, help="0 means no lookahead limit")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--target-depth",
        type=int,
        default=None,
        help="Target depth for the next action (uses parent at depth-1)",
    )
    args = parser.parse_args()

    project_dir = Path(f"projects/kaggle/{args.project}")
    exp_dir = project_dir / "experiments"
    db_path = exp_dir / "db" / "mcts.db"
    model_dir = exp_dir / "oracle" / "model"

    if not model_dir.exists():
        err("Oracle model not found.")
        return
    if not db_path.exists():
        err("MCTS DB not found.")
        return

    conn = sqlite3.connect(db_path)
    if args.target_depth is not None and args.target_depth < 1:
        err("target-depth must be >= 1.")
        return

    if args.parent_id is not None:
        row = _load_parent_by_id(conn, args.parent_id)
        if not row:
            err(f"Parent trial {args.parent_id} not found or not COMPLETE.")
            return
        if args.target_depth is not None:
            expected_parent_depth = int(args.target_depth) - 1
            if int(row[2]) != expected_parent_depth:
                err(
                    f"Parent depth {row[2]} does not match target-depth-1 "
                    f"({expected_parent_depth})."
                )
                return
    elif args.target_depth is not None:
        expected_parent_depth = int(args.target_depth) - 1
        row = _load_best_parent_at_depth(conn, args.study, expected_parent_depth)
        if not row:
            err(f"No COMPLETE parent found at depth {expected_parent_depth}.")
            return
    else:
        row = _load_best_parent(conn, args.study)
        if not row:
            err(f"Study '{args.study}' not found or empty.")
            return

    parent_id, parent_score, depth = row
    used_groups, last_step_index, prev_action_json = _reconstruct_state(conn, int(parent_id))
    conn.close()

    depth_value = int(depth) + 1

    info(f"Parent: {parent_id} | score={parent_score:.6f} | depth={depth}")
    info(f"Depth feature: {depth_value}")
    info(f"Used groups: {sorted(used_groups.keys())}")

    super_chain_path = Path("conf/preprocess/mla_super_chain.yaml")
    space = SuperChainActionSpace(super_chain_path)
    state = PipelineState(
        steps=[],
        depth=int(depth),
        used_groups=used_groups,
        last_step_index=last_step_index,
    )

    lookahead = None if args.lookahead <= 0 else args.lookahead
    actions = space.next_actions(state, lookahead=lookahead)
    if not actions:
        warn("No valid next actions available for this parent.")
        return

    info(f"Action templates: {len(actions)}")

    rng = random.Random(args.seed)
    total_candidates = 0
    for action in actions:
        space_def = space.search_spaces.get(action.template_name, {})
        variant = _find_variant(space_def, action.variant_name)
        params = variant.get("params", {}) or {}
        total_candidates += _count_param_grid(
            params,
            float_samples=args.float_samples,
        )

    info(f"Estimated candidates: {total_candidates}")

    predictor = TabularPredictor.load(str(model_dir))
    expected_cols = _load_expected_cols(model_dir)
    if not expected_cols:
        expected_cols = predictor.feature_metadata_in.get_features()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    prev_flat = parse_action_full(prev_action_json, prefix="prev_")
    chunk_rows: List[Dict[str, Any]] = []
    processed = 0
    header_written = False
    top_rows: List[Tuple[float, Dict[str, Any]]] = []

    for action in actions:
        space_def = space.search_spaces.get(action.template_name, {})
        variant = _find_variant(space_def, action.variant_name)
        params = variant.get("params", {}) or {}

        for config in _iter_param_grid(
            params,
            float_samples=args.float_samples,
            float_strategy=args.float_strategy,
            rng=rng,
        ):
            action_dict = {
                "step": action.step_name,
                "group_name": action.group_name,
                "template": action.template_name,
                "variant": action.variant_name,
                "config": config,
                "searched_index": action.searched_index,
                "original_index": action.original_index,
            }

            curr_flat = flatten_config(action_dict, prefix="")
            row = {
                "parent_score": parent_score,
                "depth": depth_value,
                "prev_duration": 0.0,
                **prev_flat,
                **curr_flat,
                "curr_action_json": json.dumps(action_dict),
            }
            chunk_rows.append(row)
            processed += 1

            if len(chunk_rows) >= args.chunk_size:
                df = pd.DataFrame(chunk_rows)
                for col in expected_cols:
                    if col == "is_improvement":
                        continue
                    if col not in df.columns:
                        df[col] = None

                if predictor.problem_type == "binary":
                    probs = predictor.predict_proba(df)
                    pos_label = 1
                    df["prob_improvement"] = probs[pos_label] if pos_label in probs.columns else probs.iloc[:, -1]
                else:
                    df["prob_improvement"] = predictor.predict(df)

                df.to_csv(out_path, mode="a", header=not header_written, index=False)
                header_written = True

                if args.top_k > 0:
                    for _, row_out in df.iterrows():
                        prob = float(row_out["prob_improvement"])
                        top_rows.append((prob, row_out.to_dict()))
                    top_rows.sort(key=lambda x: x[0], reverse=True)
                    top_rows = top_rows[: args.top_k]

                chunk_rows = []
                info(f"Processed {processed}/{total_candidates}")

    if chunk_rows:
        df = pd.DataFrame(chunk_rows)
        for col in expected_cols:
            if col == "is_improvement":
                continue
            if col not in df.columns:
                df[col] = None

        if predictor.problem_type == "binary":
            probs = predictor.predict_proba(df)
            pos_label = 1
            df["prob_improvement"] = probs[pos_label] if pos_label in probs.columns else probs.iloc[:, -1]
        else:
            df["prob_improvement"] = predictor.predict(df)

        df.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True

        if args.top_k > 0:
            for _, row_out in df.iterrows():
                prob = float(row_out["prob_improvement"])
                top_rows.append((prob, row_out.to_dict()))
            top_rows.sort(key=lambda x: x[0], reverse=True)
            top_rows = top_rows[: args.top_k]

    info(f"Saved results to {out_path}")

    if args.top_k > 0 and top_rows:
        # Deduplicate by action signature to avoid repeated rows.
        exclude = {"parent_score", "depth", "prev_duration", "prob_improvement", "curr_action_json"}
        unique_rows: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = {}
        for prob, row_out in top_rows:
            items: List[Tuple[str, Any]] = []
            for key, value in row_out.items():
                if key in exclude or key.startswith("prev_"):
                    continue
                if pd.isna(value):
                    continue
                items.append((key, value))
            items.sort(key=lambda x: x[0])
            signature = (
                row_out.get("action_group"),
                row_out.get("action_variant"),
                tuple(items),
            )
            if signature not in unique_rows or prob > unique_rows[signature][0]:
                unique_rows[signature] = (prob, row_out)

        deduped = sorted(unique_rows.values(), key=lambda x: x[0], reverse=True)
        deduped = deduped[: args.top_k]

        table = Table(title=f"Top {len(deduped)} candidates")
        table.add_column("Prob")
        table.add_column("Group")
        table.add_column("Variant")

        # Pick a few parameter columns with the most variability.
        value_sets: Dict[str, set] = {}
        for _, row in deduped:
            for key, value in row.items():
                if key in exclude or key.startswith("prev_") or key in {"action_group", "action_variant"}:
                    continue
                if pd.isna(value):
                    continue
                value_sets.setdefault(key, set()).add(str(value))

        sorted_params = sorted(
            (k for k, vals in value_sets.items() if len(vals) > 1),
            key=lambda k: (-len(value_sets[k]), k),
        )
        if not sorted_params:
            sorted_params = sorted(value_sets.keys())
        show_params = sorted_params[:3]

        for key in show_params:
            table.add_column(key)

        for prob, row_out in deduped:
            base_cols = [
                f"{prob:.6f}",
                str(row_out.get("action_group", "")),
                str(row_out.get("action_variant", "")),
            ]
            extra_cols = [str(row_out.get(key, ""))[:20] for key in show_params]
            table.add_row(*(base_cols + extra_cols))
        console.print(table)


if __name__ == "__main__":
    main()
