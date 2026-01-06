#!/usr/bin/env python
"""Generate random preprocessing experiment templates and enqueue them."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import importlib.util

# Add src to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlarena.utils.queue import TaskQueue


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _load_project_constants(project_root: Path) -> Dict[str, Any]:
    config_path = project_root / "code" / "utils" / "config.py"
    if not config_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("mlarena_project_config", config_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]

    keys = [
        "TARGET_COLUMN",
        "ID_COLUMN",
        "IGNORED_COLUMNS",
        "AUTOGLUON_PROBLEM_TYPE",
    ]
    values = {}
    for key in keys:
        if hasattr(module, key):
            values[key] = getattr(module, key)
    return values


def _load_eda_summary(project_root: Path, override_path: str | None) -> Dict[str, Any] | None:
    if override_path:
        path = Path(override_path)
    else:
        path = project_root / "experiments" / "eda" / "artifacts" / "eda" / "eda_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _infer_datetime_type(type_name: str) -> bool:
    lowered = type_name.lower()
    return "date" in lowered or "time" in lowered


def _summarize_eda(eda: Dict[str, Any], exclude_cols: List[str]) -> Dict[str, Any]:
    train_summary = eda.get("train", {}).get("summary", {})
    variables = train_summary.get("variables", {}) or {}
    filtered_vars = {
        name: details for name, details in variables.items()
        if name not in exclude_cols
    }

    numeric_cols = [name for name, v in filtered_vars.items() if v.get("type") == "Numeric"]
    categorical_cols = [name for name, v in filtered_vars.items() if v.get("type") == "Categorical"]
    datetime_cols = [
        name for name, v in filtered_vars.items()
        if isinstance(v.get("type"), str) and _infer_datetime_type(v["type"])
    ]

    missing_total = sum(int(v.get("n_missing", 0) or 0) for v in filtered_vars.values())
    has_missing = missing_total > 0

    max_cat_distinct = 0
    if categorical_cols:
        max_cat_distinct = max(int(filtered_vars[name].get("n_distinct", 0) or 0) for name in categorical_cols)

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "missing_total": missing_total,
        "has_missing": has_missing,
        "max_cat_distinct": max_cat_distinct,
    }


def _apply_eda_filters(
    preprocessors: List[Dict[str, Any]],
    eda_stats: Dict[str, Any] | None,
    problem_type: str | None,
) -> List[str]:
    decisions: List[str] = []

    def disable_group(group: str, reason: str) -> None:
        for preproc in preprocessors:
            if preproc.get("group") == group:
                preproc["enabled"] = False
        decisions.append(f"group '{group}' disabled: {reason}")

    if eda_stats is None:
        return decisions

    if not eda_stats.get("has_missing", False):
        disable_group("imputer", "no missing values detected in EDA")

    if not eda_stats.get("categorical_cols"):
        disable_group("encoding", "no categorical columns detected in EDA")
        disable_group("rare_category", "no categorical columns detected in EDA")

    if not eda_stats.get("numeric_cols"):
        disable_group("outlier", "no numeric columns detected in EDA")
        disable_group("scaler", "no numeric columns detected in EDA")
        disable_group("feature_engineer", "no numeric columns detected in EDA")
        disable_group("feature_selector", "no numeric columns detected in EDA")

    if not eda_stats.get("datetime_cols"):
        disable_group("datetime", "no datetime columns detected in EDA")

    if problem_type and problem_type.lower() == "regression":
        disable_group("imbalance", "problem_type=regression")

    return decisions


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _resolve_template_path(kind: str, name: str, project_root: Path) -> Path:
    local_path = project_root / "templates" / kind / f"{name}.yaml"
    if local_path.exists():
        return local_path
    global_path = REPO_ROOT / "src" / "mlarena" / "templates" / kind / f"{name}.yaml"
    if global_path.exists():
        return global_path
    raise FileNotFoundError(f"Template not found: {kind}/{name}.yaml")


def _weighted_choice(items: List[Tuple[Any, float]], rng: random.Random) -> Any:
    total = sum(weight for _, weight in items)
    if total <= 0:
        raise ValueError("Total weight must be positive")
    r = rng.random() * total
    upto = 0.0
    for value, weight in items:
        upto += weight
        if r <= upto:
            return value
    return items[-1][0]


def _weighted_sample_without_replacement(
    items: List[Tuple[Any, float]],
    k: int,
    rng: random.Random
) -> List[Any]:
    pool = items[:]
    selected: List[Any] = []
    for _ in range(k):
        choice = _weighted_choice(pool, rng)
        selected.append(choice)
        pool = [(v, w) for v, w in pool if v != choice]
    return selected


def _sample_param(spec: Any, rng: random.Random) -> Any:
    if isinstance(spec, dict) and "type" in spec:
        ptype = spec["type"]
        if ptype == "choice":
            values = spec.get("values", [])
            weights = spec.get("weights")
            if not values:
                raise ValueError("choice spec requires non-empty values")
            if weights:
                return rng.choices(values, weights=weights, k=1)[0]
            return rng.choice(values)
        if ptype == "int_range":
            min_v = int(spec["min"])
            max_v = int(spec["max"])
            step = int(spec.get("step", 1))
            if step <= 0:
                raise ValueError("int_range step must be positive")
            count = (max_v - min_v) // step
            return min_v + step * rng.randint(0, count)
        if ptype == "float_range":
            min_v = float(spec["min"])
            max_v = float(spec["max"])
            step = spec.get("step")
            if step:
                step = float(step)
                count = int(math.floor((max_v - min_v) / step))
                value = min_v + step * rng.randint(0, count)
            else:
                value = rng.uniform(min_v, max_v)
            if "round" in spec:
                value = round(value, int(spec["round"]))
            return value
        if ptype == "log_uniform":
            min_v = float(spec["min"])
            max_v = float(spec["max"])
            if min_v <= 0 or max_v <= 0:
                raise ValueError("log_uniform requires min/max > 0")
            value = math.exp(rng.uniform(math.log(min_v), math.log(max_v)))
            if "round" in spec:
                value = round(value, int(spec["round"]))
            return value
        if ptype == "bool":
            p_true = float(spec.get("p_true", 0.5))
            return rng.random() < p_true
        if ptype == "subset":
            values = spec.get("values", [])
            if not values:
                return []
            min_items = int(spec.get("min_items", 1))
            max_items = int(spec.get("max_items", len(values)))
            max_items = max(min_items, min(max_items, len(values)))
            size = rng.randint(min_items, max_items)
            subset = rng.sample(values, size)
            if spec.get("sort", False):
                subset = sorted(subset)
            return subset
        if ptype == "fixed":
            return spec.get("value")
        raise ValueError(f"Unknown param type: {ptype}")

    if isinstance(spec, list):
        if not spec:
            return []
        return rng.choice(spec)

    return spec


def _sample_variant(preproc: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    variants = preproc.get("variants")
    if not variants:
        return {"name": "default", "params": preproc.get("params", {})}

    weighted = []
    for variant in variants:
        weight = float(variant.get("weight", 1.0))
        weighted.append((variant, weight))

    chosen = _weighted_choice(weighted, rng)
    return chosen


def _collect_requires(preproc: Dict[str, Any], variant: Dict[str, Any]) -> List[Any]:
    requires: List[Any] = []
    if isinstance(preproc.get("requires_preproc"), list):
        requires.extend(preproc.get("requires_preproc", []))
    if isinstance(variant.get("requires_preproc"), list):
        requires.extend(variant.get("requires_preproc", []))
    return requires


def _resolve_requirement(
    requirement: Any,
    name_map: Dict[str, Dict[str, Any]],
    group_map: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, str]:
    if isinstance(requirement, dict):
        if "name" in requirement:
            return ("preproc", str(requirement["name"]))
        if "group" in requirement:
            return ("group", str(requirement["group"]))
        raise ValueError(f"Invalid requires_preproc dict: {requirement}")

    if not isinstance(requirement, str):
        raise ValueError(f"Invalid requires_preproc entry: {requirement!r}")

    if requirement in name_map:
        return ("preproc", requirement)
    if requirement in group_map:
        return ("group", requirement)

    # Unknown requirement string
    raise ValueError(f"Unknown requires_preproc target: '{requirement}'")


def _is_imputer_requirement(requirement: Any) -> bool:
    if isinstance(requirement, str):
        return requirement == "imputer"
    if isinstance(requirement, dict):
        return requirement.get("group") == "imputer" or requirement.get("name") == "imputer"
    return False


def _resolve_requires(
    selected_items: List[Dict[str, Any]],
    group_map: Dict[str, List[Dict[str, Any]]],
    name_map: Dict[str, Dict[str, Any]],
    rng: random.Random,
    reserved_groups: set,
    eda_stats: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    selected_by_group: Dict[str, Dict[str, Any]] = {}
    selected_names: set[str] = set()

    for item in selected_items:
        preproc = item["preproc"]
        group = preproc.get("group", preproc["name"])
        selected_by_group[group] = item
        selected_names.add(preproc["name"])

    def add_preproc(preproc: Dict[str, Any]) -> Dict[str, Any]:
        group = preproc.get("group", preproc["name"])
        if preproc["name"] in selected_names:
            return selected_by_group[group]
        if group in selected_by_group:
            existing = selected_by_group[group]
            if existing["preproc"]["name"] != preproc["name"]:
                raise ValueError(
                    f"requires_preproc conflict: group '{group}' already has "
                    f"'{existing['preproc']['name']}' but requirement asked for '{preproc['name']}'"
                )
            return existing
        if group in reserved_groups:
            raise ValueError(
                f"requires_preproc conflict: group '{group}' already present in base preprocess chain"
            )
        variant = _sample_variant(preproc, rng)
        item = {"preproc": preproc, "variant": variant}
        selected_items.append(item)
        selected_by_group[group] = item
        selected_names.add(preproc["name"])
        return item

    visiting: set[str] = set()
    stack: List[str] = []

    def _push(node: str) -> None:
        if node in visiting:
            cycle = " -> ".join(stack + [node])
            raise ValueError(f"requires_preproc cycle detected: {cycle}")
        visiting.add(node)
        stack.append(node)

    def _pop() -> None:
        node = stack.pop()
        visiting.discard(node)

    def resolve_item(item: Dict[str, Any]) -> None:
        preproc = item["preproc"]
        variant = item["variant"]
        group = preproc.get("group", preproc["name"])
        _push(f"group:{group}")
        _push(f"preproc:{preproc['name']}")

        requirements = _collect_requires(preproc, variant)
        for req in requirements:
            if eda_stats is not None and not eda_stats.get("has_missing", False):
                if _is_imputer_requirement(req):
                    continue
            req_kind, req_value = _resolve_requirement(req, name_map, group_map)

            if req_kind == "group" and req_value == "imputer" and eda_stats is not None:
                if not eda_stats.get("has_missing", False):
                    continue

            node_id = f"{req_kind}:{req_value}"
            if node_id in visiting:
                cycle = " -> ".join(stack + [node_id])
                raise ValueError(f"requires_preproc cycle detected: {cycle}")

            if req_kind == "preproc":
                if req_value in selected_names:
                    continue
                if req_value not in name_map:
                    raise ValueError(f"requires_preproc unknown preproc: '{req_value}'")
                dep_item = add_preproc(name_map[req_value])
                resolve_item(dep_item)
            else:
                if req_value in selected_by_group or req_value in reserved_groups:
                    continue
                if req_value not in group_map:
                    raise ValueError(f"requires_preproc unknown group: '{req_value}'")
                weighted_items = [
                    (item, float(item.get("weight", 1.0))) for item in group_map[req_value]
                ]
                chosen = _weighted_choice(weighted_items, rng)
                dep_item = add_preproc(chosen)
                resolve_item(dep_item)

        _pop()
        _pop()

    for item in list(selected_items):
        resolve_item(item)

    return selected_items


def _order_by_requires(
    selected_items: List[Dict[str, Any]],
    name_map: Dict[str, Dict[str, Any]],
    group_map: Dict[str, List[Dict[str, Any]]],
    reserved_groups: set,
    eda_stats: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    item_by_name = {item["preproc"]["name"]: item for item in selected_items}
    group_to_name = {
        item["preproc"].get("group", item["preproc"]["name"]): item["preproc"]["name"]
        for item in selected_items
    }
    order_index = {item["preproc"]["name"]: idx for idx, item in enumerate(selected_items)}

    edges: Dict[str, set[str]] = {name: set() for name in item_by_name}
    indegree: Dict[str, int] = {name: 0 for name in item_by_name}

    for item in selected_items:
        preproc = item["preproc"]
        variant = item["variant"]
        current_name = preproc["name"]
        requirements = _collect_requires(preproc, variant)
        for req in requirements:
            if eda_stats is not None and not eda_stats.get("has_missing", False):
                if _is_imputer_requirement(req):
                    continue

            req_kind, req_value = _resolve_requirement(req, name_map, group_map)
            if req_kind == "group":
                if req_value in reserved_groups:
                    continue
                req_name = group_to_name.get(req_value)
                if not req_name:
                    raise ValueError(f"requires_preproc group not selected: '{req_value}'")
            else:
                req_name = req_value
                if req_name not in item_by_name:
                    raise ValueError(f"requires_preproc preproc not selected: '{req_name}'")

            if req_name == current_name:
                raise ValueError(f"requires_preproc self-cycle: '{current_name}'")

            if current_name not in edges[req_name]:
                edges[req_name].add(current_name)
                indegree[current_name] += 1

    ready = [name for name, deg in indegree.items() if deg == 0]
    ready.sort(key=lambda n: order_index.get(n, 0))
    ordered_names: List[str] = []

    while ready:
        name = ready.pop(0)
        ordered_names.append(name)
        for dep in sorted(edges[name], key=lambda n: order_index.get(n, 0)):
            indegree[dep] -= 1
            if indegree[dep] == 0:
                ready.append(dep)
        ready.sort(key=lambda n: order_index.get(n, 0))

    if len(ordered_names) != len(item_by_name):
        remaining = [name for name, deg in indegree.items() if deg > 0]
        raise ValueError(f"requires_preproc cycle detected among: {', '.join(remaining)}")

    return [item_by_name[name] for name in ordered_names]


def _merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for cfg in configs:
        for key, value in cfg.items():
            merged[key] = value
    return merged


def _build_module_payload(
    preproc: Dict[str, Any],
    variant: Dict[str, Any],
    project_root: Path,
    template_cache: Dict[str, Dict[str, Any]],
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    template_name = preproc.get("template", preproc["name"])
    if template_name not in template_cache:
        template_path = _resolve_template_path("preprocess", template_name, project_root)
        template_cache[template_name] = _load_yaml(template_path)

    base_template = deepcopy(template_cache[template_name])
    base_config = deepcopy(base_template.get("config", {}))
    extra_base = deepcopy(preproc.get("base_config", {}))

    sampled_params = {}
    for key, spec in variant.get("params", {}).items():
        sampled_params[key] = _sample_param(spec, rng)

    final_config = _merge_configs(base_config, extra_base, sampled_params)
    base_template["config"] = final_config

    module_config_signature = {
        "module": base_template.get("module"),
        "config": final_config,
    }

    return base_template, module_config_signature


def _compute_signature(base_chain: List[str], module_configs: Dict[str, Dict[str, Any]]) -> str:
    payload = {
        "base_chain": base_chain,
        "modules": module_configs,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _load_signature_registry(path: Path) -> set:
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        return set(data.get("signatures", []))
    return set()


def _save_signature_registry(path: Path, signatures: set) -> None:
    payload = {
        "signatures": sorted(signatures),
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate random preprocess experiment templates")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "conf" / "random_preprocess_config.yaml"),
        help="Path to generator config YAML",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    
    # Overrides (support both snake_case from YAML and standard kebab-case)
    parser.add_argument("--project", help="Override project name")
    parser.add_argument("--prefix", "--experiment_prefix", "--experiment-prefix", help="Override experiment_prefix")
    parser.add_argument("--start_index", "--start-index", type=int, help="Override start_index")
    parser.add_argument("--id_width", "--id-width", type=int, help="Override id_width")
    parser.add_argument("--experiments", type=int, help="Override number of experiments to generate")
    parser.add_argument("--seed", "--random_seed", "--random-seed", type=int, help="Override random_seed")
    parser.add_argument("--model_template", "--model-template", help="Override base model template name")
    parser.add_argument("--preprocess_template", "--preprocess-template", help="Override base preprocess template name")
    parser.add_argument("--preprocessors_per_experiment", "--preprocessors-per-experiment", "--preprocessors-per-exp", help="Override preprocessors_per_experiment")
    parser.add_argument("--max_unique_attempts", "--max-unique-attempts", "--max-attempts", type=int, help="Override max_unique_attempts")
    parser.add_argument("--signature_registry", "--signature-registry", help="Override signature_registry filename")
    parser.add_argument("--auto_filter_by_eda", "--auto-filter-by-eda", "--auto-filter", action="store_true", default=None, help="Enable auto_filter_by_eda")
    parser.add_argument("--no_auto_filter", "--no-auto-filter", action="store_false", dest="auto_filter_by_eda", help="Disable auto_filter_by_eda")
    parser.add_argument("--eda_summary_path", "--eda-summary-path", "--eda-summary", help="Override eda_summary_path")

    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_yaml(config_path)

    # Apply overrides
    if args.project: config["project"] = args.project
    if args.prefix: config["experiment_prefix"] = args.prefix
    if args.start_index is not None: config["start_index"] = args.start_index
    if args.id_width is not None: config["id_width"] = args.id_width
    if args.experiments is not None: config["experiments"] = args.experiments
    if args.seed is not None: config["random_seed"] = args.seed
    if args.model_template: config["model_template"] = args.model_template
    if args.preprocess_template: config["preprocess_template"] = args.preprocess_template
    if args.preprocessors_per_experiment: config["preprocessors_per_experiment"] = args.preprocessors_per_experiment
    if args.max_unique_attempts is not None: config["max_unique_attempts"] = args.max_unique_attempts
    if args.signature_registry: config["signature_registry"] = args.signature_registry
    if args.auto_filter_by_eda is not None: config["auto_filter_by_eda"] = args.auto_filter_by_eda
    if args.eda_summary_path: config["eda_summary_path"] = args.eda_summary_path

    project = config.get("project")
    if not project:
        print("Error: config must include 'project'")
        return 1

    project_root = REPO_ROOT / "projects" / "kaggle" / project
    if not project_root.exists():
        print(f"Error: project not found: {project_root}")
        return 1

    model_template_name = config.get("model_template")
    if not model_template_name:
        print("Error: config must include 'model_template'")
        return 1

    project_constants = _load_project_constants(project_root)
    target_col = project_constants.get("TARGET_COLUMN")
    id_col = project_constants.get("ID_COLUMN")
    ignored_cols = project_constants.get("IGNORED_COLUMNS", [])
    if ignored_cols is None:
        ignored_cols = []
    exclude_cols = [c for c in [id_col, target_col, *ignored_cols] if c]
    problem_type = project_constants.get("AUTOGLUON_PROBLEM_TYPE")

    auto_filter_by_eda = config.get("auto_filter_by_eda", True)
    eda_summary_override = config.get("eda_summary_path")
    eda_summary = _load_eda_summary(project_root, eda_summary_override) if auto_filter_by_eda else None
    eda_stats = _summarize_eda(eda_summary, exclude_cols) if eda_summary else None

    preprocess_template_name = config.get("preprocess_template")
    prefix = str(config.get("experiment_prefix", "exp"))
    count = int(config.get("experiments", 1))
    id_width = int(config.get("id_width", 4))
    start_index = int(config.get("start_index", 1))
    rng = random.Random(int(config.get("random_seed", 42)))

    raw_preprocess_per_exp = config.get("preprocessors_per_experiment", 1)
    max_possible_preprocs = 0
    if isinstance(raw_preprocess_per_exp, str) and "-" in raw_preprocess_per_exp:
        try:
            _, high = map(int, raw_preprocess_per_exp.split("-"))
            max_possible_preprocs = high
        except ValueError:
            print(f"Error: invalid format for preprocessors_per_experiment: {raw_preprocess_per_exp}")
            return 1
    else:
        max_possible_preprocs = int(raw_preprocess_per_exp)

    if max_possible_preprocs < 0:
        print("Error: preprocessors_per_experiment must be >= 0")
        return 1

    max_unique_attempts = int(config.get("max_unique_attempts", 200))

    model_dir = project_root / "templates" / "model"
    preprocess_dir = project_root / "templates" / "preprocess"
    model_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    # Load base model template
    base_model_path = _resolve_template_path("model", model_template_name, project_root)
    base_model_template = _load_yaml(base_model_path)

    # Load base preprocess chain if provided
    base_chain: List[str] = []
    if preprocess_template_name:
        base_chain_path = _resolve_template_path("preprocess", preprocess_template_name, project_root)
        base_chain_payload = _load_yaml(base_chain_path)
        base_chain = list(base_chain_payload.get("chain", []))

    # Preprocessor registry
    preprocessors = list(config.get("preprocessors", []))
    decisions = _apply_eda_filters(preprocessors, eda_stats, problem_type) if auto_filter_by_eda else []
    preprocessors = [p for p in preprocessors if p.get("enabled", True)]
    
    # Build group/name maps
    group_map: Dict[str, List[Dict[str, Any]]] = {}
    name_map: Dict[str, Dict[str, Any]] = {}
    for preproc in preprocessors:
        group = preproc.get("group", preproc["name"])
        group_map.setdefault(group, []).append(preproc)
        name_map[preproc["name"]] = preproc

    # Resolve base chain groups to avoid duplicates
    reserved_groups = set()
    if base_chain:
        for entry in base_chain:
            try:
                entry_path = _resolve_template_path("preprocess", entry, project_root)
                entry_payload = _load_yaml(entry_path)
                module_name = entry_payload.get("module", entry)
            except FileNotFoundError:
                module_name = entry
            for preproc in preprocessors:
                if preproc.get("name") == module_name or preproc.get("template") == entry:
                    reserved_groups.add(preproc.get("group", preproc["name"]))
                    break

    available_groups = [g for g in group_map.keys() if g not in reserved_groups]
    
    if not preprocessors and max_possible_preprocs > 0:
        print("Error: no enabled preprocessors in config")
        return 1

    if max_possible_preprocs > len(available_groups):
        print(
            f"Error: preprocessors_per_experiment={max_possible_preprocs} exceeds available groups "
            f"({len(available_groups)}) after base chain"
        )
        return 1

    # Signature registry
    registry_name = config.get("signature_registry", ".random_preprocess_signatures.json")
    signature_path = preprocess_dir / registry_name if registry_name else None
    used_signatures = set()
    if signature_path:
        used_signatures.update(_load_signature_registry(signature_path))

    generated: List[Dict[str, Any]] = []
    template_cache: Dict[str, Dict[str, Any]] = {}
    attempt_count = 0
    current_index = start_index
    overwrite_all: bool | None = None
    auto_append: bool = False

    # Regex for finding next index if auto_append is chosen
    index_pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{id_width}}})\.yaml$")

    while len(generated) < count:
        if attempt_count >= max_unique_attempts:
            print("Error: exceeded max_unique_attempts while searching for unique configs")
            return 1

        exp_id = f"{prefix}{current_index:0{id_width}d}"

        # Sample number of preprocessors for this experiment
        if isinstance(raw_preprocess_per_exp, str) and "-" in raw_preprocess_per_exp:
            low, high = map(int, raw_preprocess_per_exp.split("-"))
            current_preproc_count = rng.randint(low, high)
        else:
            current_preproc_count = int(raw_preprocess_per_exp)
        
        current_preproc_count = min(current_preproc_count, len(available_groups))

        # Sample groups
        group_weights = [
            (group, sum(float(p.get("weight", 1.0)) for p in group_map[group]))
            for group in available_groups
        ]
        selected_groups = _weighted_sample_without_replacement(group_weights, current_preproc_count, rng)

        selected_items: List[Dict[str, Any]] = []
        for group in selected_groups:
            group_items = group_map[group]
            weighted_items = [
                (item, float(item.get("weight", 1.0))) for item in group_items
            ]
            preproc = _weighted_choice(weighted_items, rng)
            variant = _sample_variant(preproc, rng)
            selected_items.append({"preproc": preproc, "variant": variant})

        selected_items = _resolve_requires(
            selected_items=selected_items,
            group_map=group_map,
            name_map=name_map,
            rng=rng,
            reserved_groups=reserved_groups,
            eda_stats=eda_stats,
        )
        selected_items = _order_by_requires(
            selected_items=selected_items,
            name_map=name_map,
            group_map=group_map,
            reserved_groups=reserved_groups,
            eda_stats=eda_stats,
        )

        module_payloads = {}
        module_signature_configs = {}
        module_template_names = []

        for item in selected_items:
            preproc = item["preproc"]
            variant = item["variant"]
            payload, signature_config = _build_module_payload(
                preproc,
                variant,
                project_root,
                template_cache,
                rng,
            )

            module_name = preproc["name"]
            module_template_name = f"{exp_id}-{module_name}"
            module_template_names.append(module_template_name)
            module_payloads[module_template_name] = payload
            module_signature_configs[module_name] = signature_config

        chain = base_chain + module_template_names
        signature = _compute_signature(base_chain, module_signature_configs)

        if signature in used_signatures:
            attempt_count += 1
            continue

        # Check for existing files
        model_path = model_dir / f"{exp_id}.yaml"
        chain_path = preprocess_dir / f"{exp_id}.yaml"
        module_paths = [preprocess_dir / f"{name}.yaml" for name in module_template_names]
        existing_paths = [p for p in [model_path, chain_path, *module_paths] if p.exists()]

        if existing_paths:
            if auto_append:
                current_index += 1
                continue

            if overwrite_all is None:
                print(f"\nFiles already exist for experiment '{exp_id}':")
                for path in existing_paths:
                    print(f"  - {path}")
                reply = input("Overwrite all (Y) or append to existing (N)? [y/N]: ").strip().lower()
                if reply in {"y", "yes"}:
                    overwrite_all = True
                    decision = True
                    # Clear signature registry if overwriting all
                    if signature_path and signature_path.exists():
                        print(f"Clearing signature registry: {signature_path}")
                        signature_path.unlink()
                    used_signatures.clear()
                else:
                    auto_append = True
                    current_index += 1
                    continue
            else:
                decision = overwrite_all

            if not decision:
                current_index += 1
                continue

        if not args.dry_run:
            # Write model template
            model_payload = deepcopy(base_model_template)
            model_payload["preprocess_template"] = exp_id
            _save_yaml(model_path, model_payload)

            # Write chain template
            chain_payload = {"chain": chain}
            _save_yaml(chain_path, chain_payload)

            # Write module templates
            for template_name, payload in module_payloads.items():
                module_path = preprocess_dir / f"{template_name}.yaml"
                _save_yaml(module_path, payload)

        generated.append(
            {
                "exp_id": exp_id,
                "model_template": exp_id,
                "chain_template": exp_id,
                "modules": module_template_names,
            }
        )
        used_signatures.add(signature)
        current_index += 1
        attempt_count = 0

    if signature_path and not args.dry_run:
        _save_signature_registry(signature_path, used_signatures)

    # Queue tasks
    queue_ids: List[int] = []
    queue_cfg = config.get("queue", {})
    if queue_cfg.get("enable", True) and not args.dry_run:
        queue = TaskQueue(project_root)
        for gen in generated:
            task_id = queue.add_task_from_template(
                model_template=gen["model_template"],
                preprocess_template=None,
                priority=queue_cfg.get("priority"),
                submit=queue_cfg.get("submit"),
                git=queue_cfg.get("git"),
                overrides=queue_cfg.get("overrides"),
            )
            queue_ids.append(task_id)

    # Summary
    if eda_stats:
        print("\nEDA summary:")
        print(
            f"- numeric_cols={len(eda_stats['numeric_cols'])}, "
            f"categorical_cols={len(eda_stats['categorical_cols'])}, "
            f"datetime_cols={len(eda_stats['datetime_cols'])}, "
            f"missing_total={eda_stats['missing_total']}, "
            f"max_cat_distinct={eda_stats['max_cat_distinct']}"
        )
    if decisions:
        print("\nAuto-filter decisions:")
        for decision in decisions:
            print(f"- {decision}")

    print("\nGenerated experiments:")
    for gen in generated:
        print(f"- {gen['exp_id']}: model={gen['model_template']} chain={gen['chain_template']}")
        for module_name in gen["modules"]:
            print(f"    - {module_name}")

    if queue_ids:
        print("\nAdded to queue:")
        for exp_id, task_id in zip([g["exp_id"] for g in generated], queue_ids):
            print(f"- {exp_id}: task #{task_id}")

    print("\nHow to run the queue:")
    print(f"  uv run python scripts/task_queue.py --project {project} run")
    print(f"  uv run python scripts/task_queue.py --project {project} list")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())