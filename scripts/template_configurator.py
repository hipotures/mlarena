"""
Rich-based configurator that builds the ml_runner CLI command (no hidden state).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Dict

import yaml
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
console = Console()


def load_templates(project: str) -> Dict[str, Dict]:
    project_root = REPO_ROOT / "projects" / "kaggle" / project
    templates_dir = project_root / "templates"
    model_cfg = yaml.safe_load((templates_dir / "model.yaml").read_text()) or {}
    preprocess_cfg = yaml.safe_load((templates_dir / "preprocess.yaml").read_text()) or {}
    return {
        "model": model_cfg.get("templates", {}),
        "preprocess": preprocess_cfg.get("templates", {}),
        "root": project_root,
    }


def choose_template(kind: str, templates: Dict[str, Dict]) -> str:
    table = Table(title=f"{kind.title()} templates", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("name", style="cyan")
    table.add_column("details", style="green")
    names = sorted(templates.keys())
    for idx, name in enumerate(names, 1):
        payload = templates[name] or {}
        if kind == "model":
            hyper = (payload.get("config") or {}).get("hyperparameters") or {}
            preset = hyper.get("presets") or hyper.get("preset") or "-"
            time_limit = hyper.get("time_limit")
            gpu = hyper.get("use_gpu")
            details = f"preset={preset}, time={time_limit or '-'}s, gpu={'yes' if gpu else 'no'}"
        else:
            details = f"module={payload.get('module')}, cache={'yes' if payload.get('cache') else 'no'}"
        table.add_row(str(idx), name, details)
    console.print(table)
    while True:
        choice = console.input(f"Select {kind} template [1-{len(names)}]: ").strip()
        try:
            idx = int(choice)
        except ValueError:
            continue
        if 1 <= idx <= len(names):
            return names[idx - 1]


def build_command(project: str, model_template: str, preprocess_template: str, use_preprocessed: bool, skip_submit: bool, auto_submit: bool) -> str:
    parts = [
        "uv run python scripts/ml_runner.py",
        f"--project {shlex.quote(project)}",
        f"--model-template {shlex.quote(model_template)}",
        f"--preprocess-template {shlex.quote(preprocess_template)}",
    ]
    if use_preprocessed:
        parts.append("--use-preprocessed")
    if skip_submit:
        parts.append("--skip-submit")
    if auto_submit:
        parts.append("--auto-submit")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Template configurator (builds ml_runner command)")
    parser.add_argument("--project", required=True)
    parser.add_argument("--run", action="store_true", help="Execute the generated command")
    parser.add_argument("--use-preprocessed", action="store_true")
    parser.add_argument("--skip-submit", action="store_true")
    parser.add_argument("--auto-submit", action="store_true")
    args = parser.parse_args()

    templates = load_templates(args.project)
    model_choice = choose_template("model", templates["model"])
    preprocess_choice = choose_template("preprocess", templates["preprocess"])

    cmd = build_command(
        args.project,
        model_choice,
        preprocess_choice,
        args.use_preprocessed,
        args.skip_submit,
        args.auto_submit,
    )
    console.print(f"\n[bold]Generated command:[/bold] {cmd}")
    if args.run:
        console.print("[cyan]Running command...[/cyan]")
        subprocess.run(cmd, shell=True, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
