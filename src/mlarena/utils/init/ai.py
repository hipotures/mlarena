"""AI-powered detection and logging for init."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console

# AutoGluon supported metrics per problem type
# Source: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html
AUTOGLUON_METRICS = {
    "binary": [
        "roc_auc", "accuracy", "balanced_accuracy", "f1", "mcc", "precision", "recall",
        "log_loss", "pac_score"
    ],
    "multiclass": [
        "accuracy", "balanced_accuracy", "mcc", "log_loss", "pac_score",
        "quadratic_kappa", "precision_macro", "precision_micro", "precision_weighted",
        "recall_macro", "recall_micro", "recall_weighted",
        "f1_macro", "f1_micro", "f1_weighted",
        "roc_auc_ovo_macro", "roc_auc_ovr_macro"
    ],
    "regression": [
        "root_mean_squared_error", "mean_squared_error", "mean_absolute_error",
        "r2", "pearsonr", "median_absolute_error"
    ],
}


def utc_now() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_and_fix_metric(
    problem_type: str,
    metric: str,
    kaggle_metric_name: str,
    project_root: Path,
    console: Console,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Validate if metric is supported by AutoGluon and suggest alternative if not.

    Args:
        problem_type: AutoGluon problem type (binary, multiclass, regression)
        metric: Detected AutoGluon metric
        kaggle_metric_name: Original Kaggle metric name
        project_root: Project root directory
        console: Rich console for output

    Returns:
        (validated_metric, ai_log_or_None)
    """
    supported_metrics = AUTOGLUON_METRICS.get(problem_type, [])

    if metric in supported_metrics:
        return metric, None

    # Metric not supported - ask AI for best alternative
    console.print(f"[yellow]⚠ Warning: '{metric}' not supported by AutoGluon for {problem_type}[/yellow]")
    console.print(f"[yellow]  Kaggle metric: {kaggle_metric_name}[/yellow]")
    console.print(f"[yellow]  Supported metrics: {', '.join(supported_metrics)}[/yellow]")

    # Import AI helper
    repo_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo_root / "scripts"))
    from ai_helper import call_ai_json

    prompt = f"""You are an AutoGluon expert. Select the best AutoGluon metric as a substitute for a Kaggle competition metric.

KAGGLE METRIC: {kaggle_metric_name}
PROBLEM TYPE: {problem_type}
INITIALLY DETECTED METRIC: {metric}

AVAILABLE AUTOGLUON METRICS FOR {problem_type}:
{', '.join(supported_metrics)}

Select the metric that best approximates the Kaggle evaluation metric. Consider:
- Correlation with Kaggle metric
- Common practice in ML competitions
- Optimization direction alignment

Return ONLY valid JSON (no markdown, no explanation):
{{"selected_metric": "exact_metric_name", "reasoning": "1-2 sentence explanation"}}"""

    from rich.progress import Progress, SpinnerColumn, TextColumn

    start_ai = time.perf_counter()
    ai_result: Dict[str, Any] = {}
    model = "gemini-2.5-flash"
    status = "failed"
    error_text = ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Asking AI to select alternative metric...", total=None)

        try:
            ai_result, model = call_ai_json(prompt, primary="gemini", retries=2)
            status = "success"
        except Exception as e:
            error_text = str(e)
            console.print(f"[yellow]AI metric selection failed: {e}[/yellow]")
        finally:
            progress.remove_task(task)
            ai_duration = time.perf_counter() - start_ai

    # Build AI log
    ai_log = {
        "command": f'echo "<prompt>" | gemini --model {model}',
        "model": model,
        "prompt": prompt,
        "response": ai_result if ai_result else {},
        "metadata": {
            "original_metric": metric,
            "kaggle_metric": kaggle_metric_name,
            "problem_type": problem_type,
            "duration_seconds": round(ai_duration, 3),
            "status": status,
            "error": error_text,
        },
    }

    if status == "success" and "selected_metric" in ai_result:
        selected = ai_result["selected_metric"]
        reasoning = ai_result.get("reasoning", "")

        if selected in supported_metrics:
            console.print(f"[green]✓ AI selected alternative: {selected}[/green]")
            console.print(f"[dim]  Reasoning: {reasoning}[/dim]")
            return selected, ai_log
        else:
            console.print(f"[red]AI returned invalid metric: {selected}[/red]")

    # Fallback to safe defaults
    fallback_map = {
        "binary": "roc_auc",
        "multiclass": "log_loss",
        "regression": "root_mean_squared_error",
    }
    fallback = fallback_map.get(problem_type, "accuracy")
    console.print(f"[yellow]⚠ Using fallback metric: {fallback}[/yellow]")

    return fallback, ai_log


def log_ai_interaction(
    project_root: Path, log_type: str, prompt: str, response: str, metadata: Optional[Dict] = None
) -> Path:
    """Log AI request/response to project logs directory."""
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{timestamp}_{log_type}.json"

    log_entry = {
        "timestamp": utc_now(),
        "log_type": log_type,
        "prompt": prompt,
        "response": response,
        "metadata": metadata or {},
    }

    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2)

    return log_file


def detect_problem_type_and_metric(
    eval_text: str,
    competition_slug: str,
    project_root: Path,
    console: Console,
) -> Tuple[Optional[str], Optional[str], Optional[bool], Dict[str, Any]]:
    """Use AI to detect problem type, metric, and submission format from Kaggle evaluation text.

    Returns:
        (problem_type, metric, submit_probabilities, ai_log) where ai_log contains full AI interaction data
    """
    # Import AI helper
    repo_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo_root / "scripts"))
    from ai_helper import call_ai_json

    prompt = f"""You are a Kaggle competition expert analyzing evaluation metrics.

Given the Evaluation section from a Kaggle competition, determine:
1. problem_type: "binary", "regression", or "multiclass"
2. metric: AutoGluon-compatible metric name
3. submit_probabilities: true if the competition expects probability outputs in the submission (e.g., ROC AUC or log loss), false if it expects class labels or numeric values directly (e.g., accuracy, MAE)

EVALUATION SECTION:
{eval_text}

AUTOGLUON METRIC MAPPING (use exact names):
- AUC/ROC/Area Under Curve → "roc_auc"
- RMSE/Root Mean Squared Error → "root_mean_squared_error"
- MAE/Mean Absolute Error → "mean_absolute_error"
- Accuracy → "accuracy"
- Log Loss/Logarithmic Loss → "log_loss"
- F1 Score → "f1"
- Precision → "precision"
- Recall → "recall"

PROBLEM TYPE RULES:
- If predicting 0/1, True/False, or probability → "binary"
- If predicting continuous number → "regression"
- If predicting one of 3+ categories → "multiclass"

Return ONLY valid JSON (no markdown, no explanation):
{{"problem_type": "binary|regression|multiclass", "metric": "autogluon_metric_name", "submit_probabilities": true|false}}"""

    from rich.progress import Progress, SpinnerColumn, TextColumn

    start_ai = time.perf_counter()
    ai_result: Dict[str, Any] = {}
    model = "gemini-2.5-flash"
    status = "failed"
    error_text = ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Asking AI to detect problem type and metric...", total=None)

        try:
            ai_result, model = call_ai_json(prompt, primary="gemini", retries=2)
            status = "success"
        except Exception as e:
            error_text = str(e)
            console.print(f"[yellow]AI detection failed: {e}[/yellow]")
        finally:
            progress.remove_task(task)
            ai_duration = time.perf_counter() - start_ai

    # Build AI log for state.json
    ai_log = {
        "command": f'echo "<prompt>" | gemini --model {model}',
        "model": model,
        "prompt": prompt,
        "response": ai_result if ai_result else {},
        "metadata": {
            "competition": competition_slug,
            "eval_text_length": len(eval_text),
            "duration_seconds": round(ai_duration, 3),
            "status": status,
            "error": error_text,
        },
    }

    # Note: No longer logging to logs/ - all AI interactions are in state.json
    console.print(f"[dim]AI interaction logged to state.json (status: {status})[/dim]")

    if status == "success" and "problem_type" in ai_result and "metric" in ai_result:
        detected_type = ai_result["problem_type"]
        detected_metric = ai_result["metric"]

        if detected_type in ["binary", "regression", "multiclass"]:
            submit_proba = ai_result.get("submit_probabilities")
            console.print(f"[green]✓ AI detected ({model}): {detected_type} / {detected_metric}[/green]")
            return detected_type, detected_metric, submit_proba, ai_log
        else:
            console.print(f"[yellow]AI returned invalid problem_type: {detected_type}[/yellow]")

    return None, None, None, ai_log
