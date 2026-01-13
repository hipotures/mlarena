from __future__ import annotations
import json
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ExperimentResult:
    experiment_id: str
    value: Optional[float]
    metric: str
    duration: float
    success: bool
    details: Dict[str, Any]

class MlaCliExecutor:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def build_command(
        self, 
        project: str, 
        model_template: str, 
        preprocess_template: Optional[str], 
        exp_id: str,
        timeout: Optional[int] = None
    ) -> List[str]:
        # Using 'uv run' prefix is standard in this project
        # CLI syntax: mla.py model --project P ... or model_template=T
        # The spec says: mla.py model --model-template T --json-output --exp-id E
        
        cmd = [
            "uv", "run", "python", "scripts/mla.py", "model",
            "--project", project,
            "--exp-id", exp_id,
            "--json-output",
            "--force" # Force re-run if needed, MCTS manages cache
        ]
        
        # Add templates as args or flags. MLA supports both. 
        # Using key=value syntax for module params is robust in MLA.
        cmd.append(f"model_template={model_template}")
        if preprocess_template:
            cmd.append(f"preprocess_template={preprocess_template}")
        else:
            # Explicitly set to None/null if baseline
            # But bash can't pass None easily. 
            # If using OmegaConf key=value, we might need a way to say "no preprocess".
            # Or just omit if MLA handles it.
            # Assuming "preprocess_template=" (empty) or omitting works.
            pass
            
        return cmd

    def parse_result(self, stdout: str) -> ExperimentResult:
        try:
            data = json.loads(stdout)
            
            # Value logic
            value = data.get("local_cv")
            if value is None:
                value = data.get("best_value")
                
            # Metric logic
            metric = data.get("eval_metric")
            if not metric:
                metrics = data.get("metrics", {})
                metric = metrics.get("metric_name")
            
            return ExperimentResult(
                experiment_id=data.get("experiment_id", "unknown"),
                value=float(value) if value is not None else None,
                metric=metric or "unknown",
                duration=float(data.get("duration_seconds", 0.0)),
                success=True,
                details={"raw_json": data}
            )
        except Exception as e:
            return ExperimentResult(
                experiment_id="failed",
                value=None,
                metric="unknown",
                duration=0.0,
                success=False,
                details={"error": str(e), "stdout": stdout}
            )

    def run(self, cmd: List[str], timeout: Optional[int] = None) -> ExperimentResult:
        try:
            # Capture stdout for parsing
            proc = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False # We handle exit code manually via parsing
            )
            
            if proc.returncode != 0:
                return ExperimentResult(
                    experiment_id="failed",
                    value=None,
                    metric="unknown",
                    duration=0.0,
                    success=False,
                    details={"returncode": proc.returncode, "stderr": proc.stderr, "stdout": proc.stdout}
                )
                
            # Try to find JSON in stdout (it might be surrounded by logs)
            # The MLA --json-output usually prints ONLY JSON if silent, but let's be robust
            # We look for the last line that looks like JSON or just parse whole.
            # Simple approach: parse the whole output. If MLA is noisy, we might need to filter.
            # For now, assuming MLA respects --json-output.
            
            return self.parse_result(proc.stdout)
            
        except subprocess.TimeoutExpired:
             return ExperimentResult(
                experiment_id="timeout",
                value=None,
                metric="unknown",
                duration=timeout or 0.0,
                success=False,
                details={"error": "timeout"}
            )
