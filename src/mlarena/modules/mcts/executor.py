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
    def __init__(self, project_root: Path, log_root: Optional[Path] = None):
        self.project_root = project_root
        self.log_root = log_root or project_root

    def build_command(
        self, 
        project: str, 
        module: str,
        model_template: Optional[str], 
        preprocess_template: Optional[str], 
        exp_id: str,
        timeout: Optional[int] = None
    ) -> List[str]:
        cmd = [
            "uv", "run", "python", "scripts/mla.py", module,
            "--project", project,
            "--exp-id", exp_id,
            "--force",
            "--json-output"
        ]
        
        if model_template:
            cmd.append(f"model_template={model_template}")
        if preprocess_template:
            cmd.append(f"preprocess_template={preprocess_template}")
            
        return cmd

    def parse_result(self, stdout: str) -> ExperimentResult:
        try:
            # MLA --json-output can sometimes be preceded by logs
            # Find the first '{' and parse from there
            start_idx = stdout.find('{')
            if start_idx != -1:
                json_str = stdout[start_idx:]
                data = json.loads(json_str)
            else:
                data = json.loads(stdout)
            
            value = data.get("local_cv")
            if value is None:
                value = data.get("best_value")
                
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

    def run(self, cmd: List[str], timeout: Optional[int] = None, require_json: bool = True) -> ExperimentResult:
        try:
            # Set environment variables to disable colors
            env = dict(subprocess.os.environ)
            env["TERM"] = "dumb"
            env["NO_COLOR"] = "1"
            
            proc = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env
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
                
            if not require_json:
                return ExperimentResult(
                    experiment_id="success",
                    value=None,
                    metric="unknown",
                    duration=0.0,
                    success=True,
                    details={"stdout": proc.stdout}
                )

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
        except Exception as e:
            return ExperimentResult(
                experiment_id="failed",
                value=None,
                metric="unknown",
                duration=0.0,
                success=False,
                details={"error": str(e)}
            )