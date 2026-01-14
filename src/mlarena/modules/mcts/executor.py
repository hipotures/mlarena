from __future__ import annotations
import json
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
        module: str, # This should be 'model'
        model_template: Optional[str], 
        preprocess_template: Optional[str], 
        exp_id: str
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
            # Find JSON in potential noise
            start_idx = stdout.find('{')
            if start_idx != -1:
                data = json.loads(stdout[start_idx:])
            else:
                data = json.loads(stdout)
            
            value = data.get("local_cv")
            if value is None:
                value = data.get("best_value")
                
            metric = data.get("eval_metric") or (data.get("metrics") or {}).get("metric_name")
            
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
                # Truncate output to avoid massive log files but keep enough context
                err_summary = proc.stderr[-1000:] if proc.stderr else ""
                out_summary = proc.stdout[-1000:] if proc.stdout else ""
                
                # Try to parse JSON even on failure (preprocess might have emitted failure JSON)
                try:
                    res = self.parse_result(proc.stdout)
                    if res.success: # Should not happen if returncode != 0 but let's be safe
                        return res
                    error_msg = res.details.get("error", f"Exit Code: {proc.returncode}")
                except:
                    error_msg = f"Exit Code: {proc.returncode}"

                return ExperimentResult(
                    experiment_id="failed",
                    value=None,
                    metric="unknown",
                    duration=0.0,
                    success=False,
                    details={
                        "error": error_msg,
                        "returncode": proc.returncode, 
                        "stderr": err_summary, 
                        "stdout": out_summary,
                        "cmd": " ".join(cmd)
                    }
                )
                
            return self.parse_result(proc.stdout)
            
        except Exception as e:
            return ExperimentResult(
                experiment_id="failed",
                value=None,
                metric="unknown",
                duration=0.0,
                success=False,
                details={"error": str(e)}
            )