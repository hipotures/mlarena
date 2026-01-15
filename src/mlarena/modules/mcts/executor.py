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
        self.current_proc: Optional[subprocess.Popen] = None

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

    def _clean_error_msg(self, text: str, default: str = "Unknown error") -> str:
        if not text:
            return default
            
        import re
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        
        # Define box-drawing characters to strip
        box_chars_re = re.compile(r'[\u2500-\u257f┃┏┓┗┛━]')
        
        lines = []
        for line in text.splitlines():
            # Strip box characters and extra whitespace
            l = box_chars_re.sub('', line).strip()
            if not l: continue
            
            # Skip common noise
            if any(x in l for x in ["Traceback", "File \"", "python3 -c", "uv run", "MLA: uv run"]): continue
            
            # Skip very short lines or purely decorative ones (e.g. "----------")
            if len(l) > 10 and len(set(l.replace(" ", ""))) <= 2: continue
            
            lines.append(l)

        if lines:
            # 1. Priority: Look for lines containing "Error" or "Exception"
            for l in reversed(lines):
                # Check for common error markers (case-insensitive)
                lower_l = l.lower()
                if "error" in lower_l or "exception" in lower_l or "fail" in lower_l or "critical" in lower_l:
                    # If it's just the word "ERROR" (like a header), keep looking for something more descriptive
                    if len(l) < 10 and lower_l == "error":
                        continue
                    return l
            
            # 2. Fallback: Take the last non-empty line
            return lines[-1]
            
        return default

    def parse_result(self, stdout: str) -> ExperimentResult:
        try:
            # Find the outermost JSON object
            start_idx = stdout.find('{')
            end_idx = stdout.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = stdout[start_idx:end_idx + 1]
                data = json.loads(json_str)
            else:
                # Fallback to direct parse if markers not found
                data = json.loads(stdout)
            
            value = data.get("local_cv")
            if value is None:
                value = data.get("best_value")
                
            metric = data.get("eval_metric") or (data.get("metrics") or {}).get("metric_name")
            success = bool(data.get("success", True))
            error = data.get("error")
            
            return ExperimentResult(
                experiment_id=data.get("experiment_id", "unknown"),
                value=float(value) if value is not None else None,
                metric=metric or "unknown",
                duration=float(data.get("duration_seconds", 0.0)),
                success=success,
                details={"raw_json": data, "error": error}
            )
        except Exception as e:
            return ExperimentResult(
                experiment_id="failed",
                value=None,
                metric="unknown",
                duration=0.0,
                success=False,
                details={
                    "error": str(e), # Keep technical error here
                    "stdout": stdout, 
                    "original_exception": str(e)
                }
            )

    def run(self, cmd: List[str], timeout: Optional[int] = None) -> ExperimentResult:
        try:
            env = dict(subprocess.os.environ)
            env["TERM"] = "dumb"
            env["NO_COLOR"] = "1"
            
            # Start in a new session to ignore signals (SIGINT) from parent's terminal
            self.current_proc = subprocess.Popen(
                cmd, 
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True
            )
            
            try:
                stdout, stderr = self.current_proc.communicate(timeout=timeout)
                returncode = self.current_proc.returncode
            except subprocess.TimeoutExpired:
                self.current_proc.kill()
                stdout, stderr = self.current_proc.communicate()
                return ExperimentResult(
                    experiment_id="failed",
                    value=None,
                    metric="unknown",
                    duration=0.0,
                    success=False,
                    details={"error": "Timeout expired", "stdout": stdout, "stderr": stderr}
                )
            finally:
                self.current_proc = None
            
            err_content = stderr or ""
            out_content = stdout or ""
            combined_output = err_content + "\n" + out_content

            if returncode != 0:
                # Try to parse JSON even on failure
                res = self.parse_result(out_content)
                
                # If we have a structured failure from JSON (and NOT just a parsing error), return it
                is_parsing_error = "Expecting value" in str(res.details.get("original_exception", ""))
                if res.success is False and not is_parsing_error and (res.experiment_id != "failed" or res.details.get("error")):
                    res.details.update({
                        "returncode": returncode,
                        "stderr": err_content[-2000:],
                        "cmd": " ".join(cmd)
                    })
                    return res
                
                # Fallback to cleaning text output
                error_msg = self._clean_error_msg(combined_output, f"Exit Code: {returncode}")

                return ExperimentResult(
                    experiment_id="failed",
                    value=None,
                    metric="unknown",
                    duration=0.0,
                    success=False,
                    details={
                        "error": error_msg,
                        "returncode": returncode, 
                        "stderr": err_content[-2000:], 
                        "stdout": out_content[-2000:],
                        "cmd": " ".join(cmd)
                    }
                )
                
            return self.parse_result(out_content)
            
        except Exception as e:
            self.current_proc = None
            return ExperimentResult(
                experiment_id="failed",
                value=None,
                metric="unknown",
                duration=0.0,
                success=False,
                details={"error": str(e)}
            )