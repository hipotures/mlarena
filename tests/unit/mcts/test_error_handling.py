
import pytest
from pathlib import Path
from mlarena.modules.mcts.executor import MlaCliExecutor

def test_executor_handles_json_error_payload(tmp_path):
    executor = MlaCliExecutor(project_root=tmp_path)
    
    # Mocking a command that fails but outputs valid failure JSON
    json_err = '{"success": false, "error": "Handled Application Error", "experiment_id": "exp1"}'
    cmd = ["python3", "-c", f"import sys; print('{json_err}'); sys.exit(1)"]
    
    result = executor.run(cmd)
    
    assert result.success is False
    assert result.details.get("error") == "Handled Application Error"
    assert result.experiment_id == "exp1"

def test_executor_filters_rich_noise(tmp_path):
    executor = MlaCliExecutor(project_root=tmp_path)
    
    # Mocking output with Rich panels but no JSON
    rich_output = """
┏━━━━━━━━━━━━━━━━ ERROR ━━━━━━━━━━━━━━━━┓
┃ Critical: Missing Columns             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    cmd = ["python3", "-c", f"import sys; print('''{rich_output}'''); sys.exit(1)"]
    
    result = executor.run(cmd)
    
    assert result.success is False
    # Should skip the bottom border and pick the actual error line
    assert "Critical: Missing Columns" in result.details.get("error")
    assert "┗" not in result.details.get("error")
