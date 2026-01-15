import pytest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlarena.modules.mcts.executor import MlaCliExecutor, ExperimentResult

def test_build_command():
    executor = MlaCliExecutor(project_root=Path("/app"))
    cmd = executor.build_command(
        project="test_proj",
        module="model",
        model_template="model_v1",
        preprocess_template="chain_v1",
        exp_id="my_exp"
    )
    
    cmd_str = " ".join(cmd)
    assert "scripts/mla.py model" in cmd_str
    assert "project=test_proj" in cmd_str
    assert "model_template=model_v1" in cmd_str
    assert "preprocess_template=chain_v1" in cmd_str
    assert "experiment_id=my_exp" in cmd_str
    assert "json_output=true" in cmd_str

def test_parse_result_success():
    executor = MlaCliExecutor(project_root=Path("/app"))
    
    stdout = json.dumps({
        "experiment_id": "exp1",
        "local_cv": 0.85,
        "eval_metric": "auc",
        "duration_seconds": 10.5
    })
    
    result = executor.parse_result(stdout)
    assert result.value == 0.85
    assert result.metric == "auc"
    assert result.duration == 10.5
    assert result.success is True

def test_parse_result_fallback_best_value():
    executor = MlaCliExecutor(project_root=Path("/app"))
    
    stdout = json.dumps({
        "experiment_id": "exp1",
        "local_cv": None,
        "best_value": 0.9,
        "metrics": {"metric_name": "logloss"}
    })
    
    result = executor.parse_result(stdout)
    assert result.value == 0.9
    assert result.metric == "logloss"

def test_parse_result_failure():
    executor = MlaCliExecutor(project_root=Path("/app"))
    
    result = executor.parse_result("Not JSON")
    assert result.success is False
    assert result.value is None
