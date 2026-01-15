import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlarena.cli.main import main  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.environ.get("MLA_E2E_REAL") != "1",
    reason="Set MLA_E2E_REAL=1 to run real AutoGluon/Kaggle-free e2e test (may be slow).",
)


def _write_small_dataset(project_root: Path):
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        {
            "id": range(20),
            "f1": [i * 0.1 for i in range(20)],
            "target": [0, 1] * 10,
        }
    )
    test = pd.DataFrame({"id": [100, 101, 102], "f1": [0.05, 0.15, 0.25]})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    code_dir = project_root / "code" / "utils"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "__init__.py").write_text("")
    (code_dir / "config.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "PROJECT_ROOT = Path(__file__).parent.parent.parent",
                "DATA_DIR = PROJECT_ROOT / 'data'",
                "TRAIN_PATH = DATA_DIR / 'train.csv'",
                "TEST_PATH = DATA_DIR / 'test.csv'",
                "TARGET_COLUMN = 'target'",
                "ID_COLUMN = 'id'",
                "AUTOGLUON_PROBLEM_TYPE = 'binary'",
                "AUTOGLUON_EVAL_METRIC = 'roc_auc'",
                "AUTOGLUON_PRESET = 'medium'",
                "AUTOGLUON_TIME_LIMIT = 15",
                "IGNORED_COLUMNS = []",
                "SUBMISSION_PROBAS = True",
                "COMPETITION_NAME = 'demo-comp'",
            ]
        )
    )


def test_real_autogluon_pipeline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    project = "demo-real"
    project_root = tmp_path / "projects" / "kaggle" / project
    _write_small_dataset(project_root)

    exit_code = main(["predict", f"project={project}"])
    assert exit_code == 0

    submissions = list((project_root / "experiments").glob("exp-*/artifacts/predict/submission*.csv"))
    assert submissions
