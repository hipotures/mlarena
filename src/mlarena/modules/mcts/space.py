from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Hardcoded for now, mimicking preprocess_tune
REPO_ROOT = Path(__file__).resolve().parents[4]
SEARCH_SPACE_DIR = REPO_ROOT / "src" / "mlarena" / "search_spaces" / "preprocess"

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data

def _load_search_spaces() -> Dict[str, Dict[str, Any]]:
    spaces: Dict[str, Dict[str, Any]] = {}
    if not SEARCH_SPACE_DIR.exists():
        return spaces
    for path in SEARCH_SPACE_DIR.glob("*.yaml"):
        name = path.stem
        spaces[name] = _load_yaml(path)
    return spaces

class SuperChainActionSpace:
    def __init__(self, super_chain_path: Path):
        self.super_chain_path = super_chain_path
        self.super_chain_config = _load_yaml(super_chain_path)
        self.steps = self.super_chain_config.get("preprocessors", []) or []
        self.search_spaces = _load_search_spaces()

    @property
    def total_steps_count(self) -> int:
        return len(self.steps)
