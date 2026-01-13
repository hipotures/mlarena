from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any, List
from mlarena.modules.mcts.node import PipelineState

class TemplateMaterializer:
    def __init__(self, run_id: str, project_root: Path):
        self.run_id = run_id
        self.project_root = project_root
        self.templates_dir = project_root / "templates" / "preprocess"
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def materialize(self, state: PipelineState, node_id: int) -> Dict[str, Any]:
        """Convert state to YAML templates."""
        sig8 = state.signature[:8]
        depth = state.depth
        base_name = f"{self.run_id}_n{node_id:06d}_d{depth:02d}_{sig8}"
        
        chain_list: List[str] = []
        step_files: List[str] = []
        
        for i, step in enumerate(state.steps):
            step_name = step.get("name", f"step_{i}")
            # Ensure name is safe for filesystem
            step_safe = "".join(c if c.isalnum() else "_" for c in step_name)
            
            tpl_name = f"{base_name}__{i:02d}-{step_safe}"
            tpl_path = self.templates_dir / f"{tpl_name}.yaml"
            
            module = step.get("module") or step.get("template")
            config = step.get("config", {})
            
            payload = {
                "module": module,
                "config": config,
                # Include meta for debugging?
                # "meta": step.get("meta")
            }
            tpl_path.write_text(yaml.safe_dump(payload, sort_keys=False))
            
            chain_list.append(tpl_name)
            step_files.append(str(tpl_path))
            
        chain_path = self.templates_dir / f"{base_name}.yaml"
        chain_payload = {"chain": chain_list}
        chain_path.write_text(yaml.safe_dump(chain_payload, sort_keys=False))
        
        return {
            "base_name": base_name,
            "chain_path": chain_path,
            "step_paths": step_files
        }
