from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from mlarena.modules.mcts.node import PipelineState


class TemplateMaterializer:
    def __init__(self, run_id: str, project_root: Path):
        self.run_id = run_id
        self.project_root = project_root
        self.templates_dir = project_root / "templates" / "preprocess"
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def materialize(
        self,
        state: PipelineState,
        node_id: int,
        fixed_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convert state to YAML templates, including fixed harness steps."""
        sig8 = state.signature[:8]
        depth = state.depth
        base_name = f"{self.run_id}_n{node_id:06d}_d{depth:02d}_{sig8}"

        # Combine fixed steps and searched steps
        # We need to maintain original order from super-chain
        # fixed_steps is list of {"index": original_idx, "config": step_def}

        all_pipeline_steps = []

        # 1. Add fixed steps with their original index
        if fixed_steps:
            for fs in fixed_steps:
                cfg = fs["config"]
                all_pipeline_steps.append(
                    {
                        "original_index": fs["index"],
                        "name": cfg["name"],
                        "module": cfg.get("template") or cfg.get("module"),
                        "config": cfg.get("fixed_config", {}),
                    }
                )

        # 2. Add searched steps from state with their recorded index
        for s in state.steps:
            # Fallback for legacy records that might only have 'step_index'
            orig_idx = s.get("original_index", s.get("step_index", 0))
            all_pipeline_steps.append(
                {
                    "original_index": orig_idx,
                    "name": s["name"],
                    "module": s.get("module") or s.get("template"),
                    "config": s.get("config", {}),
                }
            )

        # 3. Sort by original index to maintain super-chain order
        all_pipeline_steps.sort(key=lambda x: x["original_index"])

        chain_list: List[str] = []
        step_files: List[str] = []

        for i, step in enumerate(all_pipeline_steps):
            step_name = step["name"]
            step_safe = "".join(c if c.isalnum() else "_" for c in step_name)

            # Use simple numeric index for all steps to avoid MLA parsing issues
            tpl_name = f"{base_name}_{i:02d}_{step_safe}"
            tpl_path = self.templates_dir / f"{tpl_name}.yaml"

            payload = {"module": step["module"], "config": step["config"]}
            tpl_path.write_text(yaml.safe_dump(payload, sort_keys=False))

            chain_list.append(tpl_name)
            step_files.append(str(tpl_path))

        chain_path = self.templates_dir / f"{base_name}.yaml"
        chain_payload = {"chain": chain_list}
        chain_path.write_text(yaml.safe_dump(chain_payload, sort_keys=False))

        return {
            "base_name": base_name,
            "chain_path": chain_path,
            "step_paths": step_files,
        }
