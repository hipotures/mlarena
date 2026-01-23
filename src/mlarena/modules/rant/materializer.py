from __future__ import annotations
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class TemplateMaterializer:
    """Convert RANT pipeline to YAML templates."""

    def __init__(self, run_id: str, project_root: Path):
        """Initialize materializer.

        Args:
            run_id: Run identifier (e.g., study_name)
            project_root: Project root directory
        """
        self.run_id = run_id
        self.project_root = project_root
        self.templates_dir = project_root / "templates" / "preprocess"
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def materialize(
        self,
        pipeline: Dict[str, Any],
        trial_number: int,
    ) -> Dict[str, Any]:
        """Convert pipeline to YAML templates.

        Args:
            pipeline: Pipeline dict with 'full_chain', 'pipeline_signature', etc.
            trial_number: Trial number for naming

        Returns:
            Dict with 'base_name', 'chain_path', 'step_paths'
        """
        sig8 = pipeline['pipeline_signature'][:8]
        base_name = f"{self.run_id}_t{trial_number:06d}_{sig8}"

        # Get full chain (fixed + sorted core/injected)
        full_chain = pipeline['full_chain']

        # Sort by original_index to maintain super-chain order
        sorted_chain = sorted(full_chain, key=lambda s: s.get('original_index', 0))

        chain_list: List[str] = []
        step_files: List[str] = []

        for i, step in enumerate(sorted_chain):
            step_name = step.get('step', step.get('name', f'step_{i}'))
            step_safe = "".join(c if c.isalnum() else "_" for c in step_name)

            # Template name with index
            tpl_name = f"{base_name}_{i:02d}_{step_safe}"
            tpl_path = self.templates_dir / f"{tpl_name}.yaml"

            # Build template payload
            module = step.get('template', step.get('module', step.get('group')))
            config = step.get('config', step.get('fixed_config', {}))

            payload = {
                "module": module,
                "config": config
            }

            # Write template file
            tpl_path.write_text(yaml.safe_dump(payload, sort_keys=False))

            chain_list.append(tpl_name)
            step_files.append(str(tpl_path))

        # Write chain file
        chain_path = self.templates_dir / f"{base_name}.yaml"
        chain_payload = {"chain": chain_list}
        chain_path.write_text(yaml.safe_dump(chain_payload, sort_keys=False))

        logger.info(
            f"Materialized pipeline {base_name} with {len(chain_list)} steps"
        )

        return {
            "base_name": base_name,
            "chain_path": str(chain_path),
            "step_paths": step_files,
        }
