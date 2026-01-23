from __future__ import annotations
import time
import logging
import json
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

from mlarena.modules.rant.config import RANTConfig
from mlarena.modules.rant.storage import RANTStorage, StudyDirection, TrialState, compute_signature
from mlarena.modules.rant.space import PipelineGenerator
from mlarena.modules.rant.refinement import RefinementEngine
from mlarena.modules.rant.materializer import TemplateMaterializer
from mlarena.modules.mcts.executor import MlaCliExecutor
from rich.console import Console
from rich.text import Text

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for RANT execution."""
    project: str
    project_root: Path
    config: RANTConfig
    model_template: str | None = None


class RANTRunner:
    """Main RANT orchestration runner supporting driver/worker modes."""

    def __init__(self, context: ExecutionContext):
        """Initialize RANT runner.

        Args:
            context: Execution context with project, config, etc.
        """
        self.context = context
        self.config = context.config

        # Align default storage path with project_root (like MCTS)
        if self.config.storage_url == "sqlite:///experiments/db/rant.db":
            db_dir = context.project_root / "experiments" / "db"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.config.storage_url = f"sqlite:///{db_dir / 'rant.db'}"

        # Align default storage path with project_root (like MCTS)
        if self.config.storage_url == "sqlite:///experiments/db/rant.db":
            db_dir = context.project_root / "experiments" / "db"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.config.storage_url = f"sqlite:///{db_dir / 'rant.db'}"

        # Initialize storage
        self.storage = RANTStorage(self.config.storage_url)

        # Create or get study
        direction = StudyDirection.MAXIMIZE if self.config.direction == "maximize" else StudyDirection.MINIMIZE
        study_name = self.config.study_name or f"rant_study_{int(time.time())}"
        self.study_id, created = self.storage.create_study(study_name, direction)

        self.run_id = f"rant_s{self.study_id:04d}"
        self.study_created = created

        # File logging (same location as MCTS: project_root/experiments/logs)
        self._setup_logging()

        if created:
            self.logger.info(f"Created new study: {study_name} (ID={self.study_id})")
        else:
            self.logger.info(f"Resuming study: {study_name} (ID={self.study_id})")

        # Initialize components
        # Try to find super_chain: first in project_root, then in REPO_ROOT
        super_chain_rel = Path("conf") / "preprocess" / "mla_super_chain.yaml"
        project_chain = context.project_root / super_chain_rel
        # From runner.py: src/mlarena/modules/rant/runner.py -> go up 5 levels to repo root
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        repo_chain = repo_root / super_chain_rel

        if project_chain.exists():
            super_chain_path = project_chain
        elif repo_chain.exists():
            super_chain_path = repo_chain
        else:
            # Fall back to repo_chain (will error later if not found)
            super_chain_path = repo_chain

        search_spaces_dir = Path(__file__).parent.parent.parent / "search_spaces" / "preprocess"

        self.generator = PipelineGenerator(
            super_chain_path=super_chain_path,
            search_spaces_dir=search_spaces_dir,
            seed=self.config.seed
        )

        self.refinement_engine = RefinementEngine(
            config={
                'grid_size': self.config.refinement.grid_size,
                'log_grid_size': self.config.refinement.log_grid_size,
            }
        )

        self.materializer = TemplateMaterializer(
            run_id=study_name,
            project_root=context.project_root
        )

        # Use repo root for CLI execution (scripts/mla.py lives at repo root)
        self.executor = MlaCliExecutor(
            project_root=repo_root,
            log_root=context.project_root / "logs"
        )

        self.worker_id = f"worker_{time.time()}"
        self.study_name = study_name
        self.iteration = 0
        self.target_trials = 0
        self.console = Console(force_terminal=True)
        self.model_template = context.model_template
        self._baseline_value: float | None = None
        self.best_value: float | None = None
        self.direction = StudyDirection.MAXIMIZE if self.config.direction == "maximize" else StudyDirection.MINIMIZE
        self.best_value = self.storage.select_best_value(self.study_id, self.direction)
        self._preprocess_state_cache: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Run RANT according to runtime.role configuration.

        Returns:
            Summary dict with stats
        """
        role = self.config.runtime.role

        try:
            if role in {"driver", "driver_worker"}:
                base = self.config.initial_random_trials + (self.config.top_k * self.config.children_per_parent)
                budget = base * self.config.rounds_per_run
                if self.storage.count_total_trials(self.study_id) == 0:
                    budget += 1
                status = "Starting NEW" if self.study_created else "Resuming EXISTING"
                self.console.print(f"{status} RANT Study: {self.study_name} (Budget: {budget})")
            if role == "driver_worker":
                self._run_driver_full()
            elif role == "driver":
                self._run_driver()
            elif role == "worker":
                self._run_worker(exit_on_idle=False)
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user (Ctrl+C). Exiting cleanly.")
            return self._get_summary()

        summary = self._get_summary()
        best = summary.get("best_value")
        if best is not None:
            self.console.print(f"RANT completed. Best Score: {best:.6f}")
        else:
            self.console.print("RANT completed. Best Score: NA")
        return summary

    def _run_driver(self):
        """Driver mode: Generate trials and optionally enqueue."""
        self.logger.info("Running in DRIVER mode")

        for _ in range(self.config.rounds_per_run):
            try:
                self._execute_round()
            except Exception as e:
                self.logger.error(f"Round execution failed: {e}", exc_info=True)
                break

    def _run_driver_full(self):
        """Driver+worker mode: generate and execute a full round in one run."""
        self.logger.info("Running in DRIVER+WORKER mode (full cycle)")

        for _ in range(self.config.rounds_per_run):
            try:
                self._execute_round_full()
            except Exception as e:
                self.logger.error(f"Round execution failed: {e}", exc_info=True)
                break

    def _execute_round(self):
        """Execute a single round: random + top-K + refinement."""
        # Step 1: Resolve active round or create new
        round_num, metadata = self.storage.resolve_active_round(
            study_id=self.study_id,
            n_random=self.config.initial_random_trials,
            top_k=self.config.top_k,
            children_per_parent=self.config.children_per_parent,
            min_core_len=self.config.min_core_len,
            max_core_len=self.config.max_core_len,
            refinement_config={
                'grid_size': self.config.refinement.grid_size,
                'log_grid_size': self.config.refinement.log_grid_size,
            },
            seed=self.config.seed
        )

        self.logger.info(f"Round {round_num}: N={metadata['n_random']}, K={metadata['top_k']}, P={metadata['children_per_parent']}")

        # Use frozen metadata from DB (not CLI config)
        n_random = metadata['n_random']
        top_k = metadata['top_k']
        children_per_parent = metadata['children_per_parent']

        # Step 2: Check and handle stale trials
        self._handle_stale_trials()

        # Step 2.5: Insert baseline trial once per study (if no trials yet)
        if self.storage.count_total_trials(self.study_id) == 0:
            baseline = self._build_baseline_pipeline()
            trial_id, created = self.storage.insert_trial(
                study_id=self.study_id,
                round_num=round_num,
                phase="baseline",
                pipeline=baseline,
                parent_trial_id=None,
                max_attempts=self.config.failure.max_sampling_attempts_per_insert
            )
            if created:
                self.logger.info(f"Inserted baseline trial (trial_id={trial_id})")
                self.target_trials += 1

        # Track an estimated target count for progress display
        self.target_trials += n_random + (top_k * children_per_parent)

        # Step 3: Phase A - Random generation
        current_random = self.storage.count_trials(self.study_id, round_num, phase="random")
        attempts = 0
        max_attempts = self.config.failure.max_total_sampling_attempts_per_round

        while current_random < n_random and attempts < max_attempts:
            try:
                pipeline = self.generator.generate_random_pipeline(
                    min_len=metadata['min_core_len'],
                    max_len=metadata['max_core_len'],
                    allow_heavy_steps=self.config.allow_heavy_steps,
                    allow_heavy_variants=self.config.allow_heavy_variants,
                )

                trial_id, created = self.storage.insert_trial(
                    study_id=self.study_id,
                    round_num=round_num,
                    phase="random",
                    pipeline=pipeline,
                    parent_trial_id=None,
                    max_attempts=self.config.failure.max_sampling_attempts_per_insert
                )

                if created:
                    current_random += 1
                    self.logger.debug(f"Round {round_num}: Random trial {current_random}/{n_random} created (trial_id={trial_id})")

                attempts += 1

            except Exception as e:
                self.logger.warning(f"Failed to generate random pipeline: {e}")
                attempts += 1

        if current_random < n_random:
            self.logger.warning(f"Round {round_num}: Only generated {current_random}/{n_random} random trials (max attempts reached)")

        # Step 4: Phase B - Top-K selection
        direction = StudyDirection.MAXIMIZE if self.config.direction == "maximize" else StudyDirection.MINIMIZE
        parents = self.storage.select_top_k_global(
            study_id=self.study_id,
            k=top_k,
            direction=direction,
            only_unexpanded=True,
            max_per_architecture=self.config.max_parents_per_architecture
        )

        self.logger.info(f"Round {round_num}: Selected {len(parents)} parents for refinement")

        # Step 5: Phase C - Refinement
        for parent in parents:
            try:
                # Generate children
                children = self.refinement_engine.generate_children(
                    parent=parent['pipeline'],
                    search_spaces=self.generator.search_spaces,
                    n_children=children_per_parent,
                    seed=hash(parent['trial_id'])
                )

                self.logger.debug(f"Round {round_num}: Generated {len(children)} children for parent {parent['trial_id']}")

                # Insert children
                for child in children:
                    try:
                        trial_id, created = self.storage.insert_trial(
                            study_id=self.study_id,
                            round_num=round_num,
                            phase="refine",
                            pipeline=child,
                            parent_trial_id=parent['trial_id'],
                            max_attempts=self.config.failure.max_sampling_attempts_per_insert
                        )

                        if created:
                            self.logger.debug(f"Round {round_num}: Refinement child created (trial_id={trial_id})")

                    except Exception as e:
                        self.logger.warning(f"Failed to insert refinement child: {e}")

                # Mark parent as expanded
                self.storage.mark_expanded(parent['trial_id'], round_num)

            except Exception as e:
                self.logger.warning(f"Failed to refine parent {parent['trial_id']}: {e}")

        # Step 6: If task_queue mode, exit after enqueuing
        if self.config.runtime.executor == "task_queue":
            self.logger.info(f"Round {round_num}: Enqueued all trials (executor=task_queue, driver exits)")
            return

        # Step 7: Mark round as complete
        self.storage.mark_round_complete(self.study_id, round_num)
        self.logger.info(f"Round {round_num}: Complete")

    def _execute_round_full(self):
        """Execute a full round with interleaved execution (random -> eval -> refine -> eval)."""
        # Step 1: Resolve active round or create new
        round_num, metadata = self.storage.resolve_active_round(
            study_id=self.study_id,
            n_random=self.config.initial_random_trials,
            top_k=self.config.top_k,
            children_per_parent=self.config.children_per_parent,
            min_core_len=self.config.min_core_len,
            max_core_len=self.config.max_core_len,
            refinement_config={
                'grid_size': self.config.refinement.grid_size,
                'log_grid_size': self.config.refinement.log_grid_size,
            },
            seed=self.config.seed
        )

        self.logger.info(f"Round {round_num}: N={metadata['n_random']}, K={metadata['top_k']}, P={metadata['children_per_parent']}")

        # Use frozen metadata from DB (not CLI config)
        n_random = metadata['n_random']
        top_k = metadata['top_k']
        children_per_parent = metadata['children_per_parent']

        # Step 2: Check and handle stale trials
        self._handle_stale_trials()

        # Step 2.5: Insert baseline trial once per study (if no trials yet)
        if self.storage.count_total_trials(self.study_id) == 0:
            baseline = self._build_baseline_pipeline()
            trial_id, created = self.storage.insert_trial(
                study_id=self.study_id,
                round_num=round_num,
                phase="baseline",
                pipeline=baseline,
                parent_trial_id=None,
                max_attempts=self.config.failure.max_sampling_attempts_per_insert
            )
            if created:
                self.logger.info(f"Inserted baseline trial (trial_id={trial_id})")
                self.target_trials += 1

        # Track an estimated target count for progress display
        self.target_trials += n_random + (top_k * children_per_parent)

        # Step 3: Phase A - Random generation
        current_random = self.storage.count_trials(self.study_id, round_num, phase="random")
        attempts = 0
        max_attempts = self.config.failure.max_total_sampling_attempts_per_round

        while current_random < n_random and attempts < max_attempts:
            try:
                pipeline = self.generator.generate_random_pipeline(
                    min_len=metadata['min_core_len'],
                    max_len=metadata['max_core_len'],
                    allow_heavy_steps=self.config.allow_heavy_steps,
                    allow_heavy_variants=self.config.allow_heavy_variants,
                )

                trial_id, created = self.storage.insert_trial(
                    study_id=self.study_id,
                    round_num=round_num,
                    phase="random",
                    pipeline=pipeline,
                    parent_trial_id=None,
                    max_attempts=self.config.failure.max_sampling_attempts_per_insert
                )

                if created:
                    current_random += 1
                    self.logger.debug(f"Round {round_num}: Random trial {current_random}/{n_random} created (trial_id={trial_id})")

                attempts += 1

            except Exception as e:
                self.logger.warning(f"Failed to generate random pipeline: {e}")
                attempts += 1

        if current_random < n_random:
            self.logger.warning(f"Round {round_num}: Only generated {current_random}/{n_random} random trials (max attempts reached)")

        # Step 4: Execute baseline + random trials for this round
        self._run_worker(exit_on_idle=True, round_num=round_num, phases={"baseline", "random"})

        # Step 5: Phase B - Top-K selection (now with fresh results)
        direction = StudyDirection.MAXIMIZE if self.config.direction == "maximize" else StudyDirection.MINIMIZE
        parents = self.storage.select_top_k_global(
            study_id=self.study_id,
            k=top_k,
            direction=direction,
            only_unexpanded=True,
            max_per_architecture=self.config.max_parents_per_architecture
        )

        self.logger.info(f"Round {round_num}: Selected {len(parents)} parents for refinement")

        # Step 6: Phase C - Refinement
        for parent in parents:
            try:
                children = self.refinement_engine.generate_children(
                    parent=parent['pipeline'],
                    search_spaces=self.generator.search_spaces,
                    n_children=children_per_parent,
                    seed=hash(parent['trial_id'])
                )

                self.logger.debug(f"Round {round_num}: Generated {len(children)} children for parent {parent['trial_id']}")

                for child in children:
                    try:
                        trial_id, created = self.storage.insert_trial(
                            study_id=self.study_id,
                            round_num=round_num,
                            phase="refine",
                            pipeline=child,
                            parent_trial_id=parent['trial_id'],
                            max_attempts=self.config.failure.max_sampling_attempts_per_insert
                        )

                        if created:
                            self.logger.debug(f"Round {round_num}: Refinement child created (trial_id={trial_id})")

                    except Exception as e:
                        self.logger.warning(f"Failed to insert refinement child: {e}")

                self.storage.mark_expanded(parent['trial_id'], round_num)

            except Exception as e:
                self.logger.warning(f"Failed to refine parent {parent['trial_id']}: {e}")

        # Step 7: Execute refinement trials for this round
        self._run_worker(exit_on_idle=True, round_num=round_num, phases={"refine"})

        # Step 8: Mark round as complete
        self.storage.mark_round_complete(self.study_id, round_num)
        self.logger.info(f"Round {round_num}: Complete")

    def _run_worker(
        self,
        exit_on_idle: bool = False,
        round_num: int | None = None,
        phases: set[str] | None = None,
    ):
        """Worker mode: Claim and execute trials."""
        self.logger.info(f"Running in WORKER mode (worker_id={self.worker_id})")

        idle_polls = 0
        consecutive_failures = 0

        while True:
            # Claim next trial
            trial = self.storage.claim_next_waiting_trial(
                study_id=self.study_id,
                worker_id=self.worker_id,
                round_num=round_num,
                phases=sorted(phases) if phases else None
            )

            if not trial:
                idle_polls += 1
                self.logger.debug(f"No trials available (idle_polls={idle_polls}/{self.config.runtime.max_idle_polls})")

                # If task_queue mode, worker exits immediately
                if self.config.runtime.executor == "task_queue":
                    self.logger.info("No trials available (executor=task_queue, worker exits)")
                    break
                # If driver_worker, exit when queue is empty
                if exit_on_idle:
                    self.logger.info("No trials available (driver_worker, worker exits)")
                    break

                # If max idle polls reached, exit
                if idle_polls >= self.config.runtime.max_idle_polls:
                    self.logger.info(f"Max idle polls reached ({idle_polls}), worker exits")
                    break

                # Sleep with backoff
                time.sleep(self.config.runtime.poll_interval_sec * (1.2 ** min(idle_polls, 10)))
                continue

            # Reset idle counter
            idle_polls = 0

            # Execute trial
            try:
                result = self._execute_trial(trial)

                if result['success']:
                    self.storage.store_trial_result(
                        trial_id=trial['trial_id'],
                        state=TrialState.COMPLETE,
                        value=result['value']
                    )
                    consecutive_failures = 0
                    self.logger.info(f"Trial {trial['trial_number']} COMPLETE (value={result['value']:.6f})")
                else:
                    self.storage.store_trial_result(
                        trial_id=trial['trial_id'],
                        state=TrialState.FAIL
                    )
                    consecutive_failures += 1
                    self.logger.warning(f"Trial {trial['trial_number']} FAIL (consecutive={consecutive_failures})")
                self._emit_trial_line(trial, result)

            except Exception as e:
                self.logger.error(f"Trial {trial['trial_number']} execution error: {e}", exc_info=True)
                self.storage.store_trial_result(
                    trial_id=trial['trial_id'],
                    state=TrialState.FAIL
                )
                consecutive_failures += 1
                self._emit_trial_line(trial, None, error=str(e))

            # Check max consecutive failures
            if consecutive_failures >= self.config.failure.max_consecutive_runtime_failures:
                self.logger.error(f"Max consecutive failures reached ({consecutive_failures}), worker exits")
                break

    def _execute_trial(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single trial.

        Args:
            trial: Trial dict from claim_next_waiting_trial

        Returns:
            Result dict with success, value, etc.
        """
        trial_id = trial['trial_id']
        trial_number = trial['trial_number']
        pipeline = trial['pipeline']

        # Materialize templates
        templates = self.materializer.materialize(
            pipeline=pipeline,
            trial_number=trial_number
        )

        chain_path = templates['chain_path']
        chain_name = Path(chain_path).stem

        # Build command
        cmd = self.executor.build_command(
            project=self.context.project,
            module="model",
            model_template=self.model_template,
            preprocess_template=chain_name,
            exp_id=f"rant_t{trial_number:06d}"
        )

        # Execute
        self.logger.debug(f"Executing trial {trial_number}: {' '.join(cmd)}")

        result = self.executor.run(cmd)

        if not result.success:
            err = result.details.get("error") if isinstance(result.details, dict) else None
            if err:
                self.logger.warning(f"Trial {trial_number} error: {err}")
            if self.logger.isEnabledFor(logging.DEBUG):
                stdout_tail = ""
                stderr_tail = ""
                if isinstance(result.details, dict):
                    stdout_tail = result.details.get("stdout", "") or ""
                    stderr_tail = result.details.get("stderr", "") or ""
                if stdout_tail:
                    self.logger.debug(f"Trial {trial_number} stdout (tail): {stdout_tail}")
                if stderr_tail:
                    self.logger.debug(f"Trial {trial_number} stderr (tail): {stderr_tail}")

        return {
            'success': result.success,
            'value': result.value,
            'metric': result.metric,
            'duration': result.duration,
            'details': result.details
        }

    def _handle_stale_trials(self):
        """Check and handle stale RUNNING trials."""
        stale_ids = self.storage.get_stale_trials(
            study_id=self.study_id,
            stale_sec=self.config.failure.stale_running_sec
        )

        if not stale_ids:
            return

        policy = self.config.failure.stale_running_policy

        for stale_id in stale_ids:
            if policy == "requeue":
                self.storage.requeue_trial(stale_id)
                self.logger.info(f"Requeued stale trial {stale_id}")
            else:  # "fail"
                self.storage.store_trial_result(stale_id, TrialState.FAIL)
                self.logger.info(f"Failed stale trial {stale_id}")

    def _build_baseline_pipeline(self) -> Dict[str, Any]:
        """Build a deterministic baseline pipeline using fixed harness only."""
        full_chain = self.generator.fixed_start + self.generator.fixed_end
        pipeline_sig = compute_signature(full_chain, include_params=True)
        arch_sig = compute_signature(full_chain, include_params=False)
        return {
            'core': [],
            'injected': [],
            'full_chain': full_chain,
            'pipeline_signature': pipeline_sig,
            'architecture_signature': arch_sig,
            'seed': self.config.seed
        }

    def _format_action(self, pipeline: Dict[str, Any]) -> str:
        """Create a short action string for logging."""
        core = pipeline.get('core') or []
        if not core:
            return "baseline"
        parts = []
        for step in core:
            group = step.get('group') or step.get('step') or step.get('name')
            variant = step.get('variant')
            if group and variant:
                parts.append(f"{group}/{variant}")
            elif group:
                parts.append(str(group))
        return "+".join(parts) if parts else "core"

    def _emit_trial_line(self, trial: Dict[str, Any], result: Dict[str, Any] | None, error: str | None = None):
        """Print a single-line progress update for each trial execution."""
        self.iteration += 1
        phase = trial.get("phase") or "unknown"
        if phase == "baseline":
            depth = 0
        elif phase == "random":
            depth = 1
        elif phase == "refine":
            depth = 2
        else:
            depth = "?"

        ok = bool(result and result.get("success"))
        value = result.get("value") if result else None
        is_new_best = False
        if ok and isinstance(value, (int, float)):
            if self.best_value is None:
                is_new_best = True
            elif self.direction == StudyDirection.MAXIMIZE and value > self.best_value:
                is_new_best = True
            elif self.direction == StudyDirection.MINIMIZE and value < self.best_value:
                is_new_best = True
        if is_new_best:
            self.best_value = float(value)
        status = "★" if (ok and is_new_best) else ("✓" if ok else "✗")
        value_str = f"{value:.6f}" if isinstance(value, (int, float)) else "NA"
        raw_json = None
        if result and isinstance(result.get("details"), dict):
            raw_json = result["details"].get("raw_json", {})
        dur_str = self._format_run_duration(raw_json, result.get("duration") if result else None)
        n_features = self._get_feature_count(raw_json) if raw_json else None
        feat_str = f" F={n_features}" if n_features is not None else ""
        parent = trial.get("parent_trial_id")
        if parent is not None:
            parent_num = self.storage.get_trial_number(parent)
            parent_str = str(parent_num) if parent_num is not None else "?"
        else:
            parent_str = "~"
        action = self._format_action(trial.get("pipeline", {}))
        err_suffix = f" ERR={error}" if error else ""

        ts_total = self.target_trials or "?"
        txt = Text(f"I={self.iteration} (Ts={self.iteration}/{ts_total}) ", style="dim")
        if ok:
            status_style = "bold yellow" if is_new_best else "bold green"
        else:
            status_style = "bold red"
        txt.append(status, style=status_style)
        txt.append(
            f" T={trial.get('trial_number')} P={parent_str} D={depth}{feat_str} ",
            style="dim" if ok else "bold red",
        )
        txt.append(f"{dur_str} ", style="dim")
        txt.append(f"A={action} ", style="cyan")
        model_alias = ""
        if raw_json:
            best_model = raw_json.get("best_model") or raw_json.get("best_model_implementation")
            model_alias = self._format_model_alias(best_model)
        label = f"{model_alias}=" if model_alias else "V="
        val_style = "bold yellow" if (ok and is_new_best) else ("bold white" if ok else "bold red")
        txt.append(f"{label}{value_str}", style=val_style)

        if isinstance(value, (int, float)):
            delta_val = None
            if parent is not None:
                parent_val = self.storage.get_trial_value(parent)
                if parent_val is not None:
                    delta_val = value - parent_val
            if delta_val is None and phase != "baseline":
                baseline_val = self._baseline_value
                if baseline_val is None:
                    baseline_val = self.storage.get_baseline_value(self.study_id)
                    if baseline_val is not None:
                        self._baseline_value = baseline_val
                if baseline_val is not None:
                    delta_val = value - baseline_val

            if delta_val is not None:
                sign = "+" if delta_val >= 0 else ""
                txt.append(f" ({sign}{delta_val:.5f})", style=self._delta_style(delta_val))
        if err_suffix:
            txt.append(err_suffix, style="red")

        self.console.print(txt)

    def _format_model_alias(self, model_name: str | None) -> str:
        if not model_name:
            return ""
        name = model_name.lower()
        if "xgboost" in name or name.startswith("xgb"):
            return "XGB"
        if "lightgbm" in name or "lgbm" in name:
            return "GBM"
        if "catboost" in name:
            return "CAT"
        return model_name

    def _load_preprocess_state(self, preprocess_template: str) -> Dict[str, Any] | None:
        cached = self._preprocess_state_cache.get(preprocess_template)
        if preprocess_template in self._preprocess_state_cache:
            return cached

        from mlarena.core.config import TemplateLoader
        from mlarena.utils.hash_utils import compute_chain_hash

        loader = TemplateLoader(self.context.project_root, template_type="preprocess")
        template_cfg = loader.load(preprocess_template)
        if not template_cfg:
            self._preprocess_state_cache[preprocess_template] = None
            return None

        chain = template_cfg.get("chain") if isinstance(template_cfg, dict) else None
        if chain is not None:
            if not isinstance(chain, list) or not chain:
                self._preprocess_state_cache[preprocess_template] = None
                return None
            templates = [str(t) for t in chain]
        else:
            templates = [preprocess_template]

        template_configs = []
        for tpl in templates:
            cfg = loader.load(tpl)
            if not cfg:
                self._preprocess_state_cache[preprocess_template] = None
                return None
            template_configs.append(cfg)

        combined_hash, _ = compute_chain_hash(
            template_configs, self.context.project_root
        )
        chain_exp_id = f"pre-{preprocess_template}"
        chain_base_dir = (
            self.context.project_root / "experiments" / chain_exp_id / combined_hash
        )
        final_step_id = f"{len(templates) - 1}-{templates[-1]}"
        candidate_paths = [
            chain_base_dir / "state.json",
            chain_base_dir / final_step_id / "state.json",
        ]
        state_path = next((p for p in candidate_paths if p.exists()), None)
        if not state_path:
            self._preprocess_state_cache[preprocess_template] = None
            return None

        try:
            state = json.loads(state_path.read_text())
        except Exception:
            self._preprocess_state_cache[preprocess_template] = None
            return None

        self._preprocess_state_cache[preprocess_template] = state
        return state

    def _get_preprocess_duration(self, preprocess_template: str) -> float | None:
        state = self._load_preprocess_state(preprocess_template)
        if not state:
            return None

        module_entry = state.get("modules", {}).get("preprocess", {})
        started_at = module_entry.get("started_at")
        finished_at = module_entry.get("finished_at")

        if not started_at or not finished_at:
            progress = state.get("pipeline_progress") or {}
            started_at = started_at or progress.get("start_time")
            finished_at = finished_at or progress.get("end_time")

        if not started_at or not finished_at:
            return None

        try:
            started_dt = dt.fromisoformat(started_at.replace("Z", "+00:00"))
            finished_dt = dt.fromisoformat(finished_at.replace("Z", "+00:00"))
        except Exception:
            return None

        duration = (finished_dt - started_dt).total_seconds()
        return float(duration)

    def _get_feature_count(self, raw_json: Dict[str, Any]) -> int | None:
        """Extract number of features after preprocessing from raw_json or state.json."""
        if not raw_json:
            return None

        def _from_shapes(shapes: Dict[str, Any]) -> int | None:
            if shapes and shapes.get("train_after"):
                try:
                    return int(shapes["train_after"][1])  # (n_rows, n_features)
                except (IndexError, TypeError, ValueError):
                    return None
            return None

        shapes = raw_json.get("shapes", {})
        if not shapes:
            payload = raw_json.get("payload") or {}
            shapes = payload.get("shapes", {})

        n_feat = _from_shapes(shapes)
        if n_feat is not None:
            return n_feat

        preprocess_template = raw_json.get("preprocessing_template")
        if preprocess_template:
            return self._get_feature_count_from_preprocess(preprocess_template)
        return None

    def _get_feature_count_from_preprocess(self, preprocess_template: str) -> int | None:
        def _from_shapes(shapes: Dict[str, Any]) -> int | None:
            if shapes and shapes.get("train_after"):
                try:
                    return int(shapes["train_after"][1])  # (n_rows, n_features)
                except (IndexError, TypeError, ValueError):
                    return None
            return None

        state = self._load_preprocess_state(preprocess_template)
        if not state:
            return None

        module_shapes = (
            state.get("modules", {}).get("preprocess", {}).get("payload", {}).get("shapes", {})
        )
        n_feat = _from_shapes(module_shapes)
        if n_feat is not None:
            return n_feat

        steps = (state.get("pipeline_progress") or {}).get("steps") or []
        for step in reversed(steps):
            n_feat = _from_shapes(step.get("shapes") or {})
            if n_feat is not None:
                return n_feat
        return None

    def _format_seconds_value(self, seconds: float) -> str:
        if seconds < 10:
            return f"{seconds:.1f}"
        return f"{int(round(seconds))}"

    def _format_duration_sec(self, duration: float | None) -> str:
        if duration is None:
            return "?"
        return f"{self._format_seconds_value(float(duration))}s"

    def _format_run_duration(self, raw_json: Dict[str, Any] | None, model_duration: float | None) -> str:
        model_sec = float(model_duration or 0.0)
        preprocess_template = raw_json.get("preprocessing_template") if raw_json else None
        if preprocess_template:
            pre_sec = self._get_preprocess_duration(preprocess_template)
            if pre_sec is not None:
                pre_str = self._format_seconds_value(float(pre_sec))
                model_str = self._format_seconds_value(model_sec)
                return f"{pre_str}+{model_str}s"
        return self._format_duration_sec(model_duration)

    def _delta_style(self, delta: float) -> str:
        if delta == 0:
            return "dim"
        if self.config.direction == "maximize":
            return "green" if delta > 0 else "red"
        return "green" if delta < 0 else "red"

    def _setup_logging(self):
        log_dir = self.context.project_root / "experiments" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"rant.{self.run_id}")
        self.logger.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        self.logger.handlers = []

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(log_dir / "rant.log")
        file_handler.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"--- RANT Study Started (Run ID: {self.run_id}) ---")

    def _get_summary(self) -> Dict[str, Any]:
        """Get execution summary.

        Returns:
            Summary dict with stats
        """
        direction = StudyDirection.MAXIMIZE if self.config.direction == "maximize" else StudyDirection.MINIMIZE
        best_value = self.storage.select_best_value(self.study_id, direction)
        return {
            'study_id': self.study_id,
            'study_name': self.study_name,
            'role': self.config.runtime.role,
            'executor': self.config.runtime.executor,
            'best_value': best_value
        }
