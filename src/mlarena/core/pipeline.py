"""
Pipeline execution logic for MLArena.

Resolves module dependencies and coordinates state updates.
"""

import traceback
from datetime import datetime as dt
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .module import BaseModule, ModuleResult


class PipelineExecutor:
    def __init__(self, modules: Dict[str, BaseModule]):
        self.modules = modules

    def _resolve_execution_order(self, target_module: str) -> List[str]:
        if target_module not in self.modules:
            raise KeyError(f"Module '{target_module}' is not registered in pipeline.")

        order: List[str] = []
        temp: set[str] = set()
        perm: set[str] = set()

        def visit(name: str) -> None:
            if name in perm:
                return
            if name in temp:
                raise ValueError(f"Cyclic dependency detected at '{name}'.")
            temp.add(name)
            module = self.modules.get(name)
            if module is None:
                raise KeyError(f"Dependency '{name}' not found.")
            for dep in module.dependencies:
                if dep not in self.modules:
                    raise KeyError(f"Dependency '{dep}' required by '{name}' is missing.")
                visit(dep)
            perm.add(name)
            temp.remove(name)
            order.append(name)

        visit(target_module)
        return order

    def run_module(self, module_name: str, force: bool = False, skip_deps: bool = False) -> Dict[str, ModuleResult]:
        results: Dict[str, ModuleResult] = {}
        execution_plan = [module_name] if skip_deps else self._resolve_execution_order(module_name)

        for name in execution_plan:
            module = self.modules[name]
            state_entry = module.context.state.modules.get(name)
            already_completed = state_entry and state_entry.status == "completed"

            # Validate module can run (before checking completed status)
            can_run, reason = module.can_run()
            if not can_run:
                module.context.state.fail_module(name, reason)
                module.context.state.save()
                results[name] = ModuleResult(success=False, error=reason)
                break

            # Show header for already completed modules
            if already_completed and not force:
                console = Console(force_terminal=True)
                completed_time_raw = state_entry.finished_at or state_entry.started_at or "unknown"

                # Format timestamp to seconds (remove microseconds and timezone)
                if completed_time_raw != "unknown":
                    try:
                        dt_obj = dt.fromisoformat(completed_time_raw.replace('Z', '+00:00'))
                        completed_time = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        completed_time = completed_time_raw
                else:
                    completed_time = completed_time_raw

                # Init module should always run to show project status (even if completed)
                if name == "init":
                    # Let init module execute to show config
                    pass
                else:
                    # Other modules: show completed status and skip
                    info_table = Table(show_header=False, box=None)
                    info_table.add_column(style="bold")
                    info_table.add_column(style="green")
                    info_table.add_row("Status:", "Already completed")
                    info_table.add_row("Completed at:", completed_time)

                    console.print(Panel(
                        info_table,
                        title=f"[bold yellow]{name.upper()}[/bold yellow]",
                        border_style="yellow"
                    ))
                    console.print(f"\n[dim]Use [cyan]--force[/cyan] to re-run this module[/dim]\n")

                    results[name] = ModuleResult(success=True, payload=state_entry.payload)
                    continue

            defer_save = name == "init" and not module.context.project_root.exists()

            module.context.state.start_module(name, getattr(module, "invocation_params", {}))
            if not defer_save:
                module.context.state.save()

            # Display module header
            console = Console(force_terminal=True)
            start_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
            header_content = f"[bold]Started:[/bold] {start_time}"
            console.print(Panel(header_content, title=f"[bold cyan]{name.upper()}[/bold cyan]", border_style="cyan"))

            try:
                outcome = module.execute()
            except Exception as exc:  # pragma: no cover - defensive
                error_msg = f"{exc}"
                detail = traceback.format_exc()
                module.context.state.fail_module(name, error_msg)
                if not defer_save or module.context.project_root.exists():
                    module.context.state.save()
                results[name] = ModuleResult(success=False, error=detail)
                break

            if outcome.success:
                module.context.state.complete_module(name, outcome.payload)
                if not defer_save or module.context.project_root.exists():
                    module.context.state.save()
                results[name] = outcome
            else:
                module.context.state.fail_module(name, outcome.error or "unknown error")
                if not defer_save or module.context.project_root.exists():
                    module.context.state.save()
                results[name] = outcome
                break

        return results
