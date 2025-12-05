"""
Pipeline execution logic for MLArena.

Resolves module dependencies and coordinates state updates.
"""

import traceback
from typing import Dict, List

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

            if already_completed and not force:
                results[name] = ModuleResult(success=True, payload=state_entry.payload)
                continue

            can_run, reason = module.can_run()
            if not can_run:
                module.context.state.fail_module(name, reason)
                module.context.state.save()
                results[name] = ModuleResult(success=False, error=reason)
                break

            defer_save = name == "init" and not module.context.project_root.exists()

            module.context.state.start_module(name, getattr(module, "invocation_params", {}))
            if not defer_save:
                module.context.state.save()

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
