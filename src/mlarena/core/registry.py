"""
Module registry for MLArena.

Handles discovery and lookup of BaseModule subclasses.
"""

import importlib
import sys
from typing import Dict, Iterable, Type

from .module import BaseModule


class ModuleRegistry:
    """Global registry that stores discovered ``BaseModule`` subclasses."""

    _modules: Dict[str, Type[BaseModule]] = {}

    @classmethod
    def register(cls, module_class: Type[BaseModule]) -> Type[BaseModule]:
        """
        Register a module class via decorator or direct call.

        Args:
            module_class: Class inheriting from ``BaseModule`` with a non-empty ``name`` attribute.

        Returns:
            The registered class (enables decorator usage).

        Raises:
            ValueError: If ``name`` is missing or already registered.
        """
        if not getattr(module_class, "name", None):
            raise ValueError("Module must define a non-empty 'name'.")
        if module_class.name in cls._modules:
            raise ValueError(f"Module '{module_class.name}' already registered.")
        cls._modules[module_class.name] = module_class
        return module_class

    @classmethod
    def get(cls, name: str) -> Type[BaseModule]:
        """
        Retrieve a registered module class by name.

        Args:
            name: Module identifier (e.g., ``\"model\"``).

        Returns:
            Registered ``BaseModule`` subclass.

        Raises:
            KeyError: If the module is unknown.
        """
        return cls._modules[name]

    @classmethod
    def available(cls) -> Iterable[str]:
        """
        List registered module names.

        Returns:
            Iterable of module identifiers.
        """
        return cls._modules.keys()

    @classmethod
    def clear(cls) -> None:
        """Remove all registered modules (primarily used in tests)."""
        cls._modules.clear()

    @classmethod
    def discover(cls, force_reload: bool = False) -> None:
        """
        Import the ``mlarena.modules`` package to trigger decorator registration.

        Args:
            force_reload: Reload modules from disk instead of using cached imports.
        """
        if force_reload:
            # Purge cached modules to force decorator side effects after a clear.
            for modname in list(sys.modules.keys()):
                if modname == "mlarena.modules" or modname.startswith(
                    "mlarena.modules."
                ):
                    sys.modules.pop(modname, None)

        # Import modules package (uses cache if force_reload=False)
        importlib.import_module("mlarena.modules")
