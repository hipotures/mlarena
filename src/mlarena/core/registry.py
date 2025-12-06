"""
Module registry for MLArena.

Handles discovery and lookup of BaseModule subclasses.
"""

import importlib
import sys
from typing import Dict, Iterable, Type

from .module import BaseModule


class ModuleRegistry:
    _modules: Dict[str, Type[BaseModule]] = {}

    @classmethod
    def register(cls, module_class: Type[BaseModule]) -> Type[BaseModule]:
        if not getattr(module_class, "name", None):
            raise ValueError("Module must define a non-empty 'name'.")
        if module_class.name in cls._modules:
            raise ValueError(f"Module '{module_class.name}' already registered.")
        cls._modules[module_class.name] = module_class
        return module_class

    @classmethod
    def get(cls, name: str) -> Type[BaseModule]:
        return cls._modules[name]

    @classmethod
    def available(cls) -> Iterable[str]:
        return cls._modules.keys()

    @classmethod
    def clear(cls) -> None:
        cls._modules.clear()

    @classmethod
    def discover(cls, force_reload: bool = False) -> None:
        """Discover and register modules.

        Args:
            force_reload: If True, purge cached modules (needed for tests).
                         If False (default), use cached imports for speed.
        """
        if force_reload:
            # Purge cached modules to force decorator side effects after a clear.
            for modname in list(sys.modules.keys()):
                if modname == "mlarena.modules" or modname.startswith("mlarena.modules."):
                    sys.modules.pop(modname, None)

        # Import modules package (uses cache if force_reload=False)
        importlib.import_module("mlarena.modules")
