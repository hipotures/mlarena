import pytest

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry


class _TempModule(BaseModule):
    name = "temp"

    def execute(self) -> ModuleResult:  # pragma: no cover - trivial
        return ModuleResult(success=True)


def test_register_and_clear():
    ModuleRegistry.clear()
    ModuleRegistry.register(_TempModule)
    assert "temp" in set(ModuleRegistry.available())

    with pytest.raises(ValueError):
        ModuleRegistry.register(_TempModule)

    ModuleRegistry.clear()
    assert list(ModuleRegistry.available()) == []


def test_discover_registers_builtin_modules():
    ModuleRegistry.clear()
    ModuleRegistry.discover()
    available = set(ModuleRegistry.available())
    assert {"eda", "model", "predict"}.issubset(available)
