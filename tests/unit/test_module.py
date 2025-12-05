import pytest

from mlarena.core.module import BaseModule, ModuleResult


class _DummyModule(BaseModule):
    name = "dummy"

    def execute(self) -> ModuleResult:
        return ModuleResult(success=True)


def test_module_result_defaults():
    result = ModuleResult(success=True)
    assert result.payload == {}
    assert result.artifacts == []
    assert result.error is None


def test_invocation_and_can_run(context_factory):
    ctx = context_factory("dummy")
    mod = _DummyModule(ctx)
    mod.set_invocation_params({"alpha": 1})

    ok, reason = mod.can_run()
    assert ok is True
    assert reason == ""
    assert mod.invocation_params == {"alpha": 1}

    result = mod.execute()
    assert result.success is True
