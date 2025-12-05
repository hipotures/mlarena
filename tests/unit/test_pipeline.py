from mlarena.core.experiment import ExperimentState
from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.pipeline import PipelineExecutor


class _DummyModule(BaseModule):
    def __init__(self, context, name: str, dependencies=None, succeed: bool = True, can_run: bool = True):
        self.name = name
        self.dependencies = set(dependencies or [])
        self._succeed = succeed
        self._can_run = can_run
        super().__init__(context)

    def can_run(self):
        if not self._can_run:
            return False, "blocked"
        return True, ""

    def execute(self) -> ModuleResult:
        return ModuleResult(success=self._succeed, payload={"ran": self.name})


def test_pipeline_executes_dependencies_in_order(context_factory, project_root):
    state = ExperimentState.load_or_create(project_root, "demo")
    dep_ctx = context_factory("dep", state=state)
    main_ctx = context_factory("main", state=state)

    dep = _DummyModule(dep_ctx, name="dep")
    main = _DummyModule(main_ctx, name="main", dependencies={"dep"})
    executor = PipelineExecutor({"dep": dep, "main": main})

    results = executor.run_module("main")
    assert list(results.keys()) == ["dep", "main"]
    assert state.modules["dep"].status == "completed"
    assert state.modules["main"].status == "completed"


def test_pipeline_skips_completed_without_force(context_factory, project_root):
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = context_factory("once", state=state)
    mod = _DummyModule(ctx, name="once")
    executor = PipelineExecutor({"once": mod})
    executor.run_module("once")

    mod._succeed = False  # would fail if re-executed
    results = executor.run_module("once")
    assert results["once"].success is True
    assert state.modules["once"].status == "completed"


def test_pipeline_respects_can_run(context_factory, project_root):
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = context_factory("blocked", state=state)
    blocked = _DummyModule(ctx, name="blocked", can_run=False)
    executor = PipelineExecutor({"blocked": blocked})

    results = executor.run_module("blocked")
    assert results["blocked"].success is False
    assert state.modules["blocked"].status == "failed"
    assert "blocked" in (state.modules["blocked"].error or "")
