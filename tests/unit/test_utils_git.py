from pathlib import Path

import pytest

from mlarena.utils.git import get_git_info


def test_get_git_info_happy_path(monkeypatch, tmp_path):
    calls = []

    def fake_check_output(cmd, cwd, text, stderr):
        calls.append((tuple(cmd), cwd))
        if cmd[1:] == ["rev-parse", "HEAD"]:
            return "abc123"
        if cmd[1:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "main"
        if cmd[1:] == ["status", "--short"]:
            return " M file.txt\n?? new.py"
        raise RuntimeError("unexpected")

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    info = get_git_info(tmp_path)
    assert info["commit"] == "abc123"
    assert info["branch"] == "main"
    assert info["is_dirty"] is True
    assert info["dirty_files"] == ["M file.txt", "?? new.py"]
    assert len(calls) == 3


def test_get_git_info_handles_failure(monkeypatch, tmp_path):
    def fake_check_output(*a, **k):
        raise RuntimeError("git failure")

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    info = get_git_info(tmp_path)
    assert info == {}
