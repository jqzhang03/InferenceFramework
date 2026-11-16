"""Profiler selection must not silently reuse the incompatible PATH version."""
import pytest

from benchmarks import cuda_gemm_profile as profile


def test_newest_ncu_version_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCU_BIN", raising=False)
    monkeypatch.setattr(profile, "ncu_installations", lambda: {
        "/cuda/ncu": "Version 2025.1.1.0",
        "/new/ncu": "Version 2026.3.1.0",
        "/old/ncu": "Version 2026.1.10.0",
    })
    assert profile.select_ncu() == "/new/ncu"


def test_explicit_override_precedes_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCU_BIN", "/environment/ncu")
    monkeypatch.setattr(profile.shutil, "which", lambda value: value)
    assert profile.select_ncu("/explicit/ncu") == "/explicit/ncu"
    assert profile.select_ncu() == "/environment/ncu"


def test_invalid_override_is_not_silently_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCU_BIN", "/missing/ncu")
    monkeypatch.setattr(profile.shutil, "which", lambda value: None)
    with pytest.raises(FileNotFoundError, match="not executable"):
        profile.select_ncu()


def test_no_usable_version_reports_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCU_BIN", raising=False)
    monkeypatch.setattr(profile, "ncu_installations", lambda: {"/broken/ncu": "unknown"})
    with pytest.raises(FileNotFoundError, match="set NCU_BIN"):
        profile.select_ncu()
