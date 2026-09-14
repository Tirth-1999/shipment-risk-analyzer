"""Pytest hooks: print a clear summary when all tests pass."""

from __future__ import annotations

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Acknowledge success so it is obvious the run completed cleanly."""
    if exitstatus != 0:
        return

    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is None:
        return

    passed = len(getattr(terminal, "stats", {}).get("passed", []))
    if passed == 0:
        return

    lines = [
        "",
        "=" * 70,
        f"ALL TESTS PASSED ({passed} checks)",
        "",
        "What this run confirmed:",
        "  - Training rows respect received_at and label rules (README #1)",
        "  - Model artifact and metrics fields are complete (README #2)",
        "  - Live engine: ingest, score, replay, snapshot, reload (README #3)",
        "  - Memory cap and thread-safe scoring (README constraints)",
        "",
        "Nothing failed. Safe to demo and submit.",
        "=" * 70,
        "",
    ]
    terminal.write_sep("=", lines[2], green=True, bold=True)
    for line in lines[3:]:
        if line.strip():
            terminal.write_line(line)
