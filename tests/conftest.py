"""Runs the cairnlint-py command on a throwaway tree and reads back its output."""

import re
import tempfile
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from cairnlint_py.cli import AGENT_ENV_VARS, main

_SUMMARY = re.compile(r"^\[agent\] heuristic findings written to (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Run:
    """What one command run produced, with tree paths made relative."""

    exit_code: int
    findings: list[str]
    stderr: str
    agent_file: Path | None


@dataclass(frozen=True)
class Lint:
    """Write files under a tree, run the command on it, and collect the result.

    The command runs from the tree, so `paths` are relative to it. A run whose
    exit status differs from `expect_exit` fails the test, so a fixture that
    does not parse cannot pass as a run with no findings.
    """

    root: Path
    monkeypatch: pytest.MonkeyPatch
    capsys: pytest.CaptureFixture[str]

    def __call__(
        self,
        files: Mapping[str, str],
        args: Sequence[str] = ("--agent",),
        expect_exit: int = 0,
        paths: Sequence[str] = (".",),
    ) -> Run:
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(textwrap.dedent(content), encoding="utf-8")
        self.root.mkdir(exist_ok=True)
        self.monkeypatch.chdir(self.root)
        exit_code = main([*args, *paths])
        stderr = self.capsys.readouterr().err
        if exit_code != expect_exit:
            pytest.fail(f"exit status {exit_code}, expected {expect_exit}: {stderr}")
        summary = _SUMMARY.search(stderr)
        agent_file = Path(summary.group(1)) if summary else None
        written = agent_file.read_text(encoding="utf-8") if agent_file else ""
        findings = written.replace(f"{self.root.absolute()}/", "").splitlines()
        return Run(exit_code, findings, stderr, agent_file)


@pytest.fixture
def lint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Lint:
    """A `Lint` runner on a fresh tree.

    Agent variables from the shell running pytest are cleared, so each test
    chooses agent mode itself. The temp directory points inside tmp_path so
    the agent file never lands in the real one.
    """
    for key in AGENT_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    return Lint(tmp_path / "tree", monkeypatch, capsys)
