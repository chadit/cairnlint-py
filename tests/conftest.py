"""Runs the cairnlint-py command on a throwaway tree and reads back its output."""

import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from cairnlint_py.cli import AGENT_ENV_VARS

_SUMMARY = re.compile(r"^\[agent\] heuristic findings written to (.+)$", re.MULTILINE)

# The console script uv installs beside the interpreter running pytest, so the
# tests run the command a user runs rather than calling into the package.
_COMMAND = Path(sys.executable).with_name("cairnlint-py")


@dataclass(frozen=True)
class Run:
    """What one command run produced, with tree paths made relative."""

    exit_code: int
    pid: int
    findings: list[str]
    stdout: str
    stderr: str
    agent_file: Path | None


@dataclass(frozen=True)
class Lint:
    """Write files under a tree, run the command on it, and collect the result.

    The command runs from the tree, so `paths` are relative to it, with the
    environment the test leaves in place, `TMPDIR` included. A run whose exit
    status differs from `expect_exit` fails the test, so a fixture that does
    not parse cannot pass as a run with no findings.
    """

    root: Path
    temp_dir: Path

    def __call__(
        self,
        files: Mapping[str, str],
        args: Sequence[str] = ("--agent",),
        expect_exit: int = 0,
        paths: Sequence[str] = (".",),
        plant: Callable[[Path], None] | None = None,
    ) -> Run:
        """Run the command once and return what it produced.

        `plant` runs in the new process before the command starts, given the
        agent file path its PID names, so a test can leave a file there the
        way an earlier process or another user could.
        """
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(textwrap.dedent(content), encoding="utf-8")
        self.root.mkdir(exist_ok=True)
        if not _COMMAND.is_file():
            pytest.fail(f"{_COMMAND} is missing: run the tests through uv run")
        # The arguments are the test's own literals and fixture paths, and the
        # child-side hook is how a test reaches the PID-named path; pytest
        # starts no threads that a fork could deadlock.
        with subprocess.Popen(  # noqa: S603
            [str(_COMMAND), *args, *paths],
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=(lambda: plant(_agent_path())) if plant else None,  # noqa: PLW1509
        ) as process:
            stdout, stderr = process.communicate()
        if process.returncode != expect_exit:
            pytest.fail(
                f"exit status {process.returncode}, expected {expect_exit}: {stderr}"
            )
        summary = _SUMMARY.search(stderr)
        agent_file = Path(summary.group(1)) if summary else None
        written = agent_file.read_text(encoding="utf-8") if agent_file else ""
        findings = written.replace(f"{self.root.absolute()}/", "").splitlines()
        return Run(
            process.returncode, process.pid, findings, stdout, stderr, agent_file
        )


def _agent_path() -> Path:
    return Path(os.environ["TMPDIR"]) / f"cairnlint-agent-{os.getpid()}.txt"


@pytest.fixture
def lint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lint:
    """A `Lint` runner on a fresh tree.

    Agent variables from the shell running pytest are cleared, so each test
    chooses agent mode itself. TMPDIR points inside tmp_path so the agent file
    never lands in the real temp directory.
    """
    for key in AGENT_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp_dir))
    return Lint(tmp_path / "tree", temp_dir)
