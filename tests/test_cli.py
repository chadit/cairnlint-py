"""Agent mode, the agent file, the summary line, and exit statuses."""

import logging
import os
import stat
import tempfile
from pathlib import Path

import pytest

from cairnlint_py.cli import main
from tests.conftest import Lint

STUBS = {
    "b.py": "def load(path):\n    raise NotImplementedError\n",
    "a.py": "def save(path):\n    raise NotImplementedError\n",
}
# The variables cairnlint documents in its README, copied rather than read
# from cli.py so that dropping one from the code fails a case here.
DOCUMENTED_AGENT_VARS = (
    "AI_AGENT",
    "AGENT",
    "CLAUDECODE",
    "CODEX_SANDBOX",
    "CODEX_THREAD_ID",
    "GEMINI_CLI",
    "CURSOR_AGENT",
    "QWEN_CODE",
    "GOOSE_TERMINAL",
    "CLINE_ACTIVE",
    "AUGMENT_AGENT",
    "TRAE_AI_SHELL_ID",
    "OPENCODE_CLIENT",
    "CAIRNLINT_AGENT",
)
STUB = 'is a "NotImplementedError" stub: wire up a real implementation before use'


def test_agent_flag_writes_sorted_findings_to_the_agent_file(lint: Lint) -> None:
    """Findings land in cairnlint-agent-<PID>.txt and stderr names the file once."""
    run = lint(STUBS)
    assert run.exit_code == 0
    assert run.findings == [
        f"a.py:1:5: [agent] function save {STUB}",
        f"b.py:1:5: [agent] function load {STUB}",
    ]
    assert run.agent_file == Path(tempfile.gettempdir()) / (
        f"cairnlint-agent-{os.getpid()}.txt"
    )
    assert run.stderr == f"[agent] heuristic findings written to {run.agent_file}\n"


def test_agent_file_lines_carry_absolute_paths(lint: Lint, tmp_path: Path) -> None:
    """Each line starts with the absolute file path, as Go positions do."""
    run = lint(STUBS)
    assert run.agent_file is not None
    lines = run.agent_file.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith(f"{tmp_path / 'tree' / 'a.py'}:1:5: ")


def test_agent_file_is_private_to_its_owner(lint: Lint) -> None:
    """The file sits in a shared temp directory, so it is created mode 0600."""
    run = lint(STUBS)
    assert run.agent_file is not None
    assert stat.S_IMODE(run.agent_file.stat().st_mode) == 0o600


def test_clean_run_writes_no_file(lint: Lint) -> None:
    """A run with no findings writes no file and prints no summary."""
    run = lint({"clean.py": "def add(a, b):\n    return a + b\n"})
    assert run.exit_code == 0
    assert run.agent_file is None
    assert list(Path(tempfile.gettempdir()).iterdir()) == []


@pytest.mark.parametrize("key", DOCUMENTED_AGENT_VARS)
def test_agent_variable_turns_on_agent_mode(
    lint: Lint, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """Any listed variable with a non-empty value enables agent mode."""
    monkeypatch.setenv(key, "1")
    run = lint(STUBS, args=())
    assert len(run.findings) == len(STUBS)


def test_without_agent_mode_no_checks_run(
    lint: Lint, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every check is agent-tier, so a human run checks nothing and says so."""
    monkeypatch.setenv("CAIRNLINT_AGENT", "")
    with caplog.at_level(logging.INFO):
        run = lint(STUBS, args=())
    assert run.exit_code == 0
    assert run.agent_file is None
    assert list(Path(tempfile.gettempdir()).iterdir()) == []
    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert "agent mode is off" in caplog.records[0].getMessage()


def test_unparsable_file_fails_the_run_but_others_are_checked(
    lint: Lint, caplog: pytest.LogCaptureFixture
) -> None:
    """A syntax error is logged and exits 1, and the other files still report."""
    run = lint({**STUBS, "broken.py": "def oops(:\n"}, expect_exit=1)
    assert len(run.findings) == len(STUBS)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "broken.py" in errors[0].getMessage()


def test_missing_path_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A path that does not exist is logged and exits 1."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    missing = tmp_path / "nowhere"
    assert main(["--agent", str(missing)]) == 1
    assert [r.getMessage() for r in caplog.records] == [
        f"path does not exist: path={missing}"
    ]


def test_unwritable_temp_dir_prints_findings_and_exits_3(
    lint: Lint, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the file cannot be written, findings go to stderr instead."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "absent"))
    run = lint(STUBS, expect_exit=3)
    assert run.agent_file is None
    root = tmp_path / "tree"
    assert run.stderr.splitlines() == [
        f"{root / 'a.py'}:1:5: [agent] function save {STUB}",
        f"{root / 'b.py'}:1:5: [agent] function load {STUB}",
    ]


def test_skipped_directories_are_not_walked(lint: Lint) -> None:
    """Hidden directories, caches, and virtual environments are left out."""
    stub = STUBS["a.py"]
    run = lint(
        {
            ".hidden/a.py": stub,
            "__pycache__/a.py": stub,
            "env/pyvenv.cfg": "home = /usr/bin\n",
            "env/lib/a.py": stub,
        }
    )
    assert run.findings == []


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    """--help prints usage that names the flag and exits 0."""
    with pytest.raises(SystemExit) as exit_info:
        _ = main(["--help"])
    assert exit_info.value.code == 0
    assert "--agent" in capsys.readouterr().out


@pytest.mark.parametrize(
    "paths",
    [
        pytest.param((".",), id="tree-root"),
        pytest.param(("proj",), id="project-root"),
        pytest.param(("proj/src", "proj/tests"), id="source-and-test-roots"),
        pytest.param(("proj/tests",), id="test-root"),
        pytest.param(("proj/tests/fakes.py",), id="file-by-name"),
    ],
)
def test_test_code_is_classified_the_same_from_any_root(
    lint: Lint, paths: tuple[str, ...]
) -> None:
    """A stub under a project's tests directory is test code whichever path is given."""
    files = {
        "proj/pyproject.toml": "[project]\nname = 'proj'\n",
        "proj/src/app.py": "def add(a, b):\n    return a + b\n",
        "proj/tests/fakes.py": STUBS["a.py"],
    }
    run = lint(files, paths=paths)
    assert run.findings == []


def test_ancestor_named_tests_above_the_project_root_is_not_test_code(
    lint: Lint,
) -> None:
    """Only directories below the project root mark test code."""
    files = {
        "tests/proj/pyproject.toml": "[project]\nname = 'proj'\n",
        "tests/proj/app.py": STUBS["a.py"],
    }
    run = lint(files, paths=("tests/proj",))
    assert run.findings == [f"tests/proj/app.py:1:5: [agent] function save {STUB}"]


def test_stale_agent_file_is_overwritten(lint: Lint) -> None:
    """A file an earlier process with the same PID left behind is replaced."""
    stale = Path(tempfile.gettempdir()) / f"cairnlint-agent-{os.getpid()}.txt"
    _ = stale.write_text("old.py:1:1: [agent] stale finding\n", encoding="utf-8")
    run = lint(STUBS)
    assert run.findings == [
        f"a.py:1:5: [agent] function save {STUB}",
        f"b.py:1:5: [agent] function load {STUB}",
    ]


def test_findings_from_different_checks_are_sorted_by_position(lint: Lint) -> None:
    """Output follows file position, not the order the checks run in."""
    source = '''
        from typing import Protocol

        class Storer(Protocol):
            def load(self, key: str) -> bytes: ...

        class FileStore:
            def load(self, key: str) -> bytes:
                return key.encode()

        def flush():
            """Write buffered data to disk."""
    '''
    run = lint({"mod.py": source})
    assert [line.split(": ", 1)[0] for line in run.findings] == [
        "mod.py:4:7",
        "mod.py:11:5",
    ]


def test_file_passed_by_name_is_checked(lint: Lint) -> None:
    """A single .py path is checked on its own, without the rest of the tree."""
    run = lint(STUBS, paths=("a.py",))
    assert run.findings == [f"a.py:1:5: [agent] function save {STUB}"]


def test_stub_file_passed_by_name_is_skipped(lint: Lint) -> None:
    """A .pyi path holds declarations only, so nothing in it is reported."""
    stub = '''
        def flush():
            """Write buffered data to disk."""
    '''
    run = lint({"api.pyi": stub}, paths=("api.pyi",))
    assert run.findings == []


def test_leftover_agent_file_is_made_private(lint: Lint) -> None:
    """A file left behind with a wider mode ends the run at 0600."""
    leftover = Path(tempfile.gettempdir()) / f"cairnlint-agent-{os.getpid()}.txt"
    _ = leftover.write_text("old.py:1:1: [agent] stale finding\n", encoding="utf-8")
    leftover.chmod(0o644)
    run = lint(STUBS)
    assert run.agent_file == leftover
    assert stat.S_IMODE(leftover.stat().st_mode) == 0o600


def test_symlink_at_the_agent_path_is_not_followed(lint: Lint, tmp_path: Path) -> None:
    """A symlink planted at the path is refused, and findings go to stderr."""
    target = tmp_path / "victim.txt"
    _ = target.write_text("untouched\n", encoding="utf-8")
    agent_path = Path(tempfile.gettempdir()) / f"cairnlint-agent-{os.getpid()}.txt"
    agent_path.symlink_to(target)
    run = lint(STUBS, expect_exit=3)
    assert target.read_text(encoding="utf-8") == "untouched\n"
    assert len(run.stderr.splitlines()) == len(STUBS)


def test_file_named_two_ways_is_checked_once(lint: Lint) -> None:
    """A `..` spelling of a path already given does not repeat its findings."""
    run = lint(STUBS, paths=("a.py", "../tree/a.py"))
    assert run.findings == [f"a.py:1:5: [agent] function save {STUB}"]


def test_single_dash_agent_flag_turns_on_agent_mode(lint: Lint) -> None:
    """`-agent` works as it does for the Go tool's flag package."""
    run = lint(STUBS, args=("-agent",))
    assert len(run.findings) == len(STUBS)


def test_file_with_one_very_deep_expression_is_checked(lint: Lint) -> None:
    """3000 chained terms nest the tree past Python's recursion limit."""
    deep = "X = " + " + ".join(["f(1)"] * 3000) + "\n"
    run = lint({"a.py": STUBS["a.py"], "deep.py": deep})
    assert run.findings == [f"a.py:1:5: [agent] function save {STUB}"]
