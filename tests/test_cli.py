"""Agent mode, the agent file, the summary line, and exit statuses."""

import os
import stat
import time
from collections.abc import Callable
from pathlib import Path

import pytest

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
STUB = 'is a stub ("NotImplementedError"): wire up a real implementation before use'
SAVE_FINDING = f"a.py:1:5: [agent] function save {STUB}"
LOAD_FINDING = f"b.py:1:5: [agent] function load {STUB}"
STALE = "old.py:1:1: [agent] stale finding\n"


def _write_stale_file(path: Path) -> None:
    _ = path.write_text(STALE, encoding="utf-8")


def _write_world_readable_stale_file(path: Path) -> None:
    _write_stale_file(path)
    path.chmod(0o644)


def test_agent_flag_writes_sorted_findings_to_the_agent_file(lint: Lint) -> None:
    """Findings land in cairnlint-agent-<PID>.txt and stderr names the file once."""
    run = lint(STUBS)
    assert run.exit_code == 0
    assert run.findings == [SAVE_FINDING, LOAD_FINDING]
    assert run.agent_file == lint.temp_dir / f"cairnlint-agent-{run.pid}.txt"
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
    """A run with no findings writes no file and prints nothing."""
    run = lint({"clean.py": "def add(a, b):\n    return a + b\n"})
    assert run.exit_code == 0
    assert run.agent_file is None
    assert run.stderr == ""
    assert list(lint.temp_dir.iterdir()) == []


def test_empty_directory_is_a_clean_run(lint: Lint) -> None:
    """A directory with nothing to check exits 0 and writes no file."""
    run = lint({})
    assert run.stderr == ""
    assert list(lint.temp_dir.iterdir()) == []


@pytest.mark.parametrize("key", DOCUMENTED_AGENT_VARS)
def test_agent_variable_turns_on_agent_mode(
    lint: Lint, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """Any listed variable with a non-empty value enables agent mode."""
    monkeypatch.setenv(key, "1")
    run = lint(STUBS, args=())
    assert run.findings == [SAVE_FINDING, LOAD_FINDING]


def test_without_agent_mode_no_checks_run(
    lint: Lint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every check is agent-tier, so a human run checks nothing and says so."""
    monkeypatch.setenv("CAIRNLINT_AGENT", "")
    run = lint(STUBS, args=())
    assert run.exit_code == 0
    assert run.stderr == (
        "I: agent mode is off, so no checks ran: pass --agent or set "
        + "CAIRNLINT_AGENT=1\n"
    )
    assert list(lint.temp_dir.iterdir()) == []


def test_unparsable_file_fails_the_run_but_others_are_checked(lint: Lint) -> None:
    """A syntax error is logged and exits 1, and the other files still report."""
    run = lint({**STUBS, "broken.py": "def oops(:\n"}, expect_exit=1)
    assert run.findings == [SAVE_FINDING, LOAD_FINDING]
    errors = [line for line in run.stderr.splitlines() if line.startswith("E: ")]
    assert errors == [
        "E: skipping file that could not be parsed: path=broken.py "
        + "error_type=SyntaxError error=invalid syntax (broken.py, line 1)"
    ]


def _link_to_a_missing_file(path: Path) -> None:
    path.symlink_to(path.with_name("gone.py"))


def _write_latin_1_source(path: Path) -> None:
    _ = path.write_bytes(b'X = "\xff"\n')


# A dangling link rather than a mode 000 file, because root reads the latter.
@pytest.mark.parametrize(
    ("make_file", "error_type"),
    [
        pytest.param(_link_to_a_missing_file, "FileNotFoundError", id="unreadable"),
        pytest.param(_write_latin_1_source, "SyntaxError", id="non-utf-8"),
    ],
)
def test_file_that_cannot_load_fails_the_run(
    lint: Lint, make_file: Callable[[Path], None], error_type: str
) -> None:
    """A file that cannot be read or decoded exits 1; the rest still report."""
    lint.root.mkdir()
    make_file(lint.root / "bad.py")
    run = lint(STUBS, expect_exit=1)
    assert run.findings == [SAVE_FINDING, LOAD_FINDING]
    errors = [line for line in run.stderr.splitlines() if line.startswith("E: ")]
    assert len(errors) == 1
    assert errors[0].startswith(
        "E: skipping file that could not be parsed: path=bad.py "
        + f"error_type={error_type} "
    )


def test_unreadable_directory_fails_the_run_but_others_are_checked(
    lint: Lint,
) -> None:
    """A directory the walk cannot list is logged and exits 1, as a file is."""
    lint.root.mkdir()
    unreadable = lint.root / "sub"
    unreadable.mkdir()
    unreadable.chmod(0o000)
    try:
        run = lint(STUBS, expect_exit=1)
    finally:
        # Give the mode back even when the run fails, so the temp tree can be
        # removed without a chmod first.
        unreadable.chmod(0o700)
    assert run.findings == [SAVE_FINDING, LOAD_FINDING]
    errors = [line for line in run.stderr.splitlines() if line.startswith("E: ")]
    assert errors == [
        "E: skipping directory that could not be read: path=sub "
        + "error_type=PermissionError"
    ]


@pytest.mark.parametrize(
    "args",
    [pytest.param(("--agent",), id="agent-mode"), pytest.param((), id="human-run")],
)
def test_missing_path_fails_the_run(lint: Lint, args: tuple[str, ...]) -> None:
    """A path that does not exist is logged and exits 1, in agent mode or not."""
    run = lint(STUBS, args=args, paths=("a.py", "nowhere"), expect_exit=1)
    assert run.stderr.splitlines()[0] == "E: path does not exist: path=nowhere"


def test_missing_path_does_not_stop_the_other_paths(lint: Lint) -> None:
    """The paths that exist are still checked and reported."""
    run = lint(STUBS, paths=("a.py", "nowhere"), expect_exit=1)
    assert run.findings == [SAVE_FINDING]


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        pytest.param("README.md", "Run it (see below\n", [], id="markdown"),
        pytest.param("notes.txt", STUBS["a.py"], [], id="text-file"),
        pytest.param(
            "run", "#!/usr/bin/env python3\n" + STUBS["a.py"], [2], id="python-script"
        ),
        pytest.param(
            "run",
            "#!/usr/bin/env -S uv run --script\n" + STUBS["a.py"],
            [2],
            id="uv-script",
        ),
        pytest.param("run", "#!/bin/sh\necho (\n", [], id="shell-script"),
    ],
)
def test_named_file_is_checked_only_when_it_is_python(
    lint: Lint, name: str, content: str, expected: list[int]
) -> None:
    """A .py file or a script with a python #! line is checked; others are not."""
    run = lint({name: content}, paths=(name,))
    assert run.findings == [
        f"{name}:{line}:5: [agent] function save {STUB}" for line in expected
    ]


def test_named_path_that_is_neither_file_nor_directory_is_passed_over(
    lint: Lint,
) -> None:
    """A device such as /dev/null is not opened, and the run stays clean."""
    run = lint(STUBS, paths=("a.py", "/dev/null"))
    assert run.findings == [SAVE_FINDING]
    assert run.stderr == f"[agent] heuristic findings written to {run.agent_file}\n"


HOUR = 60 * 60
CLEAN = {"clean.py": "def add(a, b):\n    return a + b\n"}


def _set_age(path: Path, seconds: int) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then), follow_symlinks=False)


def _write_agent_file_aged(path: Path, seconds: int) -> None:
    _ = path.write_text(STALE, encoding="utf-8")
    _set_age(path, seconds)


@pytest.mark.parametrize(
    ("files", "expected", "summary_lines"),
    [
        pytest.param(STUBS, [SAVE_FINDING, LOAD_FINDING], 1, id="run-with-findings"),
        pytest.param(CLEAN, [], 0, id="clean-run"),
    ],
)
def test_agent_run_removes_agent_files_older_than_a_day(
    lint: Lint, files: dict[str, str], expected: list[str], summary_lines: int
) -> None:
    """Another run's agent file left for more than a day is deleted silently."""
    old = lint.temp_dir / "cairnlint-agent-1.txt"
    _write_agent_file_aged(old, 25 * HOUR)
    run = lint(files)
    assert not old.exists()
    assert run.findings == expected
    summary = f"[agent] heuristic findings written to {run.agent_file}\n"
    assert run.stderr == summary * summary_lines


def test_agent_file_less_than_a_day_old_is_kept(lint: Lint) -> None:
    """A recent agent file may still be waiting for its caller to read it."""
    recent = lint.temp_dir / "cairnlint-agent-1.txt"
    _write_agent_file_aged(recent, 23 * HOUR)
    _ = lint(CLEAN)
    assert recent.read_text(encoding="utf-8") == STALE


def test_sweep_leaves_other_old_entries_alone(lint: Lint, tmp_path: Path) -> None:
    """Only old regular files named cairnlint-agent-<digits>.txt are deleted.

    The old matching file in the same directory shows the sweep ran.
    """
    temp_dir = lint.temp_dir
    _write_agent_file_aged(temp_dir / "cairnlint-agent-1.txt", 25 * HOUR)
    _write_agent_file_aged(temp_dir / "cairnlint-agent-notes.txt", 25 * HOUR)
    _write_agent_file_aged(temp_dir / "cairnlint-agent-.txt", 25 * HOUR)
    _write_agent_file_aged(temp_dir / "cairnlint-agent-2.txt.bak", 25 * HOUR)
    (temp_dir / "cairnlint-agent-3.txt").mkdir()
    _set_age(temp_dir / "cairnlint-agent-3.txt", 25 * HOUR)
    target = tmp_path / "target.txt"
    _write_agent_file_aged(target, 25 * HOUR)
    (temp_dir / "cairnlint-agent-4.txt").symlink_to(target)
    _set_age(temp_dir / "cairnlint-agent-4.txt", 25 * HOUR)
    _ = lint(CLEAN)
    assert sorted(entry.name for entry in temp_dir.iterdir()) == [
        "cairnlint-agent-.txt",
        "cairnlint-agent-2.txt.bak",
        "cairnlint-agent-3.txt",
        "cairnlint-agent-4.txt",
        "cairnlint-agent-notes.txt",
    ]
    assert target.read_text(encoding="utf-8") == STALE


def test_run_outside_agent_mode_keeps_old_agent_files(lint: Lint) -> None:
    """A human run does nothing agent-tier, the sweep included."""
    old = lint.temp_dir / "cairnlint-agent-1.txt"
    _write_agent_file_aged(old, 25 * HOUR)
    _ = lint(STUBS, args=())
    assert old.read_text(encoding="utf-8") == STALE


def test_missing_temp_dir_prints_findings_and_exits_3(
    lint: Lint, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TMPDIR names the directory as given; when it is missing, stderr gets them."""
    absent = tmp_path / "absent"
    monkeypatch.setenv("TMPDIR", str(absent))
    run = lint(STUBS, expect_exit=3)
    root = tmp_path / "tree"
    assert run.stderr.splitlines() == [
        "W: agent file not written, printing findings instead: "
        + f"path={absent}/cairnlint-agent-{run.pid}.txt "
        + "error_type=FileNotFoundError",
        f"{root / 'a.py'}:1:5: [agent] function save {STUB}",
        f"{root / 'b.py'}:1:5: [agent] function load {STUB}",
    ]


@pytest.mark.parametrize(
    ("args", "paths", "error"),
    [
        pytest.param(
            ("--agent", "--bogus"),
            (".",),
            "unrecognized arguments: --bogus",
            id="unknown-flag",
        ),
        pytest.param(
            ("--agent",),
            (),
            "the following arguments are required: paths",
            id="no-paths",
        ),
    ],
)
def test_bad_arguments_exit_2_with_usage(
    lint: Lint, args: tuple[str, ...], paths: tuple[str, ...], error: str
) -> None:
    """Arguments the parser rejects print usage and the reason, and exit 2."""
    run = lint(STUBS, args=args, paths=paths, expect_exit=2)
    lines = run.stderr.splitlines()
    assert lines[0].startswith("usage: cairnlint-py ")
    assert lines[-1] == f"cairnlint-py: error: {error}"
    assert list(lint.temp_dir.iterdir()) == []


def test_skipped_directories_are_not_walked(lint: Lint) -> None:
    """Hidden, cache, virtual environment, and testdata directories are left out."""
    stub = STUBS["a.py"]
    run = lint(
        {
            ".hidden/a.py": stub,
            "__pycache__/a.py": stub,
            "testdata/a.py": stub,
            "env/pyvenv.cfg": "home = /usr/bin\n",
            "env/lib/a.py": stub,
            "kept/a.py": stub,
        }
    )
    assert run.findings == [f"kept/a.py:1:5: [agent] function save {STUB}"]


def test_walk_checks_only_py_files(lint: Lint) -> None:
    """A walked directory's .pyi stubs and other files are not read as code."""
    declaration = 'def flush():\n    """Write buffered data to disk."""\n'
    run = lint(
        {
            "a.py": STUBS["a.py"],
            "api.pyi": declaration,
            "notes.txt": "def oops(:\n",
            "script": STUBS["b.py"],
        }
    )
    assert run.findings == [SAVE_FINDING]


def test_help_exits_cleanly(lint: Lint) -> None:
    """--help prints usage that names the flag and every analyzer, and exits 0."""
    run = lint({}, args=("--help",), paths=())
    assert run.stdout.startswith("usage: cairnlint-py ")
    assert "--agent" in run.stdout
    analyzers = ("agentstubbody", "agentloneimpl", "agentpassthrough", "agentloneparam")
    assert [name for name in analyzers if name not in run.stdout] == []


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
    """A stub under a project's tests directory is test code whichever path is given.

    control.py sits outside the project and is checked on every run, so a
    run that checked nothing cannot pass.
    """
    files = {
        "proj/pyproject.toml": "[project]\nname = 'proj'\n",
        "proj/src/app.py": "def add(a, b):\n    return a + b\n",
        "proj/tests/fakes.py": STUBS["a.py"],
        "control.py": STUBS["b.py"],
    }
    run = lint(files, paths=(*paths, "control.py"))
    assert run.findings == [f"control.py:1:5: [agent] function load {STUB}"]


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
    run = lint(STUBS, plant=_write_stale_file)
    assert run.findings == [SAVE_FINDING, LOAD_FINDING]


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
    assert run.findings == [SAVE_FINDING]


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
    run = lint(STUBS, plant=_write_world_readable_stale_file)
    leftover = lint.temp_dir / f"cairnlint-agent-{run.pid}.txt"
    assert run.agent_file == leftover
    assert stat.S_IMODE(leftover.stat().st_mode) == 0o600


def test_symlink_at_the_agent_path_is_not_followed(lint: Lint, tmp_path: Path) -> None:
    """A symlink planted at the path is refused, and findings go to stderr, exit 3."""
    target = tmp_path / "victim.txt"
    _ = target.write_text("untouched\n", encoding="utf-8")
    run = lint(STUBS, expect_exit=3, plant=lambda path: path.symlink_to(target))
    assert target.read_text(encoding="utf-8") == "untouched\n"
    root = tmp_path / "tree"
    assert run.stderr.splitlines() == [
        "W: agent file not written, printing findings instead: "
        + f"path={lint.temp_dir}/cairnlint-agent-{run.pid}.txt error_type=OSError",
        f"{root / 'a.py'}:1:5: [agent] function save {STUB}",
        f"{root / 'b.py'}:1:5: [agent] function load {STUB}",
    ]


# Long enough that a run which refuses the FIFO has finished well before it.
FIFO_HANG_LIMIT_SECONDS = 5


def _blocked_marker(fifo: Path) -> Path:
    return fifo.with_name("open-was-blocked")


def _plant_fifo_with_late_reader(path: Path) -> None:
    """Make a FIFO at path, and fork a reader that opens it after the limit.

    A writer that blocks opening the FIFO would hang the suite. The late
    reader frees it instead, and first leaves a marker file when the command
    is still running at the limit, which is how the test sees a blocking open
    even though the regular-file check then refuses the FIFO. It runs in the
    command's own process before the command starts, so pytest itself starts
    no thread or process.
    """
    os.mkfifo(path)
    command = os.getpid()
    if os.fork() != 0:
        return
    try:
        # Holding the command's output pipes, or subprocess's exec-status pipe,
        # would keep the test waiting on this reader even when the run ends.
        os.closerange(0, os.sysconf("SC_OPEN_MAX"))
        time.sleep(FIFO_HANG_LIMIT_SECONDS)
        # An exited command leaves this process to a new parent.
        if os.getppid() == command:
            _blocked_marker(path).touch()
        reader = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        os.set_blocking(reader, True)
        while os.read(reader, 4096):
            pass
    finally:
        os._exit(0)


def _plant_fifo_with_reader_attached(path: Path) -> None:
    """Make a FIFO at path and hold it open for reading in the command.

    The reader goes on stdin because subprocess closes every other inherited
    descriptor after this hook runs, and the command never reads stdin.
    """
    os.mkfifo(path)
    _ = os.dup2(os.open(path, os.O_RDONLY | os.O_NONBLOCK), 0)


def _fallback_lines(agent_path: Path, root: Path) -> list[str]:
    return [
        "W: agent file not written, printing findings instead: "
        + f"path={agent_path} error_type=OSError",
        f"{root / 'a.py'}:1:5: [agent] function save {STUB}",
        f"{root / 'b.py'}:1:5: [agent] function load {STUB}",
    ]


def test_fifo_at_the_agent_path_is_refused_without_blocking(
    lint: Lint, tmp_path: Path
) -> None:
    """A FIFO with no reader is refused at once: findings go to stderr, exit 3."""
    run = lint(STUBS, expect_exit=3, plant=_plant_fifo_with_late_reader)
    agent_path = lint.temp_dir / f"cairnlint-agent-{run.pid}.txt"
    assert not _blocked_marker(agent_path).exists()
    assert stat.S_ISFIFO(agent_path.lstat().st_mode)
    assert run.stderr.splitlines() == _fallback_lines(agent_path, tmp_path / "tree")


def test_fifo_with_a_reader_at_the_agent_path_is_not_written(
    lint: Lint, tmp_path: Path
) -> None:
    """A FIFO that opens because a reader holds it is still not a file to write."""
    run = lint(STUBS, expect_exit=3, plant=_plant_fifo_with_reader_attached)
    agent_path = lint.temp_dir / f"cairnlint-agent-{run.pid}.txt"
    assert stat.S_ISFIFO(agent_path.lstat().st_mode)
    assert run.stderr.splitlines() == _fallback_lines(agent_path, tmp_path / "tree")


def test_file_named_two_ways_is_checked_once(lint: Lint) -> None:
    """A `..` spelling of a path already given does not repeat its findings."""
    run = lint(STUBS, paths=("a.py", "../tree/a.py"))
    assert run.findings == [SAVE_FINDING]


def test_single_dash_agent_flag_turns_on_agent_mode(lint: Lint) -> None:
    """`-agent` works as it does for the Go tool's flag package."""
    run = lint(STUBS, args=("-agent",))
    assert run.findings == [SAVE_FINDING, LOAD_FINDING]


def test_file_with_one_very_deep_expression_is_checked(lint: Lint) -> None:
    """3000 chained terms nest the tree past Python's recursion limit."""
    deep = "X = " + " + ".join(["f(1)"] * 3000) + "\n"
    run = lint({"a.py": STUBS["a.py"], "deep.py": deep})
    assert run.findings == [SAVE_FINDING]
