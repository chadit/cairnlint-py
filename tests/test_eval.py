"""The eval: a freshly built cairnlint-py command over recorded cases.

Each testdata/eval/<analyzer>/ holds flag.py, code that analyzer must report,
clean.py, a near miss nothing may report, and expected.txt, the exact findings
one run over both files writes. It sits under testdata/ so lint.sh does not
lint the must-flag files as source. The command comes from a wheel built here
and runs the way lint.sh's tool_cairnlint_py runs the published one: file
paths only, agent mode from the environment, TMPDIR set. lint.sh leaves the
Python to uv, so every case runs on both the requires-python floor and the
.python-version pin. A dependency, build backend, or Python update that
changes what users get fails a case.
"""

import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from cairnlint_py.cli import AGENT_ENV_VARS

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "testdata" / "eval"
CASE_FILES = ("flag.py", "clean.py")
RECORD = "expected.txt"

# argparse wraps the description, so the list is matched after the help text's
# whitespace is folded to single spaces.
_ANALYZERS = re.compile(r"Analyzers: ([\w, ]+)\.")
_FLOOR = re.compile(r">=\s*(\d+\.\d+)")


@dataclass(frozen=True)
class BuiltCommand:
    """The cairnlint-py entry point of the wheel this module built."""

    argv: tuple[str, ...]

    def run(
        self, args: Sequence[str], temp_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        """Run with agent mode on through CAIRNLINT_AGENT alone, as a caller would.

        Every other agent variable is cleared, so the shell running pytest
        cannot be what turns agent mode on.
        """
        env = {
            key: value for key, value in os.environ.items() if key not in AGENT_ENV_VARS
        }
        env |= {"CAIRNLINT_AGENT": "1", "TMPDIR": str(temp_dir)}
        # argv is uv's absolute path and arguments this module builds.
        return subprocess.run(  # noqa: S603
            [*self.argv, *args],
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


def _python_versions() -> list[str]:
    """The declared floor and the pinned version, the two ends a user can run.

    lint.sh's uvx call passes no --python, so a caller runs whatever uv picks
    within requires-python, not only the version .python-version pins.
    """
    with (REPO / "pyproject.toml").open("rb") as handle:
        match tomllib.load(handle):
            case {"project": {"requires-python": str(requires)}}:
                floor = _FLOOR.fullmatch(requires)
            case _:
                pytest.fail("pyproject.toml declares no requires-python")
    if floor is None:
        pytest.fail(f"requires-python is not a plain >=X.Y floor: {requires}")
    pinned = (REPO / ".python-version").read_text(encoding="utf-8").strip()
    return sorted({floor.group(1), pinned})


PYTHONS = _python_versions()


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, BuiltCommand]:
    """Build the working tree's wheel once and return its command per Python.

    A missing uv fails the eval rather than skipping it, so `make test` cannot
    pass without having run it.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("uv is not on PATH, and the eval builds and runs its wheel with it")
    wheel_dir = tmp_path_factory.mktemp("wheel")
    # The arguments are uv's absolute path and paths this module chose.
    build = subprocess.run(  # noqa: S603
        [uv, "build", "--wheel", "--out-dir", str(wheel_dir), str(REPO)],
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        pytest.fail(f"uv build failed: {build.stderr}")
    (wheel,) = wheel_dir.glob("*.whl")
    run_wheel = (uv, "tool", "run", "--from", str(wheel))
    return {
        python: BuiltCommand((*run_wheel, "--python", python, "cairnlint-py"))
        for python in PYTHONS
    }


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _case_gaps(analyzer: str) -> list[str]:
    case_dir = EVAL_DIR / analyzer
    missing = [
        f"{analyzer}/{name} is missing"
        for name in (*CASE_FILES, RECORD)
        if not (case_dir / name).is_file()
    ]
    if missing:
        return missing
    recorded = {line.split(":", 1)[0] for line in _read_lines(case_dir / RECORD)}
    gaps: list[str] = []
    if "flag.py" not in recorded:
        gaps.append(f"{analyzer}/{RECORD} records no flag.py finding")
    if recorded - {"flag.py"}:
        gaps.append(f"{analyzer}/{RECORD} records a finding outside flag.py")
    return gaps


@pytest.mark.parametrize("python", PYTHONS)
def test_every_listed_analyzer_has_a_flag_and_a_near_miss_case(
    built: dict[str, BuiltCommand], tmp_path: Path, python: str
) -> None:
    """The analyzers the built command lists in --help each have recorded cases."""
    run = built[python].run(["--help"], tmp_path)
    listing = _ANALYZERS.search(" ".join(run.stdout.split()))
    assert listing is not None, f"--help lists no analyzers: {run.stdout}{run.stderr}"
    analyzers = listing.group(1).split(", ")
    assert [gap for analyzer in analyzers for gap in _case_gaps(analyzer)] == []


def _run_case(built: BuiltCommand, case_dir: Path, temp_dir: Path) -> list[str]:
    """Run over the case's two files and return the findings, paths relative."""
    run = built.run([str(case_dir / name) for name in CASE_FILES], temp_dir)
    assert run.returncode == 0, run.stderr
    return [
        line.removeprefix(f"{case_dir}/")
        for written in sorted(temp_dir.glob("cairnlint-agent-*.txt"))
        for line in _read_lines(written)
    ]


def _write_suppressed_copy(analyzer: str, recorded: list[str], copy_dir: Path) -> None:
    """Copy the case with `# nolint:<analyzer>` on every line expected.txt records."""
    case_dir = EVAL_DIR / analyzer
    marked = {int(line.split(":")[1]) for line in recorded}
    lines = [
        f"{text}  # nolint:{analyzer}" if number in marked else text
        for number, text in enumerate(_read_lines(case_dir / "flag.py"), start=1)
    ]
    copy_dir.mkdir()
    _ = (copy_dir / "flag.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _ = shutil.copy(case_dir / "clean.py", copy_dir / "clean.py")


CASE_DIRS = sorted(path.name for path in EVAL_DIR.iterdir() if path.is_dir())


@pytest.mark.parametrize("python", PYTHONS)
@pytest.mark.parametrize("analyzer", CASE_DIRS)
def test_built_command_reports_exactly_the_recorded_findings(
    built: dict[str, BuiltCommand], tmp_path: Path, analyzer: str, python: str
) -> None:
    """flag.py reports what expected.txt records, and clean.py reports nothing."""
    case_dir = EVAL_DIR / analyzer
    reported = _run_case(built[python], case_dir, tmp_path)
    assert reported == _read_lines(case_dir / RECORD)


@pytest.mark.parametrize("python", PYTHONS)
@pytest.mark.parametrize("analyzer", CASE_DIRS)
def test_recorded_findings_belong_to_their_directory_analyzer(
    built: dict[str, BuiltCommand], tmp_path: Path, analyzer: str, python: str
) -> None:
    """Each recorded finding goes away under its own directory's nolint directive.

    Findings carry no analyzer name, so this is what ties a case to its
    analyzer: a flag.py holding another analyzer's shape keeps its findings.
    """
    recorded = _read_lines(EVAL_DIR / analyzer / RECORD)
    copy_dir = tmp_path / "case"
    _write_suppressed_copy(analyzer, recorded, copy_dir)
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    # A trailing comment moves no position, so a surviving finding reads the
    # same as its recorded line.
    reported = _run_case(built[python], copy_dir, temp_dir)
    assert [line for line in reported if line in recorded] == []
