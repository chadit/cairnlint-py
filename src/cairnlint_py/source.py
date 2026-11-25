"""Loading the Python files a run checks, and the finding every check returns."""

import ast
import functools
import logging
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Finding", "SourceFile", "load_sources"]

logger = logging.getLogger(__name__)

_TEST_DIRECTORIES = frozenset({"tests", "test"})
_SKIPPED_DIRECTORIES = frozenset({"__pycache__", "testdata"})
# `uv run --script` runs a Python file, so a uv #! line marks one too.
_SCRIPT_INTERPRETERS = frozenset({"python", "uv"})
_VERSION_SUFFIX = re.compile(r"[0-9.].*")
_PROJECT_MARKERS = ("pyproject.toml", ".git")
_DEFINITION_KEYWORD = re.compile(rb"(?:async\s+)?(?:def|class)\s+")


@dataclass(frozen=True, order=True)
class Finding:
    """One diagnostic. Field order is the sort order, so output is stable."""

    path: Path
    line: int
    column: int
    analyzer: str
    message: str

    def render(self) -> str:
        """Format the finding the way cairnlint writes its agent file."""
        return f"{self.path}:{self.line}:{self.column}: {self.message}"


@dataclass(frozen=True)
class SourceFile:
    """A parsed file. `path` is absolute, as Go's positions are."""

    path: Path
    source_bytes: bytes
    tree: ast.Module
    is_test: bool

    def build_finding(
        self, node: ast.stmt | ast.arg, analyzer: str, message: str
    ) -> Finding:
        """Anchor a finding on node. Columns are 1-based byte offsets, as in Go.

        A `def` or `class` anchors on its name, where Go anchors a declaration,
        rather than on the keyword the AST node starts at.
        """
        column = node.col_offset
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            line = self.source_bytes.splitlines()[node.lineno - 1]
            keyword = _DEFINITION_KEYWORD.match(line, column)
            column = keyword.end() if keyword else column
        return Finding(self.path, node.lineno, column + 1, analyzer, message)


def load_sources(roots: Sequence[Path]) -> tuple[list[SourceFile], bool]:
    """Parse every Python file under roots, each of which must exist.

    Returns the parsed files and whether any directory or file failed to load.
    One that fails is logged and skipped so the rest of the tree is still
    checked.
    """
    loaded: dict[Path, SourceFile] = {}
    unreadable: list[OSError] = []
    failed = False
    for root in roots:
        found = _discover(root, unreadable)
        parsed = [_parse(path, is_test) for path, is_test in found]
        failed = failed or None in parsed
        loaded.update((source.path, source) for source in parsed if source is not None)
    for err in unreadable:
        logger.error(
            "skipping directory that could not be read: path=%s error_type=%s",
            _failed_path(err),
            type(err).__name__,
        )
    return list(loaded.values()), failed or bool(unreadable)


def _failed_path(err: OSError) -> str:
    # The walk sets filename to the str path it tried, but the type admits any.
    # normpath drops the "./" the walk joins on, matching how files are named.
    match err:
        case OSError(filename=str(path)):
            return os.path.normpath(path)
        case _:
            return str(err)


def _discover(root: Path, unreadable: list[OSError]) -> Iterator[tuple[Path, bool]]:
    """Yield each file to check under root, with whether it is test code.

    A directory the walk cannot list is appended to unreadable, since
    Path.walk otherwise drops it without a word.
    """
    # A file named on the command line is checked whatever its directory. A
    # path that is neither, such as a device or a FIFO, is passed over
    # unopened, since reading a FIFO would block the run.
    if root.is_file():
        if _is_python_file(root):
            yield root, _is_test_path(root)
        return
    if not root.is_dir():
        return
    for directory, subdirectories, filenames in root.walk(on_error=unreadable.append):
        subdirectories[:] = sorted(
            name for name in subdirectories if not _is_skipped(directory / name)
        )
        yield from (
            (directory / name, _is_test_path(directory / name))
            for name in sorted(filenames)
            if name.endswith(".py")
        )


def _is_python_file(path: Path) -> bool:
    """Whether a file named on the command line holds Python to check.

    These are the files lint.sh's python lane hands over: a .py file, or a
    file with no extension whose #! line runs python or uv. A .pyi stub holds
    declarations only, so there is nothing to check in it.
    """
    if path.suffix:
        return path.suffix == ".py"
    try:
        with path.open("rb") as handle:
            first_line = handle.readline()
    except OSError:
        # Unknown, so the parse step gets it and reports why it cannot load.
        return True
    return _shebang_interpreter(first_line) in _SCRIPT_INTERPRETERS


def _shebang_interpreter(first_line: bytes) -> str | None:
    """The interpreter a #! line runs, read the way lint.sh's shebang_interp does.

    env, its options, and VAR=value settings are passed over, and a version
    suffix is dropped, so `#!/usr/bin/env -S python3.12 -u` gives `python`.
    """
    if not first_line.startswith(b"#!"):
        return None
    for word in first_line[2:].decode(errors="replace").split():
        name = word.rsplit("/", 1)[-1]
        if name == "env" or name.startswith("-") or "=" in name:
            continue
        return _VERSION_SUFFIX.sub("", name)
    return None


def _is_skipped(directory: Path) -> bool:
    # pyvenv.cfg marks a virtual environment whatever the directory is called.
    # testdata holds fixtures, which the go tool and lint.sh leave out too.
    return (
        directory.name.startswith(".")
        or directory.name in _SKIPPED_DIRECTORIES
        or (directory / "pyvenv.cfg").exists()
    )


def _is_test_path(path: Path) -> bool:
    """Classify path the same way whichever root it was reached from.

    Directories count from the project root rather than the filesystem
    root, so a project that happens to sit under a directory named `test`
    is not all taken for test code.
    """
    directory = path.absolute().parent
    name = path.name
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or not _TEST_DIRECTORIES.isdisjoint(
            directory.relative_to(_find_project_root(directory)).parts
        )
    )


@functools.cache
def _find_project_root(directory: Path) -> Path:
    """The nearest ancestor holding a project marker, else the filesystem root."""
    for candidate in (directory, *directory.parents):
        if any((candidate / marker).exists() for marker in _PROJECT_MARKERS):
            return candidate
    return Path(directory.anchor)


def _parse(path: Path, is_test: bool) -> SourceFile | None:
    try:
        source_bytes = path.read_bytes()
        tree = ast.parse(source_bytes, filename=str(path))
    except (OSError, SyntaxError, ValueError) as err:
        logger.error(
            "skipping file that could not be parsed: path=%s error_type=%s error=%s",
            path,
            type(err).__name__,
            err,
        )
        return None
    # normpath folds `..` so one file named two ways is read once, and unlike
    # resolve() it leaves symlinks as the user spelled them.
    return SourceFile(
        Path(os.path.normpath(path.absolute())), source_bytes, tree, is_test
    )
