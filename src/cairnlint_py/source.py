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
    """Parse every Python file under roots.

    Returns the parsed files and whether any root or file failed to load. A
    failed file is logged and skipped so the rest of the tree is still checked.
    """
    loaded: dict[Path, SourceFile] = {}
    failed = False
    for root in roots:
        if not root.exists():
            logger.error("path does not exist: path=%s", root)
            failed = True
            continue
        parsed = [_parse(path, is_test) for path, is_test in _discover(root)]
        failed = failed or None in parsed
        loaded.update((source.path, source) for source in parsed if source is not None)
    return list(loaded.values()), failed


def _discover(root: Path) -> Iterator[tuple[Path, bool]]:
    # A file named on the command line is checked whatever its directory, but
    # a stub file holds declarations only, so there is nothing to check in it.
    if root.is_file():
        if root.suffix != ".pyi":
            yield root, _is_test_path(root)
        return
    for directory, subdirectories, filenames in root.walk():
        subdirectories[:] = sorted(
            name for name in subdirectories if not _is_skipped(directory / name)
        )
        yield from (
            (directory / name, _is_test_path(directory / name))
            for name in sorted(filenames)
            if name.endswith(".py")
        )


def _is_skipped(directory: Path) -> bool:
    # pyvenv.cfg marks a virtual environment whatever the directory is called.
    return (
        directory.name.startswith(".")
        or directory.name == "__pycache__"
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
