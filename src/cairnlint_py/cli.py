"""The cairnlint-py command: agent-mode detection, the checks, and their output.

Output follows cairnlint's agent mode so a caller that handles one handles
both. Findings go to `cairnlint-agent-<PID>.txt` in TMPDIR, or the system temp
directory when TMPDIR is unset, one `path:line:column: message` line each,
and a single stderr line names the file. A run with no findings writes no
file, and one whose file cannot be written prints its findings to stderr and
exits 3. A path or file that cannot be loaded exits 1. Each agent-mode run
first deletes agent files in that directory last written more than a day ago.
"""

import argparse
import contextlib
import logging
import os
import re
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from cairnlint_py import loneimpl, loneparam, passthrough, stubbody
from cairnlint_py.source import Finding, SourceFile, load_sources
from cairnlint_py.suppress import filter_suppressed

__all__ = ["AGENT_ENV_VARS", "main"]

logger = logging.getLogger(__name__)

# Variables AI coding tools set in the shells they run, the same list cairnlint
# reads. CAIRNLINT_AGENT is the opt-in for tools that set none of them.
AGENT_ENV_VARS = (
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

# Keyed by analyzer name, the name a `# nolint:` directive takes.
_CHECKS: dict[str, Callable[[Sequence[SourceFile]], list[Finding]]] = {
    stubbody.ANALYZER: stubbody.check,
    loneimpl.ANALYZER: loneimpl.check,
    passthrough.ANALYZER: passthrough.check,
    loneparam.ANALYZER: loneparam.check,
}

# Exit statuses follow the go/analysis driver cairnlint runs on: 1 when a file
# failed to load, 3 when findings were printed instead of written to the file.
_EXIT_LOAD_FAILED = 1
_EXIT_FINDINGS_PRINTED = 3

# The file sits in a shared temp directory, so only its owner may read it.
_AGENT_FILE_MODE = 0o600

# The names this tool writes, cairnlint-agent-<PID>.txt, and nothing else.
_AGENT_FILE_NAME = re.compile(r"cairnlint-agent-[0-9]+\.txt")
# A day leaves any caller time to read its file before a later run sweeps it.
_AGENT_FILE_MAX_AGE_SECONDS = 24 * 60 * 60


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks on the given paths and return the exit status."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname).1s: %(message)s")
    # Only existence is checked before agent mode, so a mistyped path fails a
    # human run too. Parsing waits for agent mode, so a human run passes over
    # a file that does not parse or a directory it cannot read.
    missing = [path for path in args.paths if not path.exists()]
    for path in missing:
        logger.error("path does not exist: path=%s", path)
    existing = [path for path in args.paths if path not in missing]
    if not (args.agent or any(os.environ.get(key) for key in AGENT_ENV_VARS)):
        # Every check is agent-tier, so outside agent mode there is nothing to
        # run, the way cairnlint skips its agent tier for a human.
        logger.info(
            "agent mode is off, so no checks ran: pass --agent or set CAIRNLINT_AGENT=1"
        )
        return _EXIT_LOAD_FAILED if missing else 0
    _sweep_old_agent_files()
    sources, load_failed = load_sources(existing)
    unfiltered_findings = [
        finding for check in _CHECKS.values() for finding in check(sources)
    ]
    findings = sorted(filter_suppressed(unfiltered_findings, sources))
    status = _write_findings(findings) if findings else 0
    return _EXIT_LOAD_FAILED if load_failed or missing else status


class _Arguments(argparse.Namespace):
    agent: bool = False
    paths: Sequence[Path] = ()


def _parse_args(argv: Sequence[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(
        prog="cairnlint-py",
        description=(
            "Heuristic checks for code an AI model tends to write: stub bodies, "
            "Protocols and ABCs with one implementation, forward-only methods, "
            "and parameters every caller passes the same constant. They run "
            "only in agent mode, and findings go to a temp file for an LLM to "
            "triage. Suppress one with '# nolint:<analyzer>' on its line or on "
            "a line of its own above the statement. Analyzers: "
            f"{', '.join(_CHECKS)}."
        ),
    )
    _ = parser.add_argument(
        "--agent",
        "-agent",
        action="store_true",
        help=(
            "run in agent mode; it also turns on when CAIRNLINT_AGENT or a "
            "variable an AI coding tool sets, such as CLAUDECODE, is non-empty"
        ),
    )
    _ = parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help=(
            "files to check, each a .py file or a script whose #! line runs "
            "python or uv, or directories searched recursively for .py files "
            "outside hidden, cache, virtual environment, and testdata directories"
        ),
    )
    return parser.parse_args(argv, namespace=_Arguments())


def _find_agent_dir() -> Path:
    # TMPDIR is used as given, as Go's os.TempDir does. tempfile.gettempdir()
    # moves on to another directory when TMPDIR is missing or unwritable, and
    # a caller reading $TMPDIR/cairnlint-agent-*.txt would then see no file.
    return Path(os.environ.get("TMPDIR") or tempfile.gettempdir())


def _sweep_old_agent_files() -> None:
    """Delete agent files in the agent directory last written more than a day ago.

    A caller reads its file after the run ends, so no run can delete its own;
    each one clears what earlier runs left. Anything it cannot find, list,
    read, or remove, such as another user's file in a shared /tmp or a
    sandbox with no writable temp directory, is skipped without a word,
    because that is the normal case and never this run's problem.
    """
    cutoff = time.time() - _AGENT_FILE_MAX_AGE_SECONDS
    try:
        # gettempdir() raises FileNotFoundError when no candidate is writable.
        entries = list(os.scandir(_find_agent_dir()))
    except OSError:
        return
    for entry in entries:
        if _is_old_agent_file(entry, cutoff):
            with contextlib.suppress(OSError):
                Path(entry.path).unlink()


def _is_old_agent_file(entry: os.DirEntry[str], cutoff: float) -> bool:
    if _AGENT_FILE_NAME.fullmatch(entry.name) is None:
        return False
    try:
        info = entry.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff


def _write_findings(findings: Sequence[Finding]) -> int:
    lines = [finding.render() for finding in findings]
    path = _find_agent_dir() / f"cairnlint-agent-{os.getpid()}.txt"
    try:
        _write_agent_file(path, lines)
    except OSError as err:
        # Losing the findings would be worse than mixing them into stderr.
        logger.warning(
            "agent file not written, printing findings instead: path=%s error_type=%s",
            path,
            type(err).__name__,
        )
        print("\n".join(lines), file=sys.stderr)
        return _EXIT_FINDINGS_PRINTED
    # This line is the contract a caller parses to find the file, so it is
    # command output rather than a log event and carries no level prefix.
    print(f"[agent] heuristic findings written to {path}", file=sys.stderr)
    return 0


def _write_agent_file(path: Path, lines: Sequence[str]) -> None:
    # Truncating means a file an earlier process with the same PID left behind
    # does not carry its findings into this run. That file keeps its old mode
    # through os.open, hence the fchmod, and O_NOFOLLOW refuses a symlink
    # planted at the path in the shared directory.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    with os.fdopen(
        os.open(path, flags, _AGENT_FILE_MODE), "w", encoding="utf-8"
    ) as handle:
        os.fchmod(handle.fileno(), _AGENT_FILE_MODE)
        handle.writelines(f"{line}\n" for line in lines)
