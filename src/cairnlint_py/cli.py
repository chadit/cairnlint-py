"""The cairnlint-py command: agent-mode detection, the checks, and their output.

Output follows cairnlint's agent mode so a caller that handles one handles
both. Findings go to `cairnlint-agent-<PID>.txt` in the temp directory, one
`path:line:column: message` line each, and a single stderr line names the
file. A run with no findings writes no file.
"""

import argparse
import logging
import os
import sys
import tempfile
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks on the given paths and return the exit status."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname).1s: %(message)s")
    if not (args.agent or any(os.environ.get(key) for key in AGENT_ENV_VARS)):
        # Every check is agent-tier, so outside agent mode there is nothing to
        # run, the way cairnlint skips its agent tier for a human.
        logger.info(
            "agent mode is off, so no checks ran: pass --agent or set CAIRNLINT_AGENT=1"
        )
        return 0
    sources, load_failed = load_sources(args.paths)
    unfiltered_findings = [
        finding for check in _CHECKS.values() for finding in check(sources)
    ]
    findings = sorted(filter_suppressed(unfiltered_findings, sources))
    status = _write_findings(findings) if findings else 0
    return _EXIT_LOAD_FAILED if load_failed else status


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
        help="Python files or directories to check, searched recursively",
    )
    return parser.parse_args(argv, namespace=_Arguments())


def _write_findings(findings: Sequence[Finding]) -> int:
    lines = [finding.render() for finding in findings]
    path = Path(tempfile.gettempdir()) / f"cairnlint-agent-{os.getpid()}.txt"
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
