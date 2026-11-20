"""`# nolint` directives, cairnlint's `//nolint` rewritten for Python comments.

A trailing directive suppresses its own line. A directive on a line of its own
suppresses every line of the statement directly below it, decorators
included, so one above a `def` or `class`, or between its decorators and the
`def`, covers the whole body. Names are analyzer names
such as `agentstubbody`, case-insensitive; a bare `# nolint` or `# nolint:all`
covers every analyzer. The name list ends at the next `#`, which starts the
explanation, so `# nolint:agentstubbody # reason` works and
`# nolint:agentstubbody reason` suppresses nothing, as in golangci-lint.
"""

import ast
import io
import re
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cairnlint_py.source import Finding, SourceFile

__all__ = ["filter_suppressed"]

# The capture keeps the colon so an empty `# nolint:` (names nothing) differs
# from a bare `# nolint` (names everything).
_DIRECTIVE = re.compile(r"^#\s*nolint(?:(:[^#]*)|\s|$)")

_ALL = "all"

# Tokens that carry layout rather than code, skipped when looking for the
# statement a directive sits above.
_LAYOUT = frozenset(
    {tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT}
)


@dataclass(frozen=True)
class _Directive:
    names: frozenset[str]
    first_line: int
    last_line: int

    def covers(self, finding: Finding) -> bool:
        if not self.first_line <= finding.line <= self.last_line:
            return False
        return _ALL in self.names or finding.analyzer in self.names


def filter_suppressed(
    findings: Sequence[Finding], sources: Sequence[SourceFile]
) -> list[Finding]:
    """Drop findings that a directive in their own file suppresses."""
    directives: dict[Path, list[_Directive]] = {
        source.path: _parse_file_directives(source) for source in sources
    }
    return [
        finding
        for finding in findings
        if not any(rule.covers(finding) for rule in directives.get(finding.path, []))
    ]


def _parse_file_directives(source: SourceFile) -> list[_Directive]:
    spans = _map_statement_spans(source.tree)
    tokens = list(tokenize.tokenize(io.BytesIO(source.source_bytes).readline))
    directives: list[_Directive] = []
    for index, token in enumerate(tokens):
        names = _parse(token) if token.type == tokenize.COMMENT else None
        if names is None:
            continue
        first, last = _find_covered_lines(token, tokens[index + 1 :], spans)
        directives.append(_Directive(names, first, last))
    return directives


def _find_covered_lines(
    comment: tokenize.TokenInfo,
    next_tokens: Sequence[tokenize.TokenInfo],
    spans: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    """The first and last line a directive covers.

    A directive on a line of its own covers the statement whose first token is
    the next code after it, provided that statement sits at the directive's
    indentation or deeper. A directive inside a bracketed expression, or one
    left at the end of a block above a shallower statement, covers only its
    own line.
    """
    line, column = comment.start
    statement = _find_next_code_token(next_tokens)
    own_line = not comment.line[:column].strip()
    if not own_line or statement is None or statement.start[1] < column:
        return line, line
    return spans.get(statement.start, (line, line))


def _find_next_code_token(
    next_tokens: Sequence[tokenize.TokenInfo],
) -> tokenize.TokenInfo | None:
    return next((token for token in next_tokens if token.type not in _LAYOUT), None)


def _parse(token: tokenize.TokenInfo) -> frozenset[str] | None:
    match = _DIRECTIVE.match(token.string)
    if match is None:
        return None
    listed = match.group(1)
    if listed is None:
        return frozenset({_ALL})
    return frozenset(name.strip().lower() for name in listed[1:].split(",")) - {""}


def _map_statement_spans(tree: ast.Module) -> dict[tuple[int, int], tuple[int, int]]:
    """Map the position of each statement's first token to its line range.

    A decorated definition is keyed both at its first decorator and at its
    `def` or `class` line, so a directive between the two still covers it.
    """
    spans: dict[tuple[int, int], tuple[int, int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        decorators = (
            node.decorator_list
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            else []
        )
        first = min([node.lineno, *(item.lineno for item in decorators)])
        span = (first, node.end_lineno or node.lineno)
        spans.update(((line, node.col_offset), span) for line in {first, node.lineno})
    return spans
