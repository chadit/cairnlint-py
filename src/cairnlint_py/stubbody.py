"""agentstubbody: functions whose body does nothing their name or docstring claims.

These parse, type-check, and pass the standard linters, so only a reader
catches the gap. The check is agent-only because a documented no-op hook or a
null object looks the same as an unfinished stub.
"""

import ast
import json
import re
from collections.abc import Sequence

from cairnlint_py.source import Finding, SourceFile
from cairnlint_py.syntax import (
    ABSTRACT_DECORATORS,
    FunctionNode,
    collect_decorator_names,
    find_last_name,
    is_dunder,
    is_protocol_class,
    iter_functions,
    list_bound_params,
    strip_docstring,
)

__all__ = ["ANALYZER", "check"]

ANALYZER = "agentstubbody"

# Kept broad on purpose: agent mode favors recall, and an LLM dismisses the
# rare false hit. The same pattern cairnlint uses for canned Go errors.
_NOT_IMPLEMENTED = re.compile(
    r"(?i)\b(not[ _-]?implemented|unimplemented|to[ _-]?do|stub|placeholder|fixme)\b"
)

# Each of these marks a body that is a declaration by design.
_DECLARATION_DECORATORS = ABSTRACT_DECORATORS | {"overload"}


def check(sources: Sequence[SourceFile]) -> list[Finding]:
    """Report stub bodies in non-test files."""
    findings: list[Finding] = []
    for source in sources:
        if not source.is_test:
            findings.extend(_check_file(source))
    return findings


def _check_file(source: SourceFile) -> list[Finding]:
    findings: list[Finding] = []
    for function, owner in iter_functions(source.tree):
        message = _build_stub_message(function, owner)
        if message is not None:
            findings.append(source.build_finding(function, ANALYZER, message))
    return findings


def _build_stub_message(
    function: FunctionNode, owner: ast.ClassDef | None
) -> str | None:
    if _is_declaration(function, owner):
        return None
    kind = "function" if owner is None else "method"
    label = f"{kind} {function.name}"
    has_doc = bool((ast.get_docstring(function) or "").strip())
    body = strip_docstring(function)
    if all(map(_is_placeholder, body)):
        return _build_empty_body_message(label, has_doc)
    if len(body) != 1:
        return None
    if isinstance(body[0], ast.Raise):
        return _build_sentinel_message(label, body[0])
    if isinstance(body[0], ast.Return) and _is_canned_value(body[0].value):
        return _build_canned_return_message(function, owner, label, has_doc)
    return None


def _is_declaration(function: FunctionNode, owner: ast.ClassDef | None) -> bool:
    if owner is not None and is_protocol_class(owner):
        return True
    return not _DECLARATION_DECORATORS.isdisjoint(collect_decorator_names(function))


def _is_placeholder(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    )


def _build_empty_body_message(label: str, has_doc: bool) -> str | None:
    # Without a docstring there is no claim to break: an undocumented `pass`
    # is an intentional no-op, as an empty Go body without a doc comment is.
    if not has_doc:
        return None
    return (
        f"[agent] {label} has an empty body but its docstring describes behavior: "
        "implement it or drop the claim"
    )


def _build_sentinel_message(label: str, statement: ast.Raise) -> str | None:
    text = _find_sentinel_text(statement.exc)
    if text is None:
        return None
    return (
        f"[agent] {label} is a {json.dumps(text)} stub: "
        "wire up a real implementation before use"
    )


def _find_sentinel_text(exception: ast.expr | None) -> str | None:
    if exception is None:
        return None
    message = _read_first_string_argument(exception)
    if find_last_name(exception) == "NotImplementedError":
        return message or "NotImplementedError"
    if message is not None and _NOT_IMPLEMENTED.search(message):
        return message
    return None


def _read_first_string_argument(exception: ast.expr) -> str | None:
    if not isinstance(exception, ast.Call) or not exception.args:
        return None
    first = exception.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    if not isinstance(first, ast.JoinedStr):
        return None
    # An f-string's literal parts carry the wording, as fmt.Errorf's format
    # string does for the Go check.
    return "".join(
        part.value
        for part in first.values
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    )


def _is_canned_value(value: ast.expr | None) -> bool:
    if value is None or isinstance(value, ast.Constant):
        return True
    if isinstance(value, ast.Tuple | ast.List | ast.Set):
        return all(map(_is_canned_value, value.elts))
    if isinstance(value, ast.Dict):
        return None not in value.keys and all(
            map(_is_canned_value, [*value.keys, *value.values])
        )
    if isinstance(value, ast.UnaryOp):
        return isinstance(value.operand, ast.Constant)
    return False


def _build_canned_return_message(
    function: FunctionNode, owner: ast.ClassDef | None, label: str, has_doc: bool
) -> str | None:
    # Dunder methods answer protocol hooks such as __bool__ or __exit__, where a
    # constant is often the whole correct answer.
    if is_dunder(function.name):
        return None
    takes_inputs = bool(
        list_bound_params(function, owner)
        or function.args.kwonlyargs
        or function.args.vararg
        or function.args.kwarg
    )
    if takes_inputs:
        return (
            f"[agent] {label} returns without doing the work its signature "
            "implies: a one-line function that takes inputs but never reads "
            "them is usually an unfinished stub"
        )
    if has_doc:
        return (
            f"[agent] {label} returns a canned value; verify it does what its "
            "docstring claims rather than standing in as a stub"
        )
    return None
