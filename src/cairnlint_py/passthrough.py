"""agentpassthrough: methods that only forward, and classes whose every method does.

The class-level shape is Fowler's Middle Man. Forwarding is sometimes the
point, as in a boundary adapter or a method that satisfies an outside
interface, so the check is agent-only and leaves the reason to an LLM.
"""

import ast
from collections import Counter
from collections.abc import Sequence

from cairnlint_py.source import Finding, SourceFile
from cairnlint_py.syntax import (
    FunctionNode,
    collect_self_attributes,
    find_last_name,
    has_only_plain_decorators,
    is_protocol_class,
    iter_nodes,
    list_bound_params,
    strip_docstring,
)

__all__ = ["ANALYZER", "check"]

ANALYZER = "agentpassthrough"

# A constructor stores the wrapped value rather than calling through it, so it
# is neither a forwarder nor a method that does other work.
_CONSTRUCTORS = frozenset({"__init__", "__new__", "__post_init__"})

# Receivers a call can start from that are values the method writes out itself.
_LITERALS = (ast.Constant, ast.JoinedStr, ast.List, ast.Tuple, ast.Set, ast.Dict)


def check(sources: Sequence[SourceFile]) -> list[Finding]:
    """Report forward-only methods and wrapper classes in non-test files."""
    findings: list[Finding] = []
    for source, node in iter_nodes(sources, ast.ClassDef):
        if not source.is_test:
            findings.extend(_check_class(source, node))
    return findings


def _check_class(source: SourceFile, owner: ast.ClassDef) -> list[Finding]:
    if is_protocol_class(owner):
        return []
    methods = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        and statement.name not in _CONSTRUCTORS
    ]
    forwards = [
        (method, target)
        for method in methods
        if (target := _find_forward_target(method, owner)) is not None
    ]
    if not forwards:
        return []
    # One class-level finding replaces one per method, so a wrapper with eight
    # methods does not produce eight findings that all say the same thing.
    if len(forwards) == len(methods) and _forwards_through_one_field(owner, forwards):
        return [
            source.build_finding(
                owner,
                ANALYZER,
                f"[agent] class {owner.name} wraps one field and every one of its "
                + f"{len(methods)} methods forwards; callers can hold the wrapped "
                + "value directly",
            )
        ]
    return [
        source.build_finding(
            method,
            ANALYZER,
            f"[agent] method {owner.name}.{method.name} only forwards its arguments "
            + f"to {target} and adds no behavior of its own",
        )
        for method, target in forwards
    ]


def _forwards_through_one_field(
    owner: ast.ClassDef, forwards: Sequence[tuple[FunctionNode, str]]
) -> bool:
    """Whether the class has one field and every forward goes through it.

    A method that forwards to a module function or another object does not make
    the class a wrapper that callers could replace with its field.
    """
    fields = _collect_fields(owner)
    if len(fields) != 1:
        return False
    prefix = f"self.{next(iter(fields))}."
    return all(target.startswith(prefix) for _, target in forwards)


def _find_forward_target(method: FunctionNode, owner: ast.ClassDef) -> str | None:
    """The spelled call target when method's body is one call passing its params.

    A body that branches, reshapes an argument, or handles an error is doing
    work. A decorator other than the plain binding ones registers or wraps the
    method, so forwarding may be the point and the method is left alone.
    """
    if not has_only_plain_decorators(method):
        return None
    body = strip_docstring(method)
    call = _find_sole_call(body[0]) if len(body) == 1 else None
    if call is None or not isinstance(call.func, ast.Attribute):
        return None
    if _has_literal_receiver(call.func) or not _passes_every_param(method, owner, call):
        return None
    if _declares_default_on_super_forward(method, call.func):
        return None
    return _spell(call.func)


def _has_literal_receiver(target: ast.Attribute) -> bool:
    """Whether the call works on a literal, such as `",".join(parts)`.

    A method on a value the body writes out itself hands nothing to another
    object, so it is real work rather than a forward.
    """
    receiver = target.value
    while isinstance(receiver, ast.Attribute | ast.Subscript | ast.Call):
        receiver = receiver.func if isinstance(receiver, ast.Call) else receiver.value
    return isinstance(receiver, _LITERALS)


def _declares_default_on_super_forward(
    method: FunctionNode, target: ast.Attribute
) -> bool:
    """Whether a super() forward declares a default on any of its parameters.

    A default the override declares may differ from its parent's, and changing
    it is a choice the override makes rather than a forward. Telling would
    mean resolving the parent, so any declared default skips the method, and
    an override that repeats its parent's default is a known miss.
    """
    arguments = method.args
    has_default = bool(arguments.defaults) or any(
        default is not None for default in arguments.kw_defaults
    )
    return has_default and _is_super_call(target.value)


def _is_super_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
    )


def _find_sole_call(statement: ast.stmt) -> ast.Call | None:
    value = statement.value if isinstance(statement, ast.Expr | ast.Return) else None
    if isinstance(value, ast.Await):
        value = value.value
    return value if isinstance(value, ast.Call) else None


def _passes_every_param(
    method: FunctionNode, owner: ast.ClassDef, call: ast.Call
) -> bool:
    arguments = method.args
    positional = [param.arg for param in list_bound_params(method, owner)]
    if arguments.vararg is not None:
        positional.append(f"*{arguments.vararg.arg}")
    keywords: Counter[tuple[str, str | None]] = Counter(
        (param.arg, param.arg) for param in arguments.kwonlyargs
    )
    if arguments.kwarg is not None:
        keywords["**", arguments.kwarg.arg] += 1
    # A blank parameter cannot be forwarded by name, as in the Go check.
    if "_" in positional:
        return False
    return [_spell_argument(arg) for arg in call.args] == positional and Counter(
        map(_spell_keyword, call.keywords)
    ) == keywords


def _spell_argument(argument: ast.expr) -> str | None:
    if isinstance(argument, ast.Name):
        return argument.id
    if isinstance(argument, ast.Starred) and isinstance(argument.value, ast.Name):
        return f"*{argument.value.id}"
    return None


def _spell_keyword(keyword: ast.keyword) -> tuple[str, str | None]:
    value = keyword.value.id if isinstance(keyword.value, ast.Name) else None
    return keyword.arg or "**", value


def _spell(node: ast.expr) -> str:
    """Render a call target such as `self._inner.join` or `super().close`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_spell(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return f"{_spell(node.func)}()"
    return find_last_name(node) or "..."


def _collect_fields(owner: ast.ClassDef) -> set[str]:
    """Instance state: annotated class-body fields and attributes set on self."""
    declared = {
        statement.target.id
        for statement in owner.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and find_last_name(statement.annotation) != "ClassVar"
    }
    return declared | collect_self_attributes(owner)
