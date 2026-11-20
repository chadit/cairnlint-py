"""agentloneimpl: a Protocol or ABC with exactly one implementing class.

The concrete class can go straight into the signatures instead, which removes
the abstraction and the indirection at every call site. Classes resolve by name
across the analyzed paths with no type inference, so the check is agent-only:
an implementation outside those paths, or one hidden behind an alias, is
invisible here, and an LLM searches before acting.
"""

import ast
from collections.abc import Sequence
from dataclasses import dataclass, replace

from cairnlint_py.source import Finding, SourceFile
from cairnlint_py.syntax import (
    ABSTRACT_DECORATORS,
    collect_decorator_names,
    collect_self_attributes,
    find_last_name,
    is_protocol_class,
    iter_nodes,
)

__all__ = ["ANALYZER", "check"]

ANALYZER = "agentloneimpl"


@dataclass(frozen=True)
class _ClassDecl:
    source: SourceFile
    node: ast.ClassDef
    bases: frozenset[str]
    members: frozenset[str]
    is_abstract: bool
    # Own members plus those inherited from bases the tree declares.
    all_members: frozenset[str] = frozenset()

    @property
    def key(self) -> tuple[str, int]:
        return str(self.source.path), self.node.lineno


def check(sources: Sequence[SourceFile]) -> list[Finding]:
    """Report Protocols and ABCs with one implementing class in sources."""
    # Test files are included: a test double is the implementation that most
    # often clears the count, as it does in the Go check.
    classes = [
        _ClassDecl(
            source,
            node,
            frozenset(name for name in map(find_last_name, node.bases) if name),
            _collect_members(node),
            _declares_abstract(node),
        )
        for source, node in iter_nodes(sources, ast.ClassDef)
    ]
    by_name: dict[str, list[_ClassDecl]] = {}
    for item in classes:
        by_name.setdefault(item.node.name, []).append(item)
    classes = [
        replace(item, all_members=_collect_all_members(item, by_name, frozenset()))
        for item in classes
    ]
    registered = _collect_registered_names(sources)
    findings: list[Finding] = []
    for interface in classes:
        kind = _classify_interface(interface)
        # A Protocol or ABC in test code is scaffolding for the test, such as
        # a type for a module it loads by path, not an abstraction the
        # production code carries.
        if kind is None or interface.source.is_test:
            continue
        implementers = _find_implementers(interface, kind, classes, registered)
        if len(implementers) == 1:
            findings.append(_build_finding(interface, kind, implementers[0]))
    return findings


def _collect_members(node: ast.ClassDef) -> frozenset[str]:
    names: set[str] = set()
    for statement in node.body:
        names.update(_collect_statement_names(statement))
    return frozenset(names | collect_self_attributes(node))


def _collect_all_members(
    item: _ClassDecl, by_name: dict[str, list[_ClassDecl]], seen: frozenset[str]
) -> frozenset[str]:
    """Members of item and of every base the tree declares, matched by name."""
    seen = seen | {item.node.name}
    parents = [parent for base in item.bases - seen for parent in by_name.get(base, [])]
    return item.members.union(
        *(_collect_all_members(parent, by_name, seen) for parent in parents)
    )


def _collect_statement_names(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
        return {statement.name}
    if isinstance(statement, ast.AnnAssign):
        return {name for name in [_read_target_name(statement.target)] if name}
    if isinstance(statement, ast.Assign):
        return {name for name in map(_read_target_name, statement.targets) if name}
    return set()


def _read_target_name(target: ast.expr) -> str | None:
    return target.id if isinstance(target, ast.Name) else None


def _declares_abstract(node: ast.ClassDef) -> bool:
    return any(
        not ABSTRACT_DECORATORS.isdisjoint(collect_decorator_names(statement))
        for statement in node.body
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
    )


def _classify_interface(candidate: _ClassDecl) -> str | None:
    if is_protocol_class(candidate.node):
        return _classify_protocol(candidate)
    uses_abc = "ABC" in candidate.bases or any(
        keyword.arg == "metaclass" and find_last_name(keyword.value) == "ABCMeta"
        for keyword in candidate.node.keywords
    )
    return "abstract base class" if uses_abc and candidate.is_abstract else None


def _classify_protocol(candidate: _ClassDecl) -> str | None:
    # A protocol with no members constrains nothing, and one whose only member
    # is __call__ is satisfied by plain functions that no class count can see.
    if not candidate.members or candidate.members == frozenset({"__call__"}):
        return None
    return "protocol"


def _collect_registered_names(sources: Sequence[SourceFile]) -> dict[str, set[str]]:
    """Map each name used as `X.register(Impl)` to the names registered with it."""
    registered: dict[str, set[str]] = {}
    for _, node in iter_nodes(sources, ast.AST):
        registration = _read_registration(node)
        if registration is not None:
            registry, implementer = registration
            registered.setdefault(registry, set()).add(implementer)
    return registered


def _read_registration(node: ast.AST) -> tuple[str, str] | None:
    if isinstance(node, ast.ClassDef):
        return _read_register_decorator(node)
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr != "register" or len(node.args) != 1:
        return None
    registry = find_last_name(node.func.value)
    implementer = find_last_name(node.args[0])
    return (registry, implementer) if registry and implementer else None


def _read_register_decorator(node: ast.ClassDef) -> tuple[str, str] | None:
    decorator = next(
        (
            item
            for item in node.decorator_list
            if isinstance(item, ast.Attribute) and item.attr == "register"
        ),
        None,
    )
    if not isinstance(decorator, ast.Attribute):
        return None
    registry = find_last_name(decorator.value)
    return (registry, node.name) if registry else None


def _find_implementers(
    interface: _ClassDecl,
    kind: str,
    classes: Sequence[_ClassDecl],
    registered: dict[str, set[str]],
) -> list[_ClassDecl | str]:
    implementers_by_key: dict[tuple[str, int] | str, _ClassDecl | str] = {
        item.key: item for item in _find_concrete_subclasses(interface, classes)
    }
    for name in registered.get(interface.node.name, set()):
        matches = [item for item in classes if item.node.name == name]
        if not matches:
            implementers_by_key[name] = name
        implementers_by_key.update((item.key, item) for item in matches)
    if kind == "protocol":
        implementers_by_key.update(
            {item.key: item for item in _find_structural_matches(interface, classes)}
        )
    return [item for key, item in implementers_by_key.items() if key != interface.key]


def _find_concrete_subclasses(
    interface: _ClassDecl, classes: Sequence[_ClassDecl]
) -> list[_ClassDecl]:
    """Concrete descendants by base name, walking through abstract ones."""
    pending = [interface.node.name]
    seen = set(pending)
    concrete_subclasses: list[_ClassDecl] = []
    while pending:
        parent = pending.pop()
        children = [item for item in classes if parent in item.bases]
        concrete_subclasses.extend(item for item in children if not _is_abstract(item))
        new_names = {item.node.name for item in children} - seen
        seen |= new_names
        pending.extend(new_names)
    return concrete_subclasses


def _find_structural_matches(
    interface: _ClassDecl, classes: Sequence[_ClassDecl]
) -> list[_ClassDecl]:
    return [
        item
        for item in classes
        if not _is_abstract(item) and interface.members <= item.all_members
    ]


def _is_abstract(item: _ClassDecl) -> bool:
    """Whether item is itself an interface, which embeds rather than implements."""
    return item.is_abstract or is_protocol_class(item.node)


def _build_finding(interface: _ClassDecl, kind: str, sole: _ClassDecl | str) -> Finding:
    name = sole if isinstance(sole, str) else sole.node.name
    return interface.source.build_finding(
        interface.node,
        ANALYZER,
        f"[agent] {kind} {interface.node.name} has exactly one implementing class "
        + f"in the analyzed paths ({name}) and no test double; using {name} directly "
        + f"removes the {kind} and the indirection at its call sites",
    )
