"""agentloneparam: a parameter that receives the same constant at every call site.

The branch such a parameter selects never varies, so the parameter and the dead
half of the body can go. Calls resolve to declarations by name across the
analyzed paths, with no type inference, so the check is agent-only and skips
every name it cannot attribute to one declaration.
"""

import ast
import builtins
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cairnlint_py.source import Finding, SourceFile
from cairnlint_py.syntax import (
    FunctionNode,
    has_only_plain_decorators,
    has_receiver,
    is_dunder,
    iter_functions,
    iter_nodes,
    list_bound_params,
    list_positional_params,
)

__all__ = ["ANALYZER", "check"]

ANALYZER = "agentloneparam"

# A call such as `", ".join(parts)` or `open(path)` names a builtin, and with
# no type information it would be pinned on a project function of the same
# name, so those names are never tracked.
_AMBIGUOUS_NAMES = frozenset(dir(builtins)).union(
    *(dir(kind) for kind in (str, bytes, list, dict, set, tuple, int, float, object))
)

# One call site proves nothing about variation, so two is the minimum before
# "always receives" means anything.
_MIN_CALL_SITES = 2

# Nodes that bind the name in their `name` field, when it is set: definitions,
# type parameters, `except ... as` handlers, and match captures.
_NAMED_BINDERS = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.TypeVar,
    ast.ParamSpec,
    ast.TypeVarTuple,
    ast.ExceptHandler,
    ast.MatchAs,
    ast.MatchStar,
)


# A constant argument as (identity, spelling). Identity decides whether two
# call sites pass the same value; the spelling is what the message shows.
type _Value = tuple[str, str]
type _Binding = dict[str, _Value | None]


@dataclass(frozen=True)
class _Import:
    """Where an imported name comes from, as the import statement wrote it."""

    module: str
    name: str | None
    level: int


# None marks a name the module binds itself, by assignment or definition.
type _Origin = _Import | None


@dataclass(frozen=True)
class _ModuleIndex:
    origins: dict[Path, dict[str, _Origin]]
    paths_by_module: dict[str, list[Path]]
    # How many places in each file bind each name, at any depth.
    binding_counts: dict[Path, Counter[str]]
    # Every directory that holds an analyzed file at any depth, built once
    # so a namespace-directory test is a set lookup, not a scan of all files.
    directories: frozenset[Path]


@dataclass
class _Callee:
    source: SourceFile
    node: FunctionNode
    owner: ast.ClassDef | None
    calls: list[_Binding] = field(default_factory=list)
    unresolvable: bool = False


def check(sources: Sequence[SourceFile]) -> list[Finding]:
    """Report parameters fixed across two or more call sites."""
    callees = _collect_tracked_callees(sources)
    call_targets: set[int] = set()
    index = _build_module_index(sources)
    # Call sites in test files count, as they do in the Go check.
    for source, call in iter_nodes(sources, ast.Call):
        call_targets.add(id(call.func))
        callee = callees.get(_read_called_name(call) or "")
        if callee is not None:
            _record_call(callee, call, index, source.path)
    for name in _iter_value_references(sources, call_targets):
        _ = callees.pop(name, None)
    findings: list[Finding] = []
    for callee in callees.values():
        findings.extend(_build_findings(callee))
    return findings


def _collect_tracked_callees(sources: Sequence[SourceFile]) -> dict[str, _Callee]:
    """Production declarations whose name no other function in the tree shares."""
    declared = Counter(node.name for _, node in iter_nodes(sources, ast.FunctionDef))
    declared.update(node.name for _, node in iter_nodes(sources, ast.AsyncFunctionDef))
    # A package's __init__ re-exports names its modules define, so every
    # __all__ in the tree counts, not only the defining module's own.
    production = [source for source in sources if not source.is_test]
    exported = set[str]().union(
        *(_collect_exported_names(source.tree) for source in production)
    )
    tracked: dict[str, _Callee] = {}
    for source in production:
        tracked.update(
            {
                node.name: _Callee(source, node, owner)
                for node, owner in iter_functions(source.tree)
                if declared[node.name] == 1 and _is_trackable(node, exported)
            }
        )
    return tracked


def _collect_exported_names(tree: ast.Module) -> set[str]:
    """Names in the module's `__all__`, which callers outside the tree may use."""
    names: set[str] = set()
    for statement in tree.body:
        if _assigns_all(statement):
            names.update(_read_string_items(getattr(statement, "value", None)))
    return names


def _assigns_all(statement: ast.stmt) -> bool:
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign):
        targets = [statement.target]
    return any(_spell_dotted_name(target) == "__all__" for target in targets)


def _read_string_items(value: ast.expr | None) -> list[str]:
    if not isinstance(value, ast.List | ast.Tuple):
        return []
    return [
        item.value
        for item in value.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]


def _is_trackable(node: FunctionNode, exported: set[str]) -> bool:
    name = node.name
    return not (
        is_dunder(name)
        or name in exported
        or name in _AMBIGUOUS_NAMES
        or node.args.vararg is not None
        or not has_only_plain_decorators(node)
    )


def _read_called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _iter_value_references(
    sources: Sequence[SourceFile], call_targets: set[int]
) -> Iterator[str]:
    """Names read as values, such as a callback, whose calls are invisible."""
    for _, node in iter_nodes(sources, ast.AST):
        if id(node) in call_targets:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            yield node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            yield node.attr
        elif isinstance(node, ast.alias) and node.asname:
            # Calls through `import render as draw` use a name the check never
            # resolves, so the original name is as good as passed around.
            yield node.name.rpartition(".")[2]


def _record_call(
    callee: _Callee, call: ast.Call, index: _ModuleIndex, caller: Path
) -> None:
    binding = _bind(callee, call, index, caller)
    if binding is None:
        callee.unresolvable = True
    else:
        callee.calls.append(binding)


def _bind(
    callee: _Callee, call: ast.Call, index: _ModuleIndex, caller: Path
) -> _Binding | None:
    """Map each parameter to the constant it receives, or None when it varies.

    Returns None when the call cannot be matched to the signature: unpacked
    arguments, a bare call to a method, or arguments the signature rejects,
    which means the name also belongs to something outside the tree.
    """
    is_bare = isinstance(call.func, ast.Name)
    if is_bare and has_receiver(callee.node, callee.owner):
        return None
    arguments_by_param = _map_supplied_arguments(callee, call)
    if arguments_by_param is None:
        return None
    binding: _Binding = {}
    for name, default in _list_parameters(callee):
        if name in arguments_by_param:
            argument = arguments_by_param[name]
            binding[name] = _build_constant_value(argument, index, caller)
        elif default is not None:
            module = callee.source.path
            binding[name] = _build_constant_value(default, index, module)
        else:
            return None
    return binding


def _map_supplied_arguments(
    callee: _Callee, call: ast.Call
) -> dict[str, ast.expr] | None:
    arguments = callee.node.args
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return None
    positional = [param.arg for param in list_bound_params(callee.node, callee.owner)]
    if len(call.args) > len(positional):
        return None
    arguments_by_param = dict(zip(positional, call.args, strict=False))
    by_keyword = {param.arg for param in [*arguments.args, *arguments.kwonlyargs]}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in arguments_by_param:
            return None
        if keyword.arg not in by_keyword and arguments.kwarg is None:
            return None
        arguments_by_param[keyword.arg] = keyword.value
    return arguments_by_param


def _list_parameters(callee: _Callee) -> list[tuple[str, ast.expr | None]]:
    """Each parameter a caller supplies, paired with its default if any."""
    arguments = callee.node.args
    positional = list_positional_params(arguments)
    defaults: list[ast.expr | None] = [None] * len(positional)
    defaults[len(positional) - len(arguments.defaults) :] = arguments.defaults
    pairs = list(zip(positional, defaults, strict=True))
    if has_receiver(callee.node, callee.owner):
        pairs = pairs[1:]
    pairs.extend(zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True))
    return [(param.arg, default) for param, default in pairs]


def _spell_dotted_name(node: ast.expr) -> str | None:
    """Spell a Name or an Attribute chain such as `abc.ABC`, or return None."""
    if isinstance(node, ast.Name):
        return node.id
    if not isinstance(node, ast.Attribute):
        return None
    base = _spell_dotted_name(node.value)
    return None if base is None else f"{base}.{node.attr}"


def _build_constant_value(
    value: ast.expr, index: _ModuleIndex, module: Path
) -> _Value | None:
    """The value a constant argument passes, or None for anything else.

    An upper-case name such as `DEFAULT_MODE` or `Mode.STRICT` counts as a
    constant by convention, the way a Go const does in the Go check. Its base
    name has to resolve to a top-level binding, so two modules that each bind
    their own `ANALYZER` pass different values.
    """
    if isinstance(value, ast.Constant):
        return repr(value.value), repr(value.value)
    if isinstance(value, ast.UnaryOp) and isinstance(value.operand, ast.Constant):
        return ast.unparse(value), ast.unparse(value)
    spelling = _spell_dotted_name(value)
    if spelling is None or not spelling.rpartition(".")[2].isupper():
        return None
    base, *attributes = spelling.split(".")
    # The name means the module's binding only when nothing else in the file
    # binds it: no parameter, local, class attribute, loop or comprehension
    # target, import alias, or definition. That gives up a finding when a
    # constant's name is reused, and never mistakes a shadow for the constant.
    if index.binding_counts[module][base] != 1:
        return None
    identity = _resolve_binding(index, module, base, frozenset())
    # `consts.KIND` names the same binding as `from consts import KIND`, so
    # attributes of an in-tree module resolve inside that module.
    while attributes and isinstance(identity, Path):
        identity = _resolve_in_module(index, identity, attributes.pop(0), frozenset())
    if identity is None:
        return None
    return ".".join([str(identity), *attributes]), spelling


def _build_module_index(sources: Sequence[SourceFile]) -> _ModuleIndex:
    origins = {source.path: _collect_origins(source.tree) for source in sources}
    paths_by_module: dict[str, list[Path]] = {}
    for source in sources:
        for module in _list_module_names(source.path):
            paths_by_module.setdefault(module, []).append(source.path)
    counts = {source.path: _count_bindings(source.tree) for source in sources}
    directories = frozenset(
        directory for source in sources for directory in source.path.parents
    )
    return _ModuleIndex(origins, paths_by_module, counts, directories)


def _list_module_names(path: Path) -> list[str]:
    """Every dotted name an absolute import could use for path.

    Without the import root, `pkg/consts.py` may be imported as `consts`,
    `pkg.consts`, or longer, so each suffix of the path is a candidate. A
    suffix that starts inside a package is not: nothing imports
    `app/logging.py` as `logging`, which would hide the standard library's.
    """
    parts = _split_module_parts(path)
    return [
        ".".join(parts[start:])
        for start in range(1, len(parts))
        if not Path(*parts[:start], "__init__.py").exists()
    ]


def _collect_origins(tree: ast.Module) -> dict[str, _Origin]:
    """Where each name the module binds at its top level comes from."""
    origins: dict[str, _Origin] = {}
    for statement in tree.body:
        origins.update(_read_origins(statement))
    return origins


def _read_origins(statement: ast.stmt) -> list[tuple[str, _Origin]]:
    if isinstance(statement, ast.Import):
        return [_read_import_origin(alias) for alias in statement.names]
    if isinstance(statement, ast.ImportFrom):
        module = statement.module or ""
        return [
            (alias.asname or alias.name, _Import(module, alias.name, statement.level))
            for alias in statement.names
        ]
    if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return [(statement.name, None)]
    if isinstance(statement, ast.Assign):
        return [
            (target.id, None)
            for target in statement.targets
            if isinstance(target, ast.Name)
        ]
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return [(statement.target.id, None)]
    return []


def _read_import_origin(alias: ast.alias) -> tuple[str, _Origin]:
    # `import a.b` binds `a`, while `import a.b as c` binds `c` to `a.b`.
    if alias.asname:
        return alias.asname, _Import(alias.name, None, 0)
    package = alias.name.partition(".")[0]
    return package, _Import(package, None, 0)


def _resolve_binding(
    index: _ModuleIndex, module: Path, name: str, seen: frozenset[tuple[Path, str]]
) -> str | Path | None:
    """An identity for the binding `name` refers to at the top of module.

    Imports are followed to the defining module in the tree. A module outside
    the tree is identified by its absolute name, which is the same wherever it
    is imported. An in-tree module object resolves to its file. A name that
    is unbound, or whose module is ambiguous or missing, resolves to None and
    so never matches another call site.
    """
    origins = index.origins.get(module, {})
    if name not in origins or (module, name) in seen:
        return None
    origin = origins[name]
    if origin is None:
        return f"{module}:{name}"
    target = _locate_module(index, module, origin)
    if target is None or isinstance(target, str):
        return _name_external(target, origin)
    if origin.name is None:
        return target
    return _resolve_in_module(index, target, origin.name, seen | {(module, name)})


def _resolve_in_module(
    index: _ModuleIndex, module: Path, name: str, seen: frozenset[tuple[Path, str]]
) -> str | Path | None:
    """Resolve name in module, falling back to a submodule of a package.

    `from pkg import consts` binds the file pkg/consts.py when the package's
    `__init__` does not bind `consts` itself.
    """
    identity = _resolve_binding(index, module, name, seen)
    if identity is not None or module.name != "__init__.py":
        return identity
    candidates = [module.parent / f"{name}.py", module.parent / name / "__init__.py"]
    return next((path for path in candidates if path in index.origins), None)


def _name_external(target: str | None, origin: _Import) -> str | None:
    if target is None or origin.name is None:
        return target
    return f"{target}.{origin.name}"


def _locate_module(
    index: _ModuleIndex, module: Path, origin: _Import
) -> Path | str | None:
    """The file an import refers to, a name standing for its target, or None."""
    if origin.level:
        return _locate_relative(index, module, origin)
    matches = index.paths_by_module.get(origin.module, [])
    # A script reaches a same-named file or directory beside it, while a package
    # module reaches the match or the outside module, so within one directory
    # the name means one thing and the directory scopes its identity, unless the
    # neighbour is the sole match. Accepted limit: such a scoped name never
    # matches one from another directory.
    neighbour = _find_neighbour(index, module, origin.module)
    if neighbour is not None and matches != [neighbour]:
        return f"{module.parent}:{origin.module}"
    if matches:
        return matches[0] if len(matches) == 1 else None
    # A directory without __init__.py is a namespace package with no file of
    # its own, so a module under it makes the name in-tree but unresolved.
    prefix = f"{origin.module}."
    if any(name.startswith(prefix) for name in index.paths_by_module):
        return None
    return origin.module


def _split_module_parts(path: Path) -> tuple[str, ...]:
    """The path's parts as a module name sees them, a package as its directory."""
    parts = path.with_suffix("").parts
    return parts[:-1] if parts[-1] == "__init__" else parts


def _find_neighbour(index: _ModuleIndex, module: Path, dotted: str) -> Path | None:
    """What a bare import of dotted from module would reach beside it, if anything.

    That is a module file, a package's `__init__.py`, or a namespace directory,
    which has no `__init__.py` of its own but holds analyzed files.
    """
    base = module.parent.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate in index.origins:
            return candidate
    return base if _is_namespace_directory(index, base) else None


def _is_namespace_directory(index: _ModuleIndex, directory: Path) -> bool:
    return (
        directory in index.directories
        and directory / "__init__.py" not in index.origins
    )


def _locate_relative(index: _ModuleIndex, module: Path, origin: _Import) -> Path | None:
    if origin.level > len(module.parents):
        return None
    base = module.parents[origin.level - 1].joinpath(*origin.module.split("."))
    candidates = [base.with_suffix(".py"), base / "__init__.py"]
    return next((path for path in candidates if path in index.origins), None)


def _build_findings(callee: _Callee) -> list[Finding]:
    if callee.unresolvable or len(callee.calls) < _MIN_CALL_SITES:
        return []
    params = {param.arg: param for param in _list_all_params(callee.node)}
    findings: list[Finding] = []
    for name in callee.calls[0]:
        first = callee.calls[0][name]
        identities = {_read_identity(binding[name]) for binding in callee.calls}
        if len(identities) != 1 or first is None or name == "_":
            continue
        _, spelling = first
        findings.append(
            callee.source.build_finding(
                params[name],
                ANALYZER,
                f"[agent] parameter {name} of {callee.node.name} receives {spelling} "
                + f"at all {len(callee.calls)} call sites in the analyzed paths; "
                + "the choice it encodes is never made",
            )
        )
    return findings


def _count_bindings(tree: ast.Module) -> Counter[str]:
    return Counter(
        name for node in ast.walk(tree) for name in _read_binding_names(node)
    )


def _read_binding_names(node: ast.AST) -> list[str]:
    """The names node binds in whatever scope it sits in."""
    if isinstance(node, ast.Name):
        return [node.id] if isinstance(node.ctx, ast.Store) else []
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.alias):
        return [node.asname or node.name.partition(".")[0]]
    if isinstance(node, _NAMED_BINDERS):
        return [node.name] if node.name else []
    if isinstance(node, ast.MatchMapping):
        return [node.rest] if node.rest else []
    return []


def _read_identity(value: _Value | None) -> str | None:
    return None if value is None else value[0]


def _list_all_params(node: FunctionNode) -> list[ast.arg]:
    arguments = node.args
    return [*list_positional_params(arguments), *arguments.kwonlyargs]
