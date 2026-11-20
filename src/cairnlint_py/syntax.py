"""AST queries that more than one check asks."""

import ast
from collections.abc import Iterator, Sequence

from cairnlint_py.source import SourceFile

__all__ = [
    "ABSTRACT_DECORATORS",
    "FunctionNode",
    "collect_decorator_names",
    "collect_self_attributes",
    "find_last_name",
    "has_only_plain_decorators",
    "has_receiver",
    "is_dunder",
    "is_protocol_class",
    "iter_functions",
    "iter_nodes",
    "list_bound_params",
    "list_positional_params",
    "strip_docstring",
]

type FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

# Decorators that change how a method is bound or read but not what its body
# does, so a function carrying only these is still judged by its body.
_PLAIN_DECORATORS = frozenset({"property", "staticmethod", "classmethod", "override"})

# Each marks a method whose body is a declaration for a subclass to fill in.
ABSTRACT_DECORATORS = frozenset(
    {
        "abstractmethod",
        "abstractproperty",
        "abstractclassmethod",
        "abstractstaticmethod",
    }
)


def iter_nodes[NodeT: ast.AST](
    sources: Sequence[SourceFile], kind: type[NodeT]
) -> Iterator[tuple[SourceFile, NodeT]]:
    """Every node of type kind in sources, paired with the file it is in."""
    for source in sources:
        yield from (
            (source, node) for node in ast.walk(source.tree) if isinstance(node, kind)
        )


def find_last_name(node: ast.expr) -> str | None:
    """The trailing identifier of a name, ignoring call and subscript wrappers.

    Both `typing.Protocol[T]` and `functools.cache()` resolve this way, which is
    what matching by name across files needs.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return find_last_name(node.value)
    if isinstance(node, ast.Call):
        return find_last_name(node.func)
    return None


def collect_decorator_names(node: FunctionNode | ast.ClassDef) -> set[str]:
    """The trailing names of node's decorators."""
    return {name for name in map(find_last_name, node.decorator_list) if name}


def has_only_plain_decorators(node: FunctionNode) -> bool:
    """Whether node carries no decorator beyond the plain binding ones."""
    return collect_decorator_names(node) <= _PLAIN_DECORATORS


def is_dunder(name: str) -> bool:
    """Whether name has the double-underscore form of a Python protocol hook."""
    return name.startswith("__") and name.endswith("__")


def is_protocol_class(node: ast.ClassDef) -> bool:
    """Whether node lists Protocol among its bases, which PEP 544 requires."""
    return any(find_last_name(base) == "Protocol" for base in node.bases)


def strip_docstring(node: FunctionNode) -> list[ast.stmt]:
    """Return `node.body` itself, not a copy, when there is no docstring."""
    if ast.get_docstring(node, clean=False) is None:
        return node.body
    return node.body[1:]


def has_receiver(node: FunctionNode, owner: ast.ClassDef | None) -> bool:
    """Whether the first parameter is bound implicitly as self or cls."""
    return owner is not None and "staticmethod" not in collect_decorator_names(node)


def list_positional_params(arguments: ast.arguments) -> list[ast.arg]:
    """Positional-only parameters, then ordinary ones, in declaration order."""
    return [*arguments.posonlyargs, *arguments.args]


def list_bound_params(node: FunctionNode, owner: ast.ClassDef | None) -> list[ast.arg]:
    """Positional parameters a caller supplies, without the receiver."""
    positional = list_positional_params(node.args)
    return positional[1:] if has_receiver(node, owner) else positional


def collect_self_attributes(node: ast.ClassDef) -> set[str]:
    """Names assigned as `self.<name>` anywhere in the class."""
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and isinstance(child.ctx, ast.Store)
        and isinstance(child.value, ast.Name)
        and child.value.id == "self"
    }


def iter_functions(
    node: ast.AST, owner: ast.ClassDef | None = None
) -> Iterator[tuple[FunctionNode, ast.ClassDef | None]]:
    """Yield module-level functions and methods with the class that owns each.

    Functions nested inside other functions are closures and are not yielded,
    the way the Go checks read declarations but not function literals. The walk
    keeps its own stack, because one long expression nests the tree deeper than
    Python's recursion limit.
    """
    pending = [(child, owner) for child in reversed(list(ast.iter_child_nodes(node)))]
    while pending:
        child, child_owner = pending.pop()
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            yield child, child_owner
            continue
        inner_owner = child if isinstance(child, ast.ClassDef) else child_owner
        pending.extend(
            (grandchild, inner_owner)
            for grandchild in reversed(list(ast.iter_child_nodes(child)))
        )
