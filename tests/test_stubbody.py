"""agentstubbody through the command: stub shapes, real bodies, and exemptions."""

import pytest

from tests.conftest import Lint

EMPTY = (
    "has an empty body but its docstring describes behavior: "
    + "implement it or drop the claim"
)
STUB = "stub: wire up a real implementation before use"
UNREAD = (
    "returns without doing the work its signature implies: a one-line function "
    + "that takes inputs but never reads them is usually an unfinished stub"
)
CANNED = (
    "returns a canned value; verify it does what its docstring claims rather "
    + "than standing in as a stub"
)


def _format_load_user_finding(text: str) -> str:
    return f'mod.py:2:5: [agent] function load_user is a "{text}" {STUB}'


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            '''
            def flush():
                """Write buffered data to the store."""
            ''',
            f"mod.py:2:5: [agent] function flush {EMPTY}",
            id="docstring-only-body",
        ),
        pytest.param(
            '''
            class Store:
                def save(self):
                    """Persist the record."""
                    pass
            ''',
            f"mod.py:3:9: [agent] method save {EMPTY}",
            id="docstring-then-pass",
        ),
        pytest.param(
            '''
            def size():
                """Report the buffer size."""
                ...
            ''',
            f"mod.py:2:5: [agent] function size {EMPTY}",
            id="docstring-then-ellipsis",
        ),
        pytest.param(
            """
            def load_user(user_id):
                raise NotImplementedError
            """,
            _format_load_user_finding("NotImplementedError"),
            id="bare-not-implemented",
        ),
        pytest.param(
            """
            def load_user(user_id):
                raise NotImplementedError("load from the database")
            """,
            _format_load_user_finding("load from the database"),
            id="not-implemented-with-message",
        ),
        pytest.param(
            """
            class Store:
                def save(self):
                    raise RuntimeError("TODO: implement save")
            """,
            f'mod.py:3:9: [agent] method save is a "TODO: implement save" {STUB}',
            id="canned-todo-error",
        ),
        pytest.param(
            """
            def save(key):
                raise ValueError(f"TODO: persist {key}")
            """,
            f'mod.py:2:5: [agent] function save is a "TODO: persist " {STUB}',
            id="canned-todo-f-string",
        ),
        pytest.param(
            """
            def validate_config(config):
                return None
            """,
            f"mod.py:2:5: [agent] function validate_config {UNREAD}",
            id="inputs-never-read",
        ),
        pytest.param(
            """
            def options(**overrides):
                return {}
            """,
            f"mod.py:2:5: [agent] function options {UNREAD}",
            id="kwargs-never-read",
        ),
        pytest.param(
            '''
            def buffer_size():
                """Report the configured buffer size in bytes."""
                return 0
            ''',
            f"mod.py:2:5: [agent] function buffer_size {CANNED}",
            id="documented-constant",
        ),
        pytest.param(
            '''
            class Cursor:
                def coords(self):
                    """Report the x and y position."""
                    return (0, -1)
            ''',
            f"mod.py:3:9: [agent] method coords {CANNED}",
            id="documented-constant-tuple",
        ),
    ],
)
def test_stub_shapes_are_reported(lint: Lint, source: str, expected: str) -> None:
    """Each stub shape yields exactly one finding anchored on its def line."""
    run = lint({"mod.py": source})
    assert run.findings == [expected]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """
            class NullCloser:
                def close(self):
                    pass
            """,
            id="undocumented-no-op",
        ),
        pytest.param(
            '''
            def add(a, b):
                """Return the sum of a and b."""
                return a + b
            ''',
            id="computed-return",
        ),
        pytest.param(
            '''
            class Box:
                def value(self):
                    """Return the stored value."""
                    return self._value
            ''',
            id="accessor",
        ),
        pytest.param(
            '''
            def check_positive(x):
                """Raise when x is negative."""
                if x < 0:
                    raise ValueError("negative")
            ''',
            id="real-validation",
        ),
        pytest.param(
            """
            def reject(x):
                raise ValueError("negative input")
            """,
            id="raise-without-stub-wording",
        ),
        pytest.param(
            '''
            class Flag:
                def __bool__(self):
                    """A flag is always set."""
                    return True
            ''',
            id="dunder-constant",
        ),
        pytest.param(
            '''
            def outer():
                """Build the handler."""
                def handler(event):
                    return None
                return handler
            ''',
            id="nested-function",
        ),
    ],
)
def test_real_bodies_are_not_reported(lint: Lint, source: str) -> None:
    """Bodies that do work, or undocumented no-ops, produce no findings."""
    run = lint({"mod.py": source})
    assert run.findings == []


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            '''
            from abc import ABC, abstractmethod

            class Store(ABC):
                @abstractmethod
                def save(self, record):
                    """Persist the record."""
                    raise NotImplementedError
            ''',
            id="abstractmethod",
        ),
        pytest.param(
            '''
            from typing import Protocol

            class Store(Protocol):
                def save(self, record) -> None:
                    """Persist the record."""
                    ...
            ''',
            id="protocol-member",
        ),
        pytest.param(
            '''
            from typing import overload

            @overload
            def parse(value: int) -> int: ...
            @overload
            def parse(value: str) -> str:
                """Parse a string."""
            def parse(value):
                return value
            ''',
            id="overload",
        ),
    ],
)
def test_declarations_are_exempt(lint: Lint, source: str) -> None:
    """Abstract methods, Protocol members, and overloads are declarations."""
    run = lint({"mod.py": source})
    assert run.findings == []


@pytest.mark.parametrize(
    "name",
    ["mod.pyi", "test_mod.py", "mod_test.py", "conftest.py", "tests/fakes.py"],
)
def test_stub_and_test_files_are_exempt(lint: Lint, name: str) -> None:
    """Stub files and test code hold intentional fakes, so they are skipped."""
    source = '''
        def flush():
            """Write buffered data to the store."""
    '''
    run = lint({name: source})
    assert run.findings == []
