"""agentpassthrough through the command: forwarders, wrappers, and real work."""

import pytest

from tests.conftest import Lint


def _format_forwards_finding(position: str, method: str, target: str) -> str:
    return (
        f"{position}: [agent] method {method} only forwards its arguments to {target} "
        + "and adds no behavior of its own"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            """
            class Wrapper:
                def __init__(self, inner):
                    self._inner = inner

                def join(self, parts):
                    return self._inner.join(parts)

                @property
                def sep(self):
                    return self._inner.sep_value()
            """,
            [
                "mod.py:2:7: [agent] class Wrapper wraps one field and every one "
                + "of its 2 methods forwards; callers can hold the wrapped value "
                + "directly"
            ],
            id="one-field-wrapper",
        ),
        pytest.param(
            """
            class Mixed:
                def __init__(self, inner):
                    self._inner = inner
                    self._seen = 0

                def join(self, parts):
                    return self._inner.join(parts)

                def count(self):
                    self._seen += 1
                    return self._seen
            """,
            [_format_forwards_finding("mod.py:7:9", "Mixed.join", "self._inner.join")],
            id="lone-forwarder-on-working-class",
        ),
        pytest.param(
            """
            class Client:
                def __init__(self, session, base):
                    self._session = session
                    self._base = base

                async def fetch(self, url, *args, timeout, **opts):
                    return await self._session.get(url, *args, timeout=timeout, **opts)

                def close(self):
                    self._session.close()
            """,
            [
                _format_forwards_finding(
                    "mod.py:7:15", "Client.fetch", "self._session.get"
                ),
                _format_forwards_finding(
                    "mod.py:10:9", "Client.close", "self._session.close"
                ),
            ],
            id="varargs-keywords-and-await-on-two-field-class",
        ),
        pytest.param(
            """
            class Child(Base):
                def render(self, page):
                    return super().render(page)

                def size(self):
                    return 3
            """,
            [_format_forwards_finding("mod.py:3:9", "Child.render", "super().render")],
            id="super-forward",
        ),
        pytest.param(
            """
            from typing import ClassVar

            class Wrapper:
                kind: ClassVar[str] = "wrapper"

                def __init__(self, inner):
                    self._inner = inner

                def join(self, parts):
                    return self._inner.join(parts)
            """,
            [
                "mod.py:4:7: [agent] class Wrapper wraps one field and every one "
                + "of its 1 methods forwards; callers can hold the wrapped value "
                + "directly"
            ],
            id="class-variable-is-not-a-field",
        ),
        pytest.param(
            """
            import os

            class Paths:
                def __init__(self, clock):
                    self.clock = clock

                def join(self, a, b):
                    return os.path.join(a, b)

                def now(self):
                    return self.clock.now()
            """,
            [
                _format_forwards_finding("mod.py:8:9", "Paths.join", "os.path.join"),
                _format_forwards_finding("mod.py:11:9", "Paths.now", "self.clock.now"),
            ],
            id="forward-that-skips-the-field",
        ),
    ],
)
def test_forwarding_is_reported(lint: Lint, source: str, expected: list[str]) -> None:
    """Forwarders report per method, or once at a one-field wrapper class."""
    run = lint({"mod.py": source})
    assert run.findings == expected


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """
            class Guarded:
                def __init__(self, inner):
                    self._inner = inner

                def join(self, parts):
                    if not parts:
                        return ""
                    return self._inner.join(parts)
            """,
            id="branches-first",
        ),
        pytest.param(
            """
            class Reordered:
                def __init__(self, inner):
                    self._inner = inner

                def pair(self, left, right):
                    return self._inner.pair(right, left)

                def head(self, parts):
                    return self._inner.join(["head", *parts])
            """,
            id="reshaped-arguments",
        ),
        pytest.param(
            """
            class Partial:
                def __init__(self, inner):
                    self._inner = inner

                def join(self, parts, sep):
                    return self._inner.join(parts)
            """,
            id="drops-a-parameter",
        ),
        pytest.param(
            """
            class KeywordPartial:
                def __init__(self, inner):
                    self._inner = inner

                def join(self, parts, *, sep):
                    return self._inner.join(parts)
            """,
            id="drops-a-keyword-only-parameter",
        ),
        pytest.param(
            """
            class Blank:
                def __init__(self, inner):
                    self._inner = inner

                def join(self, _):
                    return self._inner.join(_)
            """,
            id="blank-parameter",
        ),
        pytest.param(
            """
            import functools

            class Cached:
                def __init__(self, inner):
                    self._inner = inner

                @functools.cache
                def join(self, parts):
                    return self._inner.join(parts)
            """,
            id="behavior-adding-decorator",
        ),
        pytest.param(
            """
            def join(parts):
                return SEPARATOR.join(parts)
            """,
            id="module-function",
        ),
        pytest.param(
            """
            from typing import Protocol

            class Joiner(Protocol):
                def join(self, parts): return self.join(parts)
            """,
            id="protocol-class",
        ),
    ],
)
def test_real_work_is_not_reported(lint: Lint, source: str) -> None:
    """Methods that branch, reshape, or are wrapped do work of their own."""
    run = lint({"mod.py": source})
    assert run.findings == []


def test_test_files_are_skipped(lint: Lint) -> None:
    """A forwarder in a test file is scaffolding, not a finding."""
    source = """
        class Spy:
            def __init__(self, inner):
                self._inner = inner

            def join(self, parts):
                return self._inner.join(parts)
    """
    run = lint({"test_spy.py": source})
    assert run.findings == []
