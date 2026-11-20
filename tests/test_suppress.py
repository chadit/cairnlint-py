"""`# nolint` directives through the command."""

import pytest

from tests.conftest import Lint

STUB = (
    'mod.py:2:5: [agent] function load is a "NotImplementedError" stub: '
    + "wire up a real implementation before use"
)


@pytest.mark.parametrize(
    "directive",
    [
        pytest.param("# nolint:agentstubbody", id="named"),
        pytest.param("# nolint:AgentStubBody", id="case-insensitive"),
        pytest.param("#nolint: agentloneparam , agentstubbody", id="list-with-spaces"),
        pytest.param("# nolint", id="bare"),
        pytest.param("# nolint:all", id="all"),
        pytest.param("# nolint:agentstubbody # loaded lazily by a plugin", id="reason"),
    ],
)
def test_trailing_directive_suppresses_its_line(lint: Lint, directive: str) -> None:
    """A matching directive on the anchored line drops the finding."""
    source = f"\ndef load(path):  {directive}\n    raise NotImplementedError\n"
    run = lint({"mod.py": source})
    assert run.findings == []


@pytest.mark.parametrize(
    "directive",
    [
        pytest.param("# nolint:agentpassthrough", id="other-analyzer"),
        pytest.param("# nolint:", id="empty-list"),
        pytest.param("# nolint:agentstubbody reason", id="reason-without-separator"),
        pytest.param("# see nolint:agentstubbody", id="prose-mention"),
        pytest.param("# nolintagentstubbody", id="no-separator-after-keyword"),
    ],
)
def test_non_matching_directive_keeps_the_finding(lint: Lint, directive: str) -> None:
    """A directive naming nothing that matches leaves the finding in place."""
    source = f"\ndef load(path):  {directive}\n    raise NotImplementedError\n"
    run = lint({"mod.py": source})
    assert run.findings == [STUB]


def test_leading_directive_covers_the_whole_statement(lint: Lint) -> None:
    """A directive above a def suppresses findings on any line of the def."""
    source = """
        import functools

        # nolint:agentloneparam

        def render(
            template,
            strict,
        ):
            return template if strict else ""

        def plain(template, strict):
            return template if strict else ""

        render("a", True)
        render("b", True)
        plain("a", True)
        plain("b", True)
    """
    run = lint({"mod.py": source})
    assert run.findings == [
        "mod.py:12:21: [agent] parameter strict of plain receives True at all 2 "
        + "call sites in the analyzed paths; the choice it encodes is never made"
    ]


def test_leading_directive_above_decorators_covers_the_def(lint: Lint) -> None:
    """Decorator lines belong to the statement a leading directive covers."""
    source = """
        import functools

        # nolint:agentstubbody
        @functools.cache
        @functools.wraps(print)
        def load(path):
            raise NotImplementedError
    """
    run = lint({"mod.py": source})
    assert run.findings == []


def test_directive_applies_only_to_its_own_file(lint: Lint) -> None:
    """A directive on the same line number in another file suppresses nothing."""
    run = lint(
        {
            "mod.py": "\ndef load(path):\n    raise NotImplementedError\n",
            "other.py": "\nVALUE = 1  # nolint\n",
        }
    )
    assert run.findings == [STUB]


@pytest.mark.parametrize(
    ("source", "line"),
    [
        pytest.param(
            """
            CONFIG = dict(
                # nolint:agentstubbody
                name="x",
            )


            def load(path):
                raise NotImplementedError
            """,
            8,
            id="inside-a-bracketed-call",
        ),
        pytest.param(
            """
            def first():
                return 1
                # nolint:agentstubbody


            def load(path):
                raise NotImplementedError
            """,
            7,
            id="end-of-previous-body",
        ),
    ],
)
def test_directive_not_directly_above_a_statement_covers_only_its_line(
    lint: Lint, source: str, line: int
) -> None:
    """A directive must sit directly above a statement to cover it."""
    run = lint({"mod.py": source})
    assert run.findings == [STUB.replace("mod.py:2:5:", f"mod.py:{line}:5:")]


def test_directive_between_decorator_and_def_covers_the_def(lint: Lint) -> None:
    """A directive below the decorators still sits directly above the def."""
    source = """
        import functools


        @functools.cache
        # nolint:agentstubbody
        def load(path):
            raise NotImplementedError
    """
    run = lint({"mod.py": source})
    assert run.findings == []
