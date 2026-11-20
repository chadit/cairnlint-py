"""agentloneimpl through the command: counts by name across the analyzed tree."""

import pytest

from tests.conftest import Lint

PROTOCOL = '''
from typing import Protocol

class Storer(Protocol):
    """Loads a record by id."""

    def load(self, record_id: str) -> bytes: ...
'''

ABC_BASE = """
from abc import ABC, abstractmethod

class Notifier(ABC):
    @abstractmethod
    def notify(self, message: str) -> None: ...
"""

FILE_STORE = """
class FileStore:
    def load(self, record_id: str) -> bytes:
        return record_id.encode()
"""


def _format_lone_finding(position: str, kind: str, name: str, sole: str) -> str:
    return (
        f"{position}: [agent] {kind} {name} has exactly one implementing class in the "
        + f"analyzed paths ({sole}) and no test double; using {sole} directly "
        + f"removes the {kind} and the indirection at its call sites"
    )


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            {"iface.py": PROTOCOL, "store.py": FILE_STORE},
            [_format_lone_finding("iface.py:4:7", "protocol", "Storer", "FileStore")],
            id="protocol-with-one-structural-match",
        ),
        pytest.param(
            {
                "iface.py": PROTOCOL,
                "store.py": "from iface import Storer\n"
                + "class FileStore(Storer):\n"
                + "    def fetch(self) -> None: ...\n",
            },
            [_format_lone_finding("iface.py:4:7", "protocol", "Storer", "FileStore")],
            id="protocol-with-one-explicit-subclass",
        ),
        pytest.param(
            {
                "base.py": ABC_BASE,
                "mail.py": """
                    from base import Notifier

                    class MailNotifier(Notifier):
                        def notify(self, message: str) -> None:
                            print(message)
                """,
            },
            [
                _format_lone_finding(
                    "base.py:4:7", "abstract base class", "Notifier", "MailNotifier"
                )
            ],
            id="abc-with-one-subclass",
        ),
        pytest.param(
            {
                "base.py": ABC_BASE,
                "mail.py": """
                    import abc
                    import base

                    class Partial(base.Notifier):
                        @abc.abstractmethod
                        def render(self) -> str: ...

                    class MailNotifier(Partial):
                        def notify(self, message: str) -> None: print(message)
                        def render(self) -> str: return "mail"
                """,
            },
            [
                _format_lone_finding(
                    "base.py:4:7", "abstract base class", "Notifier", "MailNotifier"
                )
            ],
            id="abc-counted-through-abstract-intermediate",
        ),
        pytest.param(
            {
                "base.py": """
                    import abc

                    class Codec(metaclass=abc.ABCMeta):
                        @abc.abstractmethod
                        def encode(self, text: str) -> bytes: ...

                    class Utf8Codec(Codec):
                        def encode(self, text: str) -> bytes: return text.encode()
                """,
            },
            [
                _format_lone_finding(
                    "base.py:4:7", "abstract base class", "Codec", "Utf8Codec"
                )
            ],
            id="abc-through-metaclass",
        ),
    ],
)
def test_single_implementation_is_reported(
    lint: Lint, files: dict[str, str], expected: list[str]
) -> None:
    """A Protocol or ABC with one implementing class is reported at its class."""
    run = lint(files)
    assert run.findings == expected


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(
            {
                "iface.py": PROTOCOL,
                "store.py": FILE_STORE,
                "memory.py": """
                    class MemoryStore:
                        def __init__(self) -> None:
                            self.load = {}.get
                """,
            },
            id="second-structural-match-by-attribute",
        ),
        pytest.param(
            {
                "iface.py": PROTOCOL,
                "store.py": FILE_STORE,
                "tests/test_store.py": """
                    class FakeStore:
                        def load(self, record_id):
                            return b""
                """,
            },
            id="test-double-clears-the-count",
        ),
        pytest.param({"iface.py": PROTOCOL}, id="no-implementation"),
        pytest.param(
            {
                "iface.py": PROTOCOL,
                "stores.py": """
                    class Base:
                        def load(self, record_id: str) -> bytes:
                            return record_id.encode()

                    class FileStore(Base):
                        root = "/srv"

                    class S3Store(Base):
                        bucket = "records"
                """,
            },
            id="members-inherited-from-a-base",
        ),
        pytest.param(
            {"tests/test_store.py": PROTOCOL, "store.py": FILE_STORE},
            id="protocol-declared-in-test-code",
        ),
        pytest.param(
            {
                "tests/test_notify.py": f"{ABC_BASE}\nclass Fake(Notifier):\n"
                + "    def notify(self, message: str) -> None: ...\n"
            },
            id="abc-declared-in-test-code",
        ),
        pytest.param(
            {
                "iface.py": """
                    from typing import Protocol

                    class Marker(Protocol):
                        '''Members come later.'''
                """,
                "impl.py": "class Thing(object):\n    pass\n",
            },
            id="protocol-without-members",
        ),
        pytest.param(
            {
                "iface.py": """
                    from typing import Protocol

                    class Handler(Protocol):
                        def __call__(self, event: str) -> None: ...

                    class Printer:
                        def __call__(self, event: str) -> None: print(event)
                """,
            },
            id="callback-protocol",
        ),
        pytest.param(
            {
                "base.py": """
                    from abc import ABC

                    class Base(ABC):
                        def describe(self) -> str: return "base"

                    class Child(Base):
                        pass
                """,
            },
            id="abc-without-abstract-methods",
        ),
        pytest.param(
            {
                "base.py": ABC_BASE,
                "mail.py": """
                    from base import Notifier

                    class MailNotifier(Notifier):
                        def notify(self, message: str) -> None: print(message)

                    class LogNotifier:
                        def notify(self, message: str) -> None: print(message)

                    Notifier.register(LogNotifier)
                """,
            },
            id="registered-virtual-subclass",
        ),
        pytest.param(
            {
                "base.py": ABC_BASE,
                "mail.py": """
                    from base import Notifier

                    class MailNotifier(Notifier):
                        def notify(self, message: str) -> None: print(message)

                    @Notifier.register
                    class LogNotifier:
                        def notify(self, message: str) -> None: print(message)
                """,
            },
            id="register-decorator",
        ),
    ],
)
def test_other_counts_are_not_reported(lint: Lint, files: dict[str, str]) -> None:
    """Zero or two implementations, and exempt interface shapes, stay quiet."""
    run = lint(files)
    assert run.findings == []
