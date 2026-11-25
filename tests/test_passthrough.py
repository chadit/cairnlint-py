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
            class Child(Base):
                async def load(self, key):
                    return await super().load(key)
            """,
            [_format_forwards_finding("mod.py:3:15", "Child.load", "super().load")],
            id="async-super-forward-without-a-default",
        ),
        pytest.param(
            """
            class Store:
                def __init__(self, inner, other):
                    self._inner = inner
                    self._other = other

                def render(self, page="home"):
                    return self._inner.render(page)
            """,
            [
                _format_forwards_finding(
                    "mod.py:7:9", "Store.render", "self._inner.render"
                )
            ],
            id="field-forward-declaring-a-default",
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
                "mod.py:4:7: [agent] class Wrapper wraps one field and its only "
                + "method forwards; callers can hold the wrapped value directly"
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
        pytest.param(
            """
            class Inner:
                def join(self, parts):
                    return ",".join(parts)


            class Wrapper:
                def __init__(self, inner):
                    self._inner = inner

                def join(self, parts):
                    return self._inner.join(parts)
            """,
            [
                "mod.py:7:7: [agent] class Wrapper wraps one field and its only "
                + "method forwards; callers can hold the wrapped value directly"
            ],
            id="wrapper-around-a-class-that-joins-on-a-literal",
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
            class Joiner:
                def join(self, parts):
                    return ",".join(parts)

                def fill(self, items):
                    return [].extend(items)
            """,
            id="method-on-a-literal-receiver",
        ),
        pytest.param(
            """
            class Joiner:
                def first(self, parts):
                    return ",;"[0].join(parts)

                def trimmed(self, parts):
                    return ", ".strip().join(parts)
            """,
            id="method-on-an-indexed-or-called-literal",
        ),
        pytest.param(
            """
            class Base:
                def render(self, page="home"):
                    return page


            class Child(Base):
                def render(self, page="home"):
                    return super().render(page)
            """,
            id="super-forward-repeating-the-parent-default",
        ),
        pytest.param(
            """
            from views import Base


            class Child(Base):
                def render(self, page, *, strict=True):
                    return super().render(page, strict=strict)
            """,
            id="super-forward-declaring-a-keyword-only-default",
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


BROKER = """
from typing import Protocol


class Broker(Protocol):
    def publish(self, topic: str, payload: bytes) -> None: ...

    def subscribe(self, topic: str) -> None: ...

    def close(self) -> None: ...
"""

KAFKA = """
class KafkaBroker:
    def __init__(self, producer, consumer):
        self.producer = producer
        self.consumer = consumer

    def publish(self, topic, payload):
        self.producer.send(topic, payload)
        self.producer.flush()

    def subscribe(self, topic):
        self.consumer.subscribe([topic])

    def close(self):
        return self.producer.stop()
"""

NATS = """
class NatsBroker:
    def __init__(self, client):
        self.client = client
        self.subjects = []

    def publish(self, topic, payload):
        self.client.publish(topic, payload)
        self.subjects.append(topic)

    def subscribe(self, topic):
        self.subjects.append(topic)

    def close(self):
        return self.client.drain()
"""


ABC_BASE = """
from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def close(self) -> None: ...

    def flush(self):
        return None
"""

ABC_KAFKA = """
from abc_base import Base


class Kafka(Base):
    def __init__(self, producer, other):
        self.producer = producer
        self.other = other

    def close(self):
        return self.producer.stop()
"""

DIAMOND_BASE = """
from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def a(self): ...


class Left(Base):
    @abstractmethod
    def l(self): ...


class Right(Base):
    @abstractmethod
    def r(self): ...
"""

DIAMOND_IMPL = """
from base import Left, Right


class Leaf(Left, Right):
    def __init__(self, inner, other):
        self.inner = inner
        self.other = other

    def a(self):
        return self.inner.a()

    def l(self):
        return self.inner.l()

    def r(self):
        return self.inner.r()


class Sub(Leaf):
    def l(self):
        return self.inner.l2()

    def r(self):
        return self.inner.r2()
"""

ABC_MID = """
from abc import abstractmethod

from abc_base import Base


class Mid(Base):
    @abstractmethod
    def send(self, message) -> None: ...

    def ping(self):
        return None
"""

ABC_MID_KAFKA = (
    ABC_KAFKA.replace("abc_base import Base", "mid import Mid").replace(
        "Kafka(Base)", "Kafka(Mid)"
    )
    + "\n    def send(self, message):\n        return self.producer.send(message)\n"
)

ABC_NATS = """
from abc_base import Base


class Nats(Base):
    def close(self):
        self.closed = True
"""

ABC_MID_NATS = (
    ABC_NATS.replace("abc_base import Base", "mid import Mid").replace(
        "Nats(Base)", "Nats(Mid)"
    )
    + "\n    def send(self, message):\n        self.sent = message\n"
)


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            {"broker.py": BROKER, "kafka.py": KAFKA, "nats.py": NATS},
            [],
            id="classes-that-satisfy-a-protocol-in-another-file",
        ),
        pytest.param(
            {
                "broker.py": BROKER,
                "adapter.py": "from broker import Broker\n\n\n"
                + "class Adapter(Broker):\n    def __init__(self, inner):\n"
                + "        self.inner = inner\n\n    def close(self):\n"
                + "        return self.inner.stop()\n",
                "kafka.py": KAFKA,
            },
            [],
            id="class-that-names-the-protocol-as-a-base",
        ),
        pytest.param(
            {
                "broker.py": BROKER,
                "partial.py": "\nclass PartialBroker:\n"
                + "    def __init__(self, producer):\n"
                + "        self.producer = producer\n\n"
                + "    def publish(self, topic, payload):\n"
                + "        self.producer.send(topic, payload)\n"
                + "        self.producer.flush()\n\n"
                + "    def close(self):\n        return self.producer.stop()\n",
                "kafka.py": KAFKA,
                "nats.py": NATS,
            },
            [
                _format_forwards_finding(
                    "partial.py:10:9", "PartialBroker.close", "self.producer.stop"
                )
            ],
            id="class-that-defines-only-some-protocol-methods",
        ),
        pytest.param(
            {
                "broker.py": BROKER,
                "kafka.py": KAFKA
                + "\n    def flush(self):\n        return self.producer.flush()\n",
                "nats.py": NATS,
            },
            [
                _format_forwards_finding(
                    "kafka.py:17:9", "KafkaBroker.flush", "self.producer.flush"
                )
            ],
            id="forward-whose-name-no-protocol-declares",
        ),
        pytest.param(
            {
                "impl.py": "class Kafka:\n    def __init__(self, producer, other):\n"
                + "        self.producer = producer\n        self.other = other\n\n"
                + "    def close(self):\n        return self.producer.stop()\n",
                "tests/test_kafka.py": "from typing import Protocol\n\n\n"
                + "class Closer(Protocol):\n    def close(self) -> None: ...\n",
            },
            [
                _format_forwards_finding(
                    "impl.py:6:9", "Kafka.close", "self.producer.stop"
                )
            ],
            id="protocol-declared-only-in-test-code",
        ),
        pytest.param(
            {"abc_base.py": ABC_BASE, "kafka.py": ABC_KAFKA, "nats.py": ABC_NATS},
            [],
            id="subclass-forwarding-an-abstract-method",
        ),
        pytest.param(
            {
                "abc_base.py": ABC_BASE,
                "mid.py": "from abc_base import Base\n\n\nclass Mid(Base):\n"
                + "    pass\n",
                "kafka.py": ABC_KAFKA.replace(
                    "abc_base import Base", "mid import Mid"
                ).replace("Kafka(Base)", "Kafka(Mid)"),
            },
            [],
            id="abstract-method-reached-through-an-intermediate-base",
        ),
        pytest.param(
            {
                "abc_base.py": ABC_BASE,
                "mid.py": ABC_MID,
                "kafka.py": ABC_MID_KAFKA,
                "nats.py": ABC_MID_NATS,
            },
            [],
            id="abstract-method-an-intermediate-base-adds",
        ),
        pytest.param(
            {
                "abc_base.py": ABC_BASE,
                "mid.py": ABC_MID,
                "kafka.py": ABC_MID_KAFKA
                + "\n    def ping(self):\n        return self.producer.ping()\n",
                "nats.py": ABC_MID_NATS,
            },
            [
                _format_forwards_finding(
                    "kafka.py:16:9", "Kafka.ping", "self.producer.ping"
                )
            ],
            id="override-of-a-concrete-intermediate-method",
        ),
        pytest.param(
            {"base.py": DIAMOND_BASE, "impl.py": DIAMOND_IMPL},
            [],
            id="abstract-methods-of-both-branches-of-a-diamond",
        ),
        pytest.param(
            {
                "abc_base.py": ABC_BASE,
                "kafka.py": ABC_KAFKA
                + "\n    def flush(self):\n        return self.producer.flush()\n",
                "nats.py": ABC_NATS,
            },
            [
                _format_forwards_finding(
                    "kafka.py:13:9", "Kafka.flush", "self.producer.flush"
                )
            ],
            id="override-of-a-concrete-base-method",
        ),
        pytest.param(
            {
                "abc_base.py": ABC_BASE,
                "kafka.py": ABC_KAFKA
                + "\n    def drain(self):\n        return self.producer.drain()\n",
                "nats.py": ABC_NATS,
            },
            [
                _format_forwards_finding(
                    "kafka.py:13:9", "Kafka.drain", "self.producer.drain"
                )
            ],
            id="forward-whose-name-the-base-does-not-declare",
        ),
        pytest.param(
            {
                "tests/test_base.py": ABC_BASE,
                "impl.py": "class Kafka(Base):\n"
                + "    def __init__(self, producer, other):\n"
                + "        self.producer = producer\n        self.other = other\n\n"
                + "    def close(self):\n        return self.producer.stop()\n",
            },
            [
                _format_forwards_finding(
                    "impl.py:6:9", "Kafka.close", "self.producer.stop"
                )
            ],
            id="abstract-base-declared-only-in-test-code",
        ),
    ],
)
def test_interface_required_method_is_not_a_forward(
    lint: Lint, files: dict[str, str], expected: list[str]
) -> None:
    """A method a Protocol or abstract base needs stays, so it is not a forward."""
    run = lint(files)
    assert run.findings == expected
