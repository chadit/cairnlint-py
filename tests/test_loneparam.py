"""agentloneparam through the command: calls resolved by name across files."""

import pytest

from tests.conftest import Lint

RENDER = """
def render(template, strict):
    if strict:
        return "strict:" + template
    return template
"""


TAG_WITH_KIND = 'KIND = "a"\n\ndef tag(name, kind):\n    return name\n\n'


def _format_fixed_finding(
    position: str, param: str, function: str, value: str, count: int
) -> str:
    return (
        f"{position}: [agent] parameter {param} of {function} receives {value} at all "
        + f"{count} call sites in the analyzed paths; the choice it encodes "
        + "is never made"
    )


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            {
                "render.py": RENDER,
                "pages.py": """
                    from render import render

                    PAGES = [render("page", True), render("header", True)]
                    FOOTER = render("footer", strict=True)
                """,
            },
            [_format_fixed_finding("render.py:2:22", "strict", "render", "True", 3)],
            id="constant-across-files-and-keywords",
        ),
        pytest.param(
            {
                "fetch.py": """
                    def fetch(url, retries=3, *, verbose=False):
                        return url, retries, verbose

                    fetch("a")
                    fetch("b", verbose=False)
                """,
            },
            [
                _format_fixed_finding("fetch.py:2:16", "retries", "fetch", "3", 2),
                _format_fixed_finding("fetch.py:2:30", "verbose", "fetch", "False", 2),
            ],
            id="default-reaches-every-call",
        ),
        pytest.param(
            {
                "mode.py": """
                    import enum

                    class Mode(enum.Enum):
                        STRICT = 1

                    class Renderer:
                        def draw(self, page, mode):
                            return page, mode

                    def run(renderer):
                        renderer.draw("a", Mode.STRICT)
                        renderer.draw("b", Mode.STRICT)
                """,
            },
            [_format_fixed_finding("mode.py:8:26", "mode", "draw", "Mode.STRICT", 2)],
            id="method-with-enum-member",
        ),
        pytest.param(
            {
                "scale.py": """
                    def scale(value, factor):
                        return value * factor
                """,
                "tests/test_scale.py": """
                    from scale import scale

                    def test_scale():
                        assert scale(2, -1) == -2
                        assert scale(3, -1) == -3
                """,
            },
            [_format_fixed_finding("scale.py:2:18", "factor", "scale", "-1", 2)],
            id="test-call-sites-count",
        ),
        pytest.param(
            {
                "pick.py": "def pick(_, flag):\n    return flag\n\n"
                + "pick(1, True)\npick(1, True)\n",
            },
            [_format_fixed_finding("pick.py:1:13", "flag", "pick", "True", 2)],
            id="blank-parameter-left-out",
        ),
        pytest.param(
            {
                "scale.py": "FACTOR = 2\n\ndef scale(value, factor):\n"
                + "    return value * factor\n\n"
                + "scale(1, FACTOR)\nscale(2, FACTOR)\n",
            },
            [_format_fixed_finding("scale.py:3:18", "factor", "scale", "FACTOR", 2)],
            id="module-level-upper-case-constant",
        ),
        pytest.param(
            {
                "consts.py": 'KIND = "shared"\n',
                "lib.py": "def tag(name, kind):\n    return name\n",
                "a.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "b.py": "from consts import KIND as K\nfrom lib import tag\n"
                + 'tag("y", K)\n',
            },
            [_format_fixed_finding("lib.py:1:15", "kind", "tag", "KIND", 2)],
            id="one-binding-spelled-two-ways",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "a.py": "import logging\nfrom lib import tag\n"
                + 'tag("x", logging.INFO)\n',
                "b.py": "from logging import INFO\nfrom lib import tag\n"
                + 'tag("y", INFO)\n',
            },
            [_format_fixed_finding("lib.py:1:15", "kind", "tag", "logging.INFO", 2)],
            id="outside-constant-spelled-two-ways",
        ),
        pytest.param(
            {
                "consts.py": 'KIND = "shared"\n',
                "lib.py": "def tag(name, kind):\n    return name\n",
                "a.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "b.py": "import consts\nfrom lib import tag\n"
                + 'tag("y", consts.KIND)\n',
            },
            [_format_fixed_finding("lib.py:1:15", "kind", "tag", "KIND", 2)],
            id="module-attribute-and-from-import",
        ),
        pytest.param(
            {
                "pkg/__init__.py": "",
                "pkg/consts.py": 'KIND = "shared"\n',
                "pkg/lib.py": "def tag(name, kind):\n    return name\n",
                "pkg/a.py": "from pkg import consts\nfrom pkg.lib import tag\n"
                + 'tag("x", consts.KIND)\n',
                "pkg/b.py": "from . import consts\nfrom pkg.lib import tag\n"
                + 'tag("y", consts.KIND)\n',
            },
            [_format_fixed_finding("pkg/lib.py:1:15", "kind", "tag", "consts.KIND", 2)],
            id="submodule-imported-from-its-package",
        ),
        pytest.param(
            {
                "emit.py": "def emit(*, level=1):\n    return level\n\n"
                + "emit()\nemit(level=1)\n",
            },
            [_format_fixed_finding("emit.py:1:13", "level", "emit", "1", 2)],
            id="keyword-only-callee",
        ),
        pytest.param(
            {
                "t.py": TAG_WITH_KIND
                + "def run(xs):\n    return [tag(x, KIND) for x in xs]\n\n"
                + "def walk(ys):\n    return [tag(y, KIND) for y in ys]\n",
            },
            [_format_fixed_finding("t.py:3:15", "kind", "tag", "KIND", 2)],
            id="constant-read-inside-comprehensions",
        ),
        pytest.param(
            {
                "g.py": TAG_WITH_KIND
                + 'def run[T](x: T) -> T:\n    tag("x", KIND)\n'
                + '    tag("y", KIND)\n    return x\n',
            },
            [_format_fixed_finding("g.py:3:15", "kind", "tag", "KIND", 2)],
            id="calls-inside-a-generic-function",
        ),
        pytest.param(
            {
                "app/__init__.py": "",
                "app/logging.py": "def setup():\n    return None\n",
                "app/lib.py": "def emit(text, level):\n    return text\n",
                "app/a.py": "import logging\nfrom app.lib import emit\n"
                + 'emit("x", logging.INFO)\nemit("y", logging.INFO)\n',
            },
            [
                _format_fixed_finding(
                    "app/lib.py:1:16", "level", "emit", "logging.INFO", 2
                )
            ],
            id="standard-library-name-shared-by-a-package-module",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "proj/__init__.py": "",
                "proj/celery.py": "APP = None\n",
                "a.py": "import celery\nfrom lib import tag\n"
                + 'tag("x", celery.STATES)\n',
                "b.py": "import celery\nfrom lib import tag\n"
                + 'tag("y", celery.STATES)\n',
            },
            [_format_fixed_finding("lib.py:1:15", "kind", "tag", "celery.STATES", 2)],
            id="outside-name-shared-by-a-file-in-another-package",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "consts.py": 'KIND = "root"\n',
                "a.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "sub/b.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("y", KIND)\n',
            },
            [_format_fixed_finding("lib.py:1:15", "kind", "tag", "KIND", 2)],
            id="importer-beside-the-sole-match-joins-one-elsewhere",
        ),
        pytest.param(
            {
                "consts.py": 'KIND = "shared"\n',
                "lib.py": "def tag(name, kind):\n    return name\n",
                "a.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "b.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("y", KIND)\n',
            },
            [_format_fixed_finding("lib.py:1:15", "kind", "tag", "KIND", 2)],
            id="constant-imported-from-a-shared-module",
        ),
        pytest.param(
            {
                "pkg/__init__.py": "",
                "pkg/consts.py": 'KIND = "shared"\n',
                "pkg/lib.py": "def tag(name, kind):\n    return name\n",
                "pkg/a.py": "from .consts import KIND\nfrom .lib import tag\n"
                + 'tag("x", KIND)\n',
                "pkg/b.py": "from pkg.consts import KIND\nfrom pkg.lib import tag\n"
                + 'tag("y", KIND)\n',
            },
            [_format_fixed_finding("pkg/lib.py:1:15", "kind", "tag", "KIND", 2)],
            id="constant-imported-relatively-and-absolutely",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "a.py": "import logging\nfrom lib import tag\n"
                + 'tag("x", logging.INFO)\n',
                "b.py": "import logging\nfrom lib import tag\n"
                + 'tag("y", logging.INFO)\n',
            },
            [_format_fixed_finding("lib.py:1:15", "kind", "tag", "logging.INFO", 2)],
            id="constant-from-a-module-outside-the-tree",
        ),
    ],
)
def test_fixed_parameter_is_reported(
    lint: Lint, files: dict[str, str], expected: list[str]
) -> None:
    """A parameter given one constant at two or more call sites is reported."""
    run = lint(files)
    assert run.findings == expected


@pytest.mark.parametrize(
    "calls",
    [
        pytest.param('render("page", True)\nrender("header", False)', id="varies"),
        pytest.param('render("page", True)', id="single-call-site"),
        pytest.param(
            'render("page", True)\nrender("header", True)\nhandlers = [render]',
            id="used-as-a-value",
        ),
        pytest.param(
            'render("page", True)\nrender(*["header"], True)',
            id="unpacked-arguments",
        ),
        pytest.param(
            'render("page", True)\nrender("header", True, "extra")',
            id="call-the-signature-rejects",
        ),
        pytest.param(
            'render("page", True)\nrender("header", flag)',
            id="non-constant-argument",
        ),
        pytest.param(
            'render("page", True)\nrender("header", True)\n__all__ = ["render"]',
            id="exported-in-all",
        ),
    ],
)
def test_unproven_parameter_is_not_reported(lint: Lint, calls: str) -> None:
    """Variation, too few calls, or calls the name cannot be pinned on stay quiet."""
    run = lint({"render.py": f"{RENDER}{calls}\n"})
    assert run.findings == []


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(
            {
                "a.py": "class A:\n    def emit(self, level): return level\n",
                "b.py": "class B:\n    def emit(self, level): return level\n",
                "run.py": "def run(x):\n    x.emit(1)\n    x.emit(1)\n",
            },
            id="name-declared-twice",
        ),
        pytest.param(
            {
                "store.py": "class Store:\n    def get(self, key): return key\n",
                "run.py": "def run(x):\n    x.get(1)\n    x.get(1)\n",
            },
            id="builtin-method-name",
        ),
        pytest.param(
            {
                "store.py": "class Store:\n    def put(self, key): return key\n",
                "run.py": "def run():\n    put(1)\n    put(1)\n",
            },
            id="bare-call-to-a-method-name",
        ),
        pytest.param(
            {
                "cli.py": "import app\n\n@app.command\n"
                + "def serve(port): return port\n\nserve(80)\nserve(80)\n",
            },
            id="decorated-function",
        ),
        pytest.param(
            {
                "tests/helpers.py": "def build(size): return size\n",
                "tests/test_build.py": "from helpers import build\n"
                + "build(1)\nbuild(1)\n",
            },
            id="declared-in-test-code",
        ),
        pytest.param(
            {
                "lib.py": RENDER,
                "use.py": "import lib\nfrom lib import render as draw\n\n"
                + 'lib.render("a", True)\nlib.render("b", True)\n'
                + 'draw("c", False)\n',
            },
            id="call-through-aliased-import",
        ),
        pytest.param(
            {
                "pkg/lib.py": RENDER,
                "pkg/__init__.py": "from pkg.lib import render\n\n"
                + '__all__ = ["render"]\n',
                "pkg/use.py": 'from pkg import render\nrender("a", True)\n'
                + 'render("b", True)\n',
            },
            id="re-exported-by-package-init",
        ),
        pytest.param(
            {
                "scale.py": "def scale(value, factor):\n    return value * factor\n\n"
                + "def run(N):\n    scale(1, N)\n    scale(2, N)\n",
            },
            id="upper-case-parameter-of-the-caller",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "a.py": 'from lib import tag\nKIND = "a"\ntag("x", KIND)\n',
                "b.py": 'from lib import tag\nKIND = "b"\ntag("y", KIND)\n',
            },
            id="same-name-bound-in-two-modules",
        ),
        pytest.param(
            {
                "net.py": "TIMEOUT = 5\n\ndef fetch(url, timeout):\n"
                + "    return url, timeout\n\ndef slow(url):\n    TIMEOUT = 30\n"
                + "    return fetch(url, TIMEOUT)\n\ndef fast(url):\n"
                + "    return fetch(url, TIMEOUT)\n",
            },
            id="local-shadows-module-constant",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "one/shared/consts.py": 'KIND = "one"\n',
                "one/use.py": "import shared.consts\nfrom lib import tag\n"
                + 'tag("x", shared.consts.KIND)\n',
                "two/shared/consts.py": 'KIND = "two"\n',
                "two/use.py": "import shared.consts\nfrom lib import tag\n"
                + 'tag("y", shared.consts.KIND)\n',
            },
            id="two-namespace-packages-with-the-same-name",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "one/consts.py": 'KIND = "one"\n',
                "one/use.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "two/consts.py": 'KIND = "two"\n',
                "two/use.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("y", KIND)\n',
            },
            id="two-modules-matching-one-import",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "a.py": "from b import KIND\n",
                "b.py": "from a import KIND\n",
                "c.py": "from a import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "d.py": "from a import KIND\nfrom lib import tag\n"
                + 'tag("y", KIND)\n',
            },
            id="import-cycle",
        ),
        pytest.param(
            {
                "t.py": TAG_WITH_KIND
                + 'tag("x", KIND)\nitems = [tag("y", KIND) for KIND in ("b", "c")]\n',
            },
            id="comprehension-target-shadows-constant",
        ),
        pytest.param(
            {
                "t.py": TAG_WITH_KIND
                + 'def run():\n    tag("x", KIND)\n'
                + '    return [tag("y", KIND) for KIND in ("b", "c")]\n',
            },
            id="comprehension-target-shadows-constant-in-a-function",
        ),
        pytest.param(
            {
                "r.py": 'MODE = "module"\n\nclass Renderer:\n    MODE = "class"\n\n'
                + "    def draw(self, page, mode=MODE):\n        return page, mode\n\n"
                + 'def run(r):\n    r.draw("a")\n    r.draw("b", MODE)\n',
            },
            id="method-default-reads-a-class-attribute",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "tools/__init__.py": "",
                "tools/a/__init__.py": "",
                "tools/a/consts.py": 'KIND = "a"\n',
                "tools/a/run.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "tools/b/__init__.py": "",
                "tools/b/consts.py": 'KIND = "b"\n',
                "tools/b/run.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("y", KIND)\n',
            },
            id="bare-imports-of-two-same-named-package-files",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "tools/__init__.py": "",
                "tools/a/__init__.py": "",
                "tools/a/secrets.py": 'KIND = "a"\n',
                "tools/a/run.py": "from secrets import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "tools/b/__init__.py": "",
                "tools/b/secrets.py": 'KIND = "b"\n',
                "tools/b/run.py": "from secrets import KIND\nfrom lib import tag\n"
                + 'tag("y", KIND)\n',
            },
            id="standard-library-name-shadowed-in-two-directories",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "a/__init__.py": "",
                "a/logging.py": "def setup():\n    return None\n",
                "a/run.py": "import logging\nfrom lib import tag\n"
                + 'tag("x", logging.INFO)\n',
                "b.py": "import logging\nfrom lib import tag\n"
                + 'tag("y", logging.INFO)\n',
            },
            id="importer-beside-a-same-named-file-stays-apart",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "tools/__init__.py": "",
                "tools/a/__init__.py": "",
                "tools/b/__init__.py": "",
                "tools/a/consts/__init__.py": 'KIND = "a"\n',
                "tools/a/run.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "tools/b/consts/__init__.py": 'KIND = "b"\n',
                "tools/b/run.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("y", KIND)\n',
            },
            id="package-beside-each-importer",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "tools/__init__.py": "",
                "tools/a/__init__.py": "",
                "tools/b/__init__.py": "",
                "tools/a/sub/consts.py": 'KIND = "a"\n',
                "tools/a/run.py": "import sub.consts\nfrom lib import tag\n"
                + 'tag("x", sub.consts.KIND)\n',
                "tools/b/sub/consts.py": 'KIND = "b"\n',
                "tools/b/run.py": "import sub.consts\nfrom lib import tag\n"
                + 'tag("y", sub.consts.KIND)\n',
            },
            id="namespace-directory-beside-each-importer",
        ),
        pytest.param(
            {
                "lib.py": "def tag(name, kind):\n    return name\n",
                "tools/__init__.py": "",
                "tools/a/__init__.py": "",
                "tools/b/__init__.py": "",
                "consts.py": 'KIND = "root"\n',
                "tools/a/consts.py": 'KIND = "a"\n',
                "tools/a/run.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("x", KIND)\n',
                "tools/b/consts.py": 'KIND = "b"\n',
                "tools/b/run.py": "from consts import KIND\nfrom lib import tag\n"
                + 'tag("y", KIND)\n',
            },
            id="neighbour-and-exact-match-disagree",
        ),
        pytest.param(
            {
                "gather.py": "def gather(first, *rest):\n    return first\n\n"
                + "gather(1)\ngather(1)\n",
            },
            id="varargs-callee",
        ),
        pytest.param(
            {
                "box.py": "class Box:\n    def __pack__(self, mode):\n"
                + "        return mode\n\n"
                + "def run(box):\n    box.__pack__(1)\n    box.__pack__(1)\n",
            },
            id="dunder-method",
        ),
        pytest.param(
            {
                "fetch.py": "def fetch(retries=3, **extra):\n    return retries\n\n"
                + "fetch()\nfetch()\nfetch(**OPTIONS)\n",
            },
            id="call-unpacking-keywords",
        ),
        pytest.param(
            {
                "tag.py": "def tag(name, strict):\n    return name\n\n"
                + 'tag("a", True)\ntag("b", True)\ntag("c", True, colour=1)\n',
            },
            id="keyword-the-signature-lacks",
        ),
    ],
)
def test_ambiguous_or_exempt_callee_is_not_reported(
    lint: Lint, files: dict[str, str]
) -> None:
    """Names that resolve to more than one thing, or exempt callees, are skipped."""
    run = lint(files)
    assert run.findings == []


@pytest.mark.parametrize(
    "second_binding",
    [
        pytest.param("def other(KIND):\n    return KIND\n", id="parameter"),
        pytest.param(
            "def guard():\n    try:\n        pass\n"
            + "    except ValueError as KIND:\n        return KIND\n",
            id="except-handler",
        ),
        pytest.param(
            "def first(x):\n    match x:\n        case [KIND]:\n"
            + "            return KIND\n",
            id="match-capture",
        ),
        pytest.param(
            "def rest(x):\n    match x:\n        case [*KIND]:\n"
            + "            return KIND\n",
            id="match-star",
        ),
        pytest.param(
            "def extra(x):\n    match x:\n        case {**KIND}:\n"
            + "            return KIND\n",
            id="match-mapping-rest",
        ),
        pytest.param(
            "def ident[KIND](x: KIND) -> KIND:\n    return x\n", id="type-variable"
        ),
        pytest.param(
            "def wrap[**KIND](x: int) -> int:\n    return x\n", id="param-spec"
        ),
        pytest.param(
            "def pack[*KIND](x: int) -> int:\n    return x\n",
            id="type-variable-tuple",
        ),
        pytest.param(
            "class Holder:\n    def KIND(self):\n        return 1\n",
            id="method-definition",
        ),
    ],
)
def test_second_binding_of_a_constant_name_unresolves_it(
    lint: Lint, second_binding: str
) -> None:
    """Any other binding of the name in the file stops it counting as the constant."""
    source = f'{TAG_WITH_KIND}{second_binding}\ntag("x", KIND)\ntag("y", KIND)\n'
    run = lint({"t.py": source})
    assert run.findings == []
