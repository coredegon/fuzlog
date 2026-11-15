"""Run the code blocks of the documentation and check the output they print.

A page's ```python blocks run in order in one namespace. The output of a statement
that calls ``print`` is written as comments: one after the call on the same line,
or comment lines right below it.
"""

import ast
import contextlib
import io
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def comments(code):
    """Map line numbers to comment text, and note which lines hold only a comment."""
    found, alone = {}, set()
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            found[row] = token.string.removeprefix("#").strip()
            if not token.line[:col].strip():
                alone.add(row)
    return found, alone


def calls_print(node):
    return any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"
        for n in ast.walk(node)
    )


def expected(node, found, alone):
    """The comments that hold the output of a statement."""
    lines, row = [], node.end_lineno
    if row in found and row not in alone:
        lines.append(found[row])
    row += 1
    while row in alone:
        lines.append(found[row])
        row += 1
    return "\n".join(lines)


def same(printed, wanted):
    return printed.split() == wanted.split()


@pytest.fixture
def _sandbox(monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)  # examples that save do not litter the repository
    yield
    plt.close("all")


@pytest.mark.usefixtures("_sandbox")
@pytest.mark.parametrize(
    "page", PAGES, ids=[page.relative_to(ROOT).as_posix() for page in PAGES]
)
def test_examples_in_the_documentation_run(page):
    namespace = {"__name__": "__docs__"}
    for code in BLOCK.findall(page.read_text(encoding="utf-8")):
        found, alone = comments(code)
        for node in ast.parse(code).body:
            source = ast.get_source_segment(code, node)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(ast.Module([node], []), page.name, "exec"), namespace)
            if calls_print(node):
                wanted = expected(node, found, alone)
                assert same(out.getvalue(), wanted), (
                    f"{page.name}: {source}\nprinted:\n{out.getvalue()}"
                    f"documented:\n{wanted}"
                )
