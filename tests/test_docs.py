import doctest
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


@pytest.mark.parametrize(
    "page", PAGES, ids=[page.relative_to(ROOT).as_posix() for page in PAGES]
)
def test_examples_in_the_documentation_run(page):
    text = page.read_text(encoding="utf-8").replace("```", "")  # a fence is no output
    examples = doctest.DocTestParser().get_doctest(text, {}, page.name, str(page), 0)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    runner.run(examples)
    assert runner.summarize().failed == 0
