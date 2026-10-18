"""Format of stories.md; the texts themselves are reviewed in the pull request."""
import json
from pathlib import Path

import pytest

from museum.stories import FIELDS, NOT_SURE, WORD_LIMITS, load_stories

ROOT = Path(__file__).resolve().parents[1]
CLASSES = sorted(json.loads((ROOT / "data/wikiart_style5/subset_info.json").read_text(encoding="utf-8"))["classes"])
STORIES = load_stories()


def test_one_section_per_style_with_every_field():
    missing = set(CLASSES) - set(STORIES)
    assert not missing, f"add a '## ' section for {sorted(missing)}"
    for style in CLASSES:
        assert list(STORIES[style]) == FIELDS, f"{style}: the bold labels must be {FIELDS}"
    assert list(STORIES.get(NOT_SURE, {})) == ["Message"], f"'## {NOT_SURE}' needs a **Message:** field"


@pytest.mark.parametrize("section", [*CLASSES, NOT_SURE])
def test_text_is_written_and_short_enough(section):
    fields = STORIES[section]
    if any("TODO" in text for text in fields.values()):
        pytest.skip(f"{section}: not written yet")
    for label, text in fields.items():
        assert text, f"{section}: {label} is empty"
        n_words = len(text.split())
        assert n_words <= WORD_LIMITS.get(label, n_words), f"{section}: {label} has {n_words} words"


def test_loader_reads_multiline_fields(tmp_path):
    md = tmp_path / "stories.md"
    md.write_text("intro\n\n## Baroque\n\n**Period:** 1600s\n\n**Story:** line one\nline two\n\n## Sources\n\n- a\n")
    assert load_stories(md) == {"Baroque": {"Period": "1600s", "Story": "line one line two"}, "Sources": {}}
