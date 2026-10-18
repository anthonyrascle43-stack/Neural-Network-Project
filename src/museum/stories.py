"""Movement texts shown by the guide, read from stories.md."""
from __future__ import annotations

import re
from pathlib import Path

STORIES_MD = Path(__file__).with_name("stories.md")
FIELDS = ["Period", "Story", "What to look for", "Famous example"]
NOT_SURE = "Not sure"
WORD_LIMITS = {"Story": 120, "What to look for": 60, "Message": 40}


def load_stories(path: Path = STORIES_MD) -> dict[str, dict[str, str]]:
    """{section heading: {bold label: text}}, with each text joined onto one line."""
    parts = re.split(r"^## +(.+?)\s*$", path.read_text(encoding="utf-8"), flags=re.M)
    stories = {}
    for heading, body in zip(parts[1::2], parts[2::2], strict=True):
        fields = re.findall(r"^\*\*(.+?):\*\*(.*?)(?=^\*\*|\Z)", body, flags=re.M | re.S)
        stories[heading] = {label: " ".join(text.split()) for label, text in fields}
    return stories
