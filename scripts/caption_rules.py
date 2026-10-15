"""Brand rules the weekly caption must pass before it leaves this repo.

spottid-app's scripts/lib/brand.js refuses to publish a planner post that has an
em/en dash, two or more lines starting with an emoji, or more than 4 hashtags
(enforced since 2026-10-12). On 2026-10-15 the weekly caption broke all three and
the Thursday post was not sent. These helpers mirror those three rules so the
caption is checked here, at the source, on every run.
"""
import re

_EMOJI = "\U0001F300-\U0001FAFF☀-➿⭐⬆✅❌"
DASHES = re.compile(r"[—–]")
EMOJI_LINE = re.compile(rf"^\s*[{_EMOJI}]️?\s*\S")
HASHTAG = re.compile(r"(?:^|\s)#[0-9_]*[^\W\d_]\w*", re.UNICODE)
MAX_HASHTAGS = 4


def caption_problems(text: str):
    """Return a list of brand-rule violations (empty list = passes)."""
    out = []
    if DASHES.search(text):
        out.append("em/en dash")
    n = sum(1 for ln in text.split("\n") if EMOJI_LINE.match(ln))
    if n >= 2:
        out.append(f"{n} lines start with an emoji")
    tags = len(HASHTAG.findall(text))
    if tags > MAX_HASHTAGS:
        out.append(f"{tags} hashtags (max {MAX_HASHTAGS})")
    return out


def scrub_title(title: str) -> str:
    """Event titles come from outside; turn dashes into commas."""
    return re.sub(r"\s*[—–]\s*", ", ", title)


def scrub_caption(text: str) -> str:
    """Last-resort cleanup: dashes become commas, leading/trailing emoji are dropped."""
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    lines = []
    for ln in text.split("\n"):
        ln = re.sub(rf"^\s*[{_EMOJI}]️?\s*", "", ln)
        ln = re.sub(rf"\s*[{_EMOJI}️]+\s*$", "", ln)
        lines.append(ln)
    return "\n".join(lines)
