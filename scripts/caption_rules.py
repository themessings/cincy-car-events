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


def strip_title_noise(title: str) -> str:
    """Slide titles: the slide font has no emoji, so each one printed as an empty
    box (2026-10-22: "Indianapolis Cars & Coffee [Trunk-or-Treat] <4 emoji> - Hosted by").
    Drop the emoji, and a "- Hosted by ..." tail: the host is tagged in the caption, and
    the full title ("... - Hosted by Model 1 and Lakefront at Keystone") was cut to "-..."."""
    t = re.sub(rf"[{_EMOJI}\u200d\ufe0f]", "", title)
    t = re.sub(r"\s*[-,:|]\s*hosted by\b.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+hosted by\s*$", "", t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip()


KIND_WORDS = (
    ("Cars & coffee", re.compile(r"\bcoffee\b", re.IGNORECASE)),
    ("cruise-ins", re.compile(r"\bcruise", re.IGNORECASE)),
    ("full shows", re.compile(r"\bshows?\b|\bconcours\b", re.IGNORECASE)),
)


def _and_join(items):
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def what_and_when(titles, event_days) -> str:
    """The caption's "what and when" sentence, built from what the slides really carry.

    It was fixed text: "Cars & coffee, cruise-ins, and full shows from Friday morning
    through Thursday night." The 2026-10-22 repost had nine events, all on Saturday and
    Sunday and none of them a cruise-in, and the same caption went on to say "Friday,
    Monday, Tuesday, Wednesday, and Thursday? Nothing posted." Now only the kinds in the
    event titles and the days that have events are named; a full week reads as before.
    `event_days` are the dates with at least one event. No days gives an empty string."""
    days = sorted(set(event_days))
    if not days:
        return ""
    kinds = [word for word, rx in KIND_WORDS if any(rx.search(t or "") for t in titles)]
    what = _and_join(kinds) if kinds else "Meets"
    what = what[0].upper() + what[1:]
    names = [d.strftime("%A") for d in days]
    if len(names) == 1:
        when = f"on {names[0]}"
    elif len(names) == 2:
        when = f"on {names[0]} and {names[1]}"
    else:
        when = f"from {names[0]} morning through {names[-1]} night"
    return f"{what} {when}."


def scrub_caption(text: str) -> str:
    """Last-resort cleanup: dashes become commas, leading/trailing emoji are dropped."""
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    lines = []
    for ln in text.split("\n"):
        ln = re.sub(rf"^\s*[{_EMOJI}]️?\s*", "", ln)
        ln = re.sub(rf"\s*[{_EMOJI}️]+\s*$", "", ln)
        lines.append(ln)
    return "\n".join(lines)
