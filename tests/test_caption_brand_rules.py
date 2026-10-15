"""The weekly caption must pass spottid-app's brand check (no dashes, <2 emoji-led lines, <=4 hashtags)."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from caption_rules import caption_problems, scrub_caption, scrub_title  # noqa: E402

BAD = "Six markets. One weekend. A, B — all loaded.\n\U0001F525 FRIDAY IS THE ONE \U0001F525\n\U0001F4F1 Get it\n#a #b #c #d #e"


def test_detects_the_2026_10_15_failures():
    p = " ".join(caption_problems(BAD))
    assert "dash" in p and "emoji" in p and "hashtags" in p


def test_scrub_makes_it_pass():
    assert caption_problems(scrub_caption(BAD.replace("#e", ""))) == []
    assert scrub_title("Columbus Cars & Coffee – Starbucks") == "Columbus Cars & Coffee, Starbucks"


def test_generator_source_is_clean():
    src = (ROOT / "scripts" / "generate_weekly_lineup.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "make_caption")
    body = ast.get_source_segment(src, fn)
    assert "—" not in body and "–" not in body
    for ch in "\U0001F525\U0001F4F1⚠":
        assert ch not in body
    tags = re.findall(r"#\w+", re.search(r"^BASE_HASHTAGS = \[(.*?)\]", src, re.M).group(1))
    assert len(tags) <= 4
