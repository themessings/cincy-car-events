"""The weekly caption must pass spottid-app's brand check (no dashes, <2 emoji-led lines, <=4 hashtags)."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from caption_rules import caption_problems, scrub_caption, scrub_title, strip_title_noise  # noqa: E402

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


def test_slide_titles_drop_emoji_and_a_cut_off_host():
    # 2026-10-22: the slide printed four empty boxes and "-..." for this title.
    t = "Indianapolis Cars & Coffee [Trunk-or-Treat] \U0001F47B\U0001F383\U0001F342\U0001F697 - Hosted by"
    assert strip_title_noise(t) == "Indianapolis Cars & Coffee [Trunk-or-Treat]"
    assert strip_title_noise("Fall 2026 Rennsport Dragon Rally\u00ae") == "Fall 2026 Rennsport Dragon Rally\u00ae"
    assert strip_title_noise("No Limits 937\u2122 Cars & Coffee") == "No Limits 937\u2122 Cars & Coffee"
    assert strip_title_noise("Cruise In Hosted by Tri-State Mustang") == "Cruise In Hosted by Tri-State Mustang"


def test_generator_cleans_slide_titles():
    src = (ROOT / "scripts" / "generate_weekly_lineup.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "clean_title")
    assert "strip_title_noise(" in ast.get_source_segment(src, fn)


def test_slide_titles_drop_a_hosted_by_tail():
    t = "Indianapolis Cars & Coffee [Trunk-or-Treat] \U0001F47B\U0001F383\U0001F342\U0001F697 - Hosted by Model 1 and Lakefront at Keystone"
    assert strip_title_noise(t) == "Indianapolis Cars & Coffee [Trunk-or-Treat]"


def test_fonts_are_picked_by_exact_name_and_never_slanted(tmp_path):
    src = (ROOT / "scripts" / "generate_weekly_lineup.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "find_font_file")
    ns = {"os": __import__("os"), "List": __import__("typing").List, "Optional": __import__("typing").Optional, "FONT_ROOTS": []}
    exec(ast.get_source_segment(src, fn), ns)
    d = tmp_path / "dejavu"
    d.mkdir()
    for name in ("DejaVuSans-BoldOblique.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        (d / name).write_bytes(b"")
    bold = ns["find_font_file"](["dejavusans-bold.ttf", "liberationsans-bold.ttf"], roots=[str(tmp_path)])
    reg = ns["find_font_file"](["dejavusans.ttf", "liberationsans-regular.ttf"], roots=[str(tmp_path)])
    assert bold.endswith("DejaVuSans-Bold.ttf") and reg.endswith("DejaVuSans.ttf")


def test_one_slide_week_does_not_say_swipe():
    src = (ROOT / "scripts" / "generate_weekly_lineup.py").read_text(encoding="utf-8")
    i = src.index("caption = make_caption(")
    tail = src[i:i + 400]
    assert "if len(slide_paths) == 1:" in tail and '"See the full regional lineup"' in tail
    assert "Swipe for the full regional lineup" in src  # the multi-slide wording is unchanged
