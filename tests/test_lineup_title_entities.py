"""Event titles reach the lineup slide as plain text, never as HTML entities.

The 2026-11-12 slide (run 43215148239) printed "Fall Cruise-In &amp; Shopping
Extravaganza" on a public post: the collector stored the title escaped, and the
slide builder drew it as-is.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from caption_rules import decode_entities, strip_title_noise  # noqa: E402

# generate_weekly_lineup.py runs the whole build on import, so pull out just the
# sheet-cleaning functions and run them here.
_SRC = (ROOT / "scripts" / "generate_weekly_lineup.py").read_text(encoding="utf-8")
_WANT = {"coalesce_str", "normalize_spaces", "clean_title", "clean_place"}
_NS = {"re": re, "decode_entities": decode_entities, "strip_title_noise": strip_title_noise}
for _node in ast.parse(_SRC).body:
    if isinstance(_node, ast.FunctionDef) and _node.name in _WANT:
        exec(compile(ast.Module(body=[_node], type_ignores=[]), "generate_weekly_lineup.py", "exec"), _NS)
clean_title = _NS["clean_title"]
clean_place = _NS["clean_place"]


def test_the_2026_11_12_title():
    assert clean_title("Fall Cruise-In &amp; Shopping Extravaganza") == "Fall Cruise-In & Shopping Extravaganza"


def test_ampersand_and_apostrophe_escaped_once():
    assert clean_title("Joe&#39;s Cars &amp; Coffee") == "Joe's Cars & Coffee"
    assert clean_title("Joe&#x27;s &quot;Cars&quot; &amp; Coffee") == "Joe's \"Cars\" & Coffee"


def test_ampersand_and_apostrophe_escaped_twice():
    assert clean_title("Joe&amp;#39;s Cars &amp;amp; Coffee") == "Joe's Cars & Coffee"


def test_plain_text_is_untouched():
    assert clean_title("Joe's Cars & Coffee") == "Joe's Cars & Coffee"
    assert decode_entities("R&D Motors") == "R&D Motors"


def test_places_are_decoded_too():
    assert clean_place("Mike&#39;s Garage &amp; Grill, Mason") == "Mike's Garage & Grill, Mason"
