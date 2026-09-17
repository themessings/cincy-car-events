"""Tests for the collector workflow's schedule and its gate job.

The gate has now failed silently twice — once comparing the wall clock to
"06" so every delayed run was discarded, and once with a Thursday cron that
GitHub simply never dispatched. Both times the workflow looked healthy while
producing nothing, so these tests run the gate's real shell script against a
stubbed clock and git history rather than re-describing it in Python.
"""

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "car_events.yml"

# Seconds. Fixed points in time for the stubbed clock; the exact values only
# matter relative to each other.
MIDNIGHT_ET = 1_789_617_600  # 2026-09-17 00:00 EDT
THIS_MORNING = MIDNIGHT_ET + 7 * 3600
YESTERDAY = MIDNIGHT_ET - 3 * 3600


@pytest.fixture(scope="module")
def workflow():
    # PyYAML resolves the bare key `on` to the boolean True (YAML 1.1), so the
    # trigger block has to be looked up under both spellings.
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text())
    loaded["_on"] = loaded.get("on", loaded.get(True))
    return loaded


@pytest.fixture(scope="module")
def cron_hours(workflow):
    hours = []
    for entry in workflow["_on"]["schedule"]:
        minute, hour, dom, month, dow = entry["cron"].split()
        assert minute == "20", f"{entry['cron']}: :00 is GitHub's most congested slot"
        assert (dom, month, dow) == ("*", "*", "4"), f"{entry['cron']}: Thursdays only"
        hours.append(hour)
    return hours


@pytest.fixture(scope="module")
def gate_script(workflow):
    steps = workflow["jobs"]["gate"]["steps"]
    script = next(s["run"] for s in steps if s.get("id") == "check")
    return script


@pytest.fixture(scope="module")
def gate_hours(gate_script):
    """The UTC hours the gate accepts, per Eastern offset."""
    hours = {}
    for offset in ("-0400", "-0500"):
        match = re.search(rf'{offset}\)\s*TODAYS_HOURS="([^"]+)"', gate_script)
        assert match, f"gate no longer branches on {offset}"
        hours[offset] = match.group(1).split()
    return hours


def test_every_cron_is_a_slot_the_gate_accepts(cron_hours, gate_hours):
    """A cron the gate doesn't know is a run that fires and skips itself."""
    accepted = gate_hours["-0400"] + gate_hours["-0500"]
    assert sorted(cron_hours) == sorted(accepted)


def test_dst_slots_never_collide(gate_hours):
    """No UTC hour may belong to both offsets, or the gate can't tell which
    cron fired from the hour alone."""
    assert not set(gate_hours["-0400"]) & set(gate_hours["-0500"])


def test_standard_time_slots_are_the_daylight_ones_an_hour_later(gate_hours):
    """Both sets must land on the same Eastern wall-clock times."""
    assert len(gate_hours["-0400"]) == len(gate_hours["-0500"])
    for edt, est in zip(gate_hours["-0400"], gate_hours["-0500"]):
        assert int(est) == int(edt) + 1


def test_there_is_a_primary_slot_and_at_least_one_backstop(gate_hours):
    """One slot means one dropped dispatch costs the whole week."""
    assert len(gate_hours["-0400"]) >= 2


def run_gate(gate_script, tmp_path, *, event, cron, offset, last_bot):
    """Execute the gate's real script with the clock and git history stubbed.

    `last_bot` is the commit timestamp of the newest car-events-bot commit on
    the branch, or None for "no such commit in the fetched history".
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)  # one tmp_path may drive several gate runs

    real_date = shutil.which("date")
    (bin_dir / "date").write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            if [ "${{TZ:-}}" = "America/New_York" ]; then
              case "$1" in
                +%z) echo "{offset}"; exit 0 ;;
                -d)  echo "{MIDNIGHT_ET}"; exit 0 ;;
              esac
            fi
            exec {real_date} "$@"
            """
        )
    )

    real_git = shutil.which("git")
    (bin_dir / "git").write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            case "$1" in
              fetch) exit 0 ;;
              log)   echo "{'' if last_bot is None else last_bot}"; exit 0 ;;
            esac
            exec {real_git} "$@"
            """
        )
    )

    for stub in bin_dir.iterdir():
        stub.chmod(0o755)

    # The gate is a GitHub Actions step, so its expressions are interpolated
    # before bash ever sees them.
    script = gate_script.replace("${{ github.event_name }}", event)
    script = script.replace("${{ github.event.schedule }}", cron)
    script_path = tmp_path / "gate.sh"
    script_path.write_text(script)

    github_output = tmp_path / "github_output"
    github_output.write_text("")  # each run must be read on its own
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(github_output),
        "GITHUB_REF_NAME": "main",
    }
    result = subprocess.run(
        ["bash", str(script_path)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    written = github_output.read_text()
    assert written.count("proceed=") == 1, f"gate wrote {written!r}"
    return written.strip() == "proceed=true", result.stdout


@pytest.mark.parametrize(
    "case, event, cron, offset, last_bot, expected",
    [
        # The primary slot, on a day nothing has run yet.
        ("edt primary", "schedule", "20 10 * * 4", "-0400", None, True),
        ("est primary", "schedule", "20 11 * * 4", "-0500", None, True),
        # A backstop covering a primary GitHub never dispatched — the failure
        # this schedule exists for.
        ("edt backstop, primary dropped", "schedule", "20 13 * * 4", "-0400", None, True),
        ("edt late backstop, primary dropped", "schedule", "20 16 * * 4", "-0400", None, True),
        ("est backstop, primary dropped", "schedule", "20 14 * * 4", "-0500", None, True),
        # A backstop after the primary already did the work.
        ("edt backstop, already collected", "schedule", "20 13 * * 4", "-0400", THIS_MORNING, False),
        ("est backstop, already collected", "schedule", "20 14 * * 4", "-0500", THIS_MORNING, False),
        # Last week's commit is not this week's run.
        ("edt backstop, stale commit", "schedule", "20 13 * * 4", "-0400", YESTERDAY, True),
        # The other DST set's crons fire year-round and must no-op.
        ("est cron on an edt day", "schedule", "20 11 * * 4", "-0400", None, False),
        ("edt cron on an est day", "schedule", "20 10 * * 4", "-0500", None, False),
        ("est backstop on an edt day", "schedule", "20 17 * * 4", "-0400", None, False),
        # Never skip a week over an offset we don't recognise.
        ("unexpected offset", "schedule", "20 10 * * 4", "+0000", None, True),
        # Manual runs are deliberate and always proceed.
        ("manual run", "workflow_dispatch", "", "-0400", THIS_MORNING, True),
    ],
)
def test_gate_decision(case, event, cron, offset, last_bot, expected, gate_script, tmp_path):
    proceeded, log = run_gate(
        gate_script, tmp_path, event=event, cron=cron, offset=offset, last_bot=last_bot
    )
    assert proceeded is expected, f"{case}: gate said {proceeded}\n{log}"


def test_gate_explains_every_decision(gate_script, tmp_path):
    """A skip that prints nothing is how the last two failures stayed hidden."""
    for cron, last_bot in (("20 10 * * 4", None), ("20 11 * * 4", None), ("20 13 * * 4", THIS_MORNING)):
        _, log = run_gate(
            gate_script, tmp_path, event="schedule", cron=cron, offset="-0400", last_bot=last_bot
        )
        assert log.strip(), f"{cron} decided silently"
