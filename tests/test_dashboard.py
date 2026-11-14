"""The dashboard shows every run graded by today's graders, or says why not.

What matters here is not the page layout but what the page claims: that a run is
graded by the current graders only when its replay covered the recording
completely, that the meters shown are the recorded ones, and that model output
cannot break out of the page.
"""

from __future__ import annotations

import json

import pytest
from test_replay import CASE, SCRIPT

from proofbench.client import shopdesk_client
from proofbench.dashboard import collect, render
from proofbench.providers import ScriptedProvider
from proofbench.report import summarise
from proofbench.runner import Runner


def write_run(folder, script, meters=None):
    client = shopdesk_client()
    runner = Runner(client=client, provider=ScriptedProvider(list(script)), repeats=1)
    report = runner.run_suite(type("S", (), {"name": "support", "cases": [CASE]})())
    folder.mkdir(parents=True)
    report.write(folder / "artifact.json")
    summary = summarise(report)
    if meters:
        summary.update(meters)
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    client.close()


@pytest.fixture
def suite_dir(tmp_path):
    import yaml

    folder = tmp_path / "suite"
    folder.mkdir()
    (folder / "cases.yaml").write_text(yaml.safe_dump({
        "suite": "support", "cases": [json.loads(CASE.model_dump_json())],
    }), encoding="utf-8")
    return folder


def test_a_run_that_replays_whole_is_shown_with_current_grades(tmp_path, suite_dir):
    write_run(tmp_path / "eval" / "good", SCRIPT)
    data = collect(tmp_path / "eval", suite_dir)
    run = data["runs"][0]
    assert run["graded"] == "current"
    assert "100% coverage" in run["grading_note"]
    assert run["pass"]["passed"] == 1


def test_a_run_that_does_not_replay_whole_keeps_its_recorded_grades_and_says_so(
    tmp_path, suite_dir
):
    """A replay that half matched is a measurement of something else, so none of
    it is used - the recorded verdicts are shown, flagged with why."""
    folder = tmp_path / "eval" / "drifted"
    write_run(folder, SCRIPT)
    artifact = json.loads((folder / "artifact.json").read_text(encoding="utf-8"))
    for step in artifact["attempts"][0]["run"]["steps"]:
        if step.get("request_hash"):
            step["request_hash"] = "0" * 16
    (folder / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")

    run = collect(tmp_path / "eval", suite_dir)["runs"][0]
    assert run["graded"] == "recorded"
    assert "not re-graded" in run["grading_note"]


def test_meters_come_from_the_recording_not_from_the_replay(tmp_path, suite_dir):
    """A replay runs in milliseconds for nothing; showing that as the run's
    latency and cost would describe the replay."""
    recorded = {"latency_ms": {"mean": 12345, "p50": 12000, "p95": 20000}}
    write_run(tmp_path / "eval" / "good", SCRIPT, meters=recorded)
    run = collect(tmp_path / "eval", suite_dir)["runs"][0]
    assert run["latency_ms"] == recorded["latency_ms"]


def test_model_output_cannot_close_the_script_element(tmp_path, suite_dir):
    hostile = "</script><script>alert(1)</script> & <b>"
    page = render({"runs": [], "cases": [{"id": "x", "prompt": hostile}],
                   "judge": [], "panels": []})
    data_block = page.split('<script id="data" type="application/json">', 1)[1]
    data_block = data_block.split("</script>", 1)[0]
    assert "<" not in data_block and ">" not in data_block
    assert json.loads(data_block)["cases"][0]["prompt"] == hostile
