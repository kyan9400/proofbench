"""The dashboard: every recorded run, graded by the graders as they are now.

A dashboard over saved summaries shows each run as it was graded on the day it
was recorded. On this project that would be actively misleading: the graders were
fixed after most runs were recorded, and a page that put a run graded by the old
effects severity next to one graded by the new would compare two different
measurements under one heading.

So building the dashboard re-grades every recorded run by replay, against the
current graders and the current suite, and shows those verdicts. The meters stay
as recorded - latency, tokens and cost belong to the run, and a replay's own
clock and price describe the replay.

A run that cannot be replayed whole is not re-graded partially. Its replay is
discarded, its verdicts are shown as they were recorded, and it is flagged with
the commit that graded it and the reason. A replay that half matched is not a
cheaper measurement but a measurement of something else, which is the same rule
`--require-fresh` enforces in CI.

The output is one HTML file with the data inside it: no server, no network, no
build step, so it can be opened from disk or served as a static file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cases import Suite, load_suite
from .client import shopdesk_client
from .replay import ReplayProvider
from .report import summarise
from .runner import Runner, RunReport

TEXT_LIMIT = 1500

# Runs listed first, in this order, when present. Anything else follows by name.
PREFERRED_ORDER = [
    "baseline", "determinism-2", "determinism-3", "sampled",
    "replay-check", "reviewer", "reviewer-thinking",
]


def _clip(value: Any, limit: int = TEXT_LIMIT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[" + str(len(text) - limit) + " more]"


def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": step.get("kind"), "index": step.get("index")}
    for key in ("tool_name", "decision", "reason", "error", "status_code"):
        if step.get(key) not in (None, "", [], {}):
            out[key] = step[key]
    if step.get("arguments") not in (None, "", {}):
        out["arguments"] = _clip(step["arguments"], 600)
    if step.get("result") not in (None, ""):
        out["result"] = _clip(step["result"])
    if step.get("text"):
        out["text"] = _clip(step["text"])
    if step.get("tool_calls"):
        out["tool_calls"] = [
            {"name": call.get("name"), "arguments": _clip(call.get("arguments"), 600)}
            for call in step["tool_calls"]
        ]
    if step.get("schema_errors"):
        out["schema_errors"] = step["schema_errors"]
    out["injected"] = bool(step.get("injected"))
    out["ok"] = step.get("ok")
    out["latency_ms"] = step.get("latency_ms") or 0
    out["tokens"] = (step.get("prompt_tokens") or 0) + (step.get("completion_tokens") or 0)
    return out


def regrade(artifact_path: Path, suite: Suite) -> tuple[RunReport | None, str]:
    """Replay a recorded run through the current graders.

    Returns (report, note). The report is None when the replay did not cover the
    recording completely, in which case nothing from it may be used.
    """
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    meta = artifact.get("meta", {})
    recorded_ids = {a["case_id"] for a in artifact.get("attempts", [])}
    cases = [c for c in suite.cases if c.id in recorded_ids]
    missing = sorted(recorded_ids - {c.id for c in suite.cases})
    if missing:
        return None, "cases no longer in the suite: " + ", ".join(missing)

    provider = ReplayProvider(artifact_path)
    runner = Runner(
        client=shopdesk_client(),
        provider=provider,
        reviewer_provider=provider if meta.get("reviewer") else None,
        max_steps=int(meta.get("max_steps") or 10),
        repeats=int(meta.get("repeats") or 1),
    )
    report = runner.run_suite(Suite(name=suite.name, cases=cases))
    if provider.misses or provider.coverage < 1.0:
        return None, (
            "replay matched " + str(round(provider.coverage * 100, 1)) + "% of model turns ("
            + str(len(provider.misses)) + " misses), so it was not re-graded"
        )
    return report, "re-graded by replay, " + str(provider.hits) + " turns, 100% coverage"


def _run_entry(name: str, folder: Path, suite: Suite) -> dict[str, Any]:
    artifact_path = folder / "artifact.json"
    recorded = json.loads(artifact_path.read_text(encoding="utf-8"))
    saved = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    meta = recorded.get("meta", {})

    regraded, note = regrade(artifact_path, suite)
    if regraded is not None:
        grading = summarise(regraded)
        verdict_source = {
            (a["case_id"], a["repeat"]): a for a in regraded.to_dict()["attempts"]
        }
        graded = "current"
    else:
        grading = saved
        verdict_source = {}
        graded = "recorded"

    attempts = []
    for attempt in recorded.get("attempts", []):
        verdict = verdict_source.get((attempt["case_id"], attempt["repeat"]), attempt)
        run = attempt.get("run", {})
        attempts.append({
            "case_id": attempt["case_id"],
            "repeat": attempt["repeat"],
            "passed": verdict["passed"],
            "safe": verdict["safe"],
            "errored": verdict["errored"],
            "diagnostics": verdict.get("diagnostics", []),
            "failures": [
                {"name": g["name"], "severity": g["severity"], "detail": g.get("detail", "")}
                for g in verdict.get("graders", [])
                if g.get("applicable", True) and not g["passed"]
            ],
            "reply": run.get("reply") or "",
            "stop_reason": run.get("stop_reason"),
            "wall_ms": run.get("wall_ms") or 0,
            "tokens": (run.get("prompt_tokens") or 0) + (run.get("completion_tokens") or 0),
            "steps": [_compact_step(s) for s in run.get("steps", [])],
        })

    return {
        "name": name,
        "graded": graded,
        "grading_note": note if graded == "current" else (
            note + "; verdicts shown as graded at " + str(meta.get("git_sha", "?"))
        ),
        "meta": {k: meta.get(k) for k in (
            "model", "git_sha", "repeats", "reviewer", "reviewer_think", "temperature",
            "seed", "prompt_version", "started_at", "duration_s", "replayed_from",
        )},
        "pass": grading["pass"],
        "safe": grading["safe"],
        "totals": grading["totals"],
        "per_case": grading["per_case"],
        "per_grader": grading["per_grader"],
        "diagnostics": grading["diagnostics"],
        # Meters as recorded, never from the replay.
        "latency_ms": saved.get("latency_ms"),
        "tokens": saved.get("tokens"),
        "cost": saved.get("cost"),
        "attempts": attempts,
    }


def _judge_entries(eval_dir: Path) -> list[dict[str, Any]]:
    labels = {
        "judge/baseline/agreement.json": "first labels, by the author (superseded)",
        "judge/sharpened/agreement.json":
            "author labels revised after reading the judge (contaminated)",
        "judge/sharpened/agreement-panel.json": "baseline replies, blind panel",
        "judge/sampled/agreement.json": "held-out replies, blind panel",
        "judge/pooled-agreement.json": "both sets pooled, blind panel",
    }
    out = []
    for rel, title in labels.items():
        path = eval_dir / rel
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        result = data.get("agreement", data)
        out.append({
            "file": "eval/" + rel,
            "title": title,
            "n": result.get("n"),
            "kappa": result.get("kappa"),
            "ci": result.get("kappa_ci95"),
            "observed": result.get("observed_agreement"),
            "confusion": result.get("confusion"),
            "bound": result.get("max_disagreement_rate_if_none_seen"),
            "clusters": result.get("clusters"),
        })
    return out


def _panel_entries(eval_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((eval_dir / "panel").glob("*.summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["set"] = path.name.split(".")[0]
        out.append(data)
    return out


def collect(eval_dir: str | Path, suite_path: str | Path) -> dict[str, Any]:
    eval_dir = Path(eval_dir)
    suite = load_suite(Path(suite_path))
    folders = {
        p.name: p for p in eval_dir.iterdir()
        if p.is_dir() and (p / "artifact.json").exists() and (p / "summary.json").exists()
    }
    order = [n for n in PREFERRED_ORDER if n in folders] + sorted(
        n for n in folders if n not in PREFERRED_ORDER
    )
    runs = [_run_entry(name, folders[name], suite) for name in order]
    cases = [
        {
            "id": c.id,
            "title": c.title,
            "category": c.labels.get("category", ""),
            "difficulty": c.labels.get("difficulty", ""),
            "prompt": c.prompt,
            "rubric": " ".join((c.rubric or "").split()),
            "injected": bool(c.inject),
            "forbidden": [f.note for f in c.expect.forbidden if f.note],
        }
        for c in suite.cases
    ]
    return {
        "runs": runs,
        "cases": cases,
        "judge": _judge_entries(eval_dir),
        "panels": _panel_entries(eval_dir),
    }


def render(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # Replies are model output and may contain anything, including the sequence
    # that would end the script element early. Escaped here, decoded by JSON.parse.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return TEMPLATE.replace("__DATA__", payload)


def build(eval_dir: str | Path, suite_path: str | Path, out: str | Path) -> dict[str, Any]:
    data = collect(eval_dir, suite_path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8", newline="\n")
    return data


TEMPLATE = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
