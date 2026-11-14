"""Turning attempts into numbers, with the uncertainty left in.

Pass rates come with a Wilson 95% interval. On a suite of twenty cases run three
times each, a difference of one or two attempts is not a regression, and an
interval is the cheapest way to stop a reader — or a CI gate — from treating it
as one.

A case that passes some of its repeats and fails others is reported as flaky and
counted separately. Averaging a flaky case into the headline rate turns "this
behaviour is unreliable" into "this behaviour is 67% correct", which is a
different and much less useful statement.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

from .runner import RunReport


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because suites are small and rates sit
    near 0 or 1, where the normal interval runs past the ends of the scale and
    reports a lower bound below zero.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation: with a handful of attempts,
    interpolating invents a latency nothing actually took."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[rank]


def _rate(successes: int, total: int) -> dict[str, Any]:
    low, high = wilson(successes, total)
    return {
        "passed": successes,
        "of": total,
        "rate": (successes / total) if total else 0.0,
        "ci95": [round(low, 4), round(high, 4)],
    }


def summarise(report: RunReport) -> dict[str, Any]:
    attempts = report.attempts
    scored = [a for a in attempts if not a.errored]
    errored = [a for a in attempts if a.errored]

    per_case: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        entry = per_case.setdefault(
            attempt.case_id,
            {
                "attempts": 0, "passed": 0, "safe": 0, "errored": 0,
                "failed_graders": {}, "labels": attempt.labels,
            },
        )
        entry["attempts"] += 1
        if attempt.errored:
            entry["errored"] += 1
            continue
        entry["passed"] += int(attempt.passed)
        entry["safe"] += int(attempt.safe)
        for name in attempt.failed_graders:
            entry["failed_graders"][name] = entry["failed_graders"].get(name, 0) + 1

    flaky: list[str] = []
    for case_id, entry in per_case.items():
        scorable = entry["attempts"] - entry["errored"]
        entry["scored"] = scorable
        entry["pass_rate"] = (entry["passed"] / scorable) if scorable else None
        entry["flaky"] = bool(scorable > 1 and 0 < entry["passed"] < scorable)
        if entry["flaky"]:
            flaky.append(case_id)

    per_grader: dict[str, dict[str, int]] = {}
    for attempt in scored:
        for grader in attempt.graders:
            if not grader.applicable:
                continue
            entry = per_grader.setdefault(
                grader.name, {"checked": 0, "failed": 0, "severity": grader.severity}
            )
            entry["checked"] += 1
            entry["failed"] += int(not grader.passed)

    diagnostics: dict[str, int] = {}
    for attempt in scored:
        for label in attempt.diagnostics:
            diagnostics[label] = diagnostics.get(label, 0) + 1

    latencies = [float(a.run.wall_ms) for a in scored]
    costs = [a.run.cost_usd for a in scored]
    tokens = [float(a.run.total_tokens) for a in scored]
    peaks = [float(a.run.peak_prompt_tokens) for a in scored]

    cases_fully_passing = sum(
        1 for e in per_case.values() if e["scored"] and e["passed"] == e["scored"]
    )

    return {
        "meta": report.meta,
        "totals": {
            "attempts": len(attempts),
            "scored": len(scored),
            "errored": len(errored),
            "cases": len(per_case),
            "cases_passing_every_repeat": cases_fully_passing,
            "flaky_cases": sorted(flaky),
        },
        "pass": _rate(sum(1 for a in scored if a.passed), len(scored)),
        "safe": _rate(sum(1 for a in scored if a.safe), len(scored)),
        "per_case": per_case,
        "per_grader": per_grader,
        "diagnostics": diagnostics,
        "cost": {
            "total_usd": round(sum(costs), 6),
            "mean_usd": round(mean(costs), 6) if costs else 0.0,
        },
        "latency_ms": {
            "mean": round(mean(latencies)) if latencies else 0,
            "p50": round(percentile(latencies, 50)),
            "p95": round(percentile(latencies, 95)),
        },
        "tokens": {
            "mean": round(mean(tokens)) if tokens else 0,
            "total": int(sum(tokens)),
            "peak_prompt_max": int(max(peaks)) if peaks else 0,
            "peak_prompt_p95": round(percentile(peaks, 95)),
        },
        "errors": [
            {"case_id": a.case_id, "repeat": a.repeat, "error": a.run.error}
            for a in errored
        ],
    }


def _pct(value: float | None) -> str:
    return "   -  " if value is None else (str(round(value * 100, 1)).rjust(5) + "%")


def format_text(summary: dict[str, Any], width: int = 78) -> str:
    """The CLI view. Reads top-down: what ran, how it did, what broke."""
    meta = summary["meta"]
    totals = summary["totals"]
    lines: list[str] = []

    lines.append("=" * width)
    lines.append(
        "suite " + str(meta.get("suite"))
        + "   model " + str(meta.get("model"))
        + ("  + reviewer " + str(meta.get("reviewer_model")) if meta.get("reviewer") else "")
    )
    lines.append(
        "prompt " + str(meta.get("prompt_version"))
        + "   git " + str(meta.get("git_sha") or "-")
        + "   " + str(totals["cases"]) + " cases x " + str(meta.get("repeats"))
        + " = " + str(totals["attempts"]) + " attempts"
        + "   " + str(meta.get("duration_s")) + "s"
    )
    lines.append("=" * width)

    p, s = summary["pass"], summary["safe"]
    lines.append(
        "fully correct   " + _pct(p["rate"]) + "  (" + str(p["passed"]) + "/" + str(p["of"])
        + ")   95% CI " + str(round(p["ci95"][0] * 100, 1)) + "-"
        + str(round(p["ci95"][1] * 100, 1)) + "%"
    )
    lines.append(
        "no critical     " + _pct(s["rate"]) + "  (" + str(s["passed"]) + "/" + str(s["of"])
        + ")   95% CI " + str(round(s["ci95"][0] * 100, 1)) + "-"
        + str(round(s["ci95"][1] * 100, 1)) + "%"
    )
    lines.append(
        "cases clean on every repeat   " + str(totals["cases_passing_every_repeat"])
        + "/" + str(totals["cases"])
    )
    if totals["errored"]:
        lines.append(
            "ERRORED         " + str(totals["errored"])
            + " attempts excluded from the rates above"
        )
    if totals["flaky_cases"]:
        lines.append("flaky           " + ", ".join(totals["flaky_cases"]))

    lines.append("-" * width)
    lines.append("graders (failures / checked)")
    for name, entry in sorted(
        summary["per_grader"].items(), key=lambda kv: (-kv[1]["failed"], kv[0])
    ):
        bar = "!" * min(20, entry["failed"])
        lines.append(
            "  " + name.ljust(12) + str(entry["failed"]).rjust(4) + " / "
            + str(entry["checked"]).ljust(5) + entry["severity"].ljust(10) + bar
        )

    if summary.get("diagnostics"):
        lines.append("-" * width)
        lines.append("what happened (attempts, not exclusive)")
        for label, count in sorted(
            summary["diagnostics"].items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append("  " + label.ljust(34) + str(count).rjust(4))

    failing = {
        cid: e for cid, e in summary["per_case"].items()
        if e["scored"] and e["passed"] < e["scored"]
    }
    if failing:
        lines.append("-" * width)
        lines.append("cases that failed at least one repeat")
        for case_id, entry in sorted(failing.items()):
            which = ", ".join(
                name + " x" + str(count)
                for name, count in sorted(entry["failed_graders"].items())
            )
            lines.append(
                "  " + case_id.ljust(28) + _pct(entry["pass_rate"])
                + ("  FLAKY " if entry["flaky"] else "        ") + which
            )

    lines.append("-" * width)
    lines.append(
        "latency  mean " + str(summary["latency_ms"]["mean"]) + "ms"
        + "   p50 " + str(summary["latency_ms"]["p50"]) + "ms"
        + "   p95 " + str(summary["latency_ms"]["p95"]) + "ms"
        + "   tokens mean " + str(summary["tokens"]["mean"])
        + "   peak prompt " + str(summary["tokens"].get("peak_prompt_max", 0))
        + "   cost $" + str(summary["cost"]["total_usd"])
    )
    for error in summary["errors"][:5]:
        lines.append("  error  " + error["case_id"] + " #" + str(error["repeat"])
                     + "  " + str(error["error"])[:60])
    return "\n".join(lines)


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Put two runs side by side, per case.

    The headline difference between two configurations is usually smaller than
    the interval around either of them, so the per-case movement is the part
    worth reading: which cases it fixed, which it broke, and which it left alone.
    A configuration that fixes three cases and breaks three has not improved
    anything, and the two rates alone would call that a draw.
    """
    fixed: list[str] = []
    broken: list[str] = []
    both_fail: list[str] = []
    unchanged_pass: list[str] = []

    cases = sorted(set(before["per_case"]) | set(after["per_case"]))
    for case_id in cases:
        was, now = before["per_case"].get(case_id), after["per_case"].get(case_id)
        if not was or not now or not was["scored"] or not now["scored"]:
            continue
        was_clean = was["passed"] == was["scored"]
        now_clean = now["passed"] == now["scored"]
        if now_clean and not was_clean:
            fixed.append(case_id)
        elif was_clean and not now_clean:
            broken.append(case_id)
        elif not was_clean:
            both_fail.append(case_id)
        else:
            unchanged_pass.append(case_id)

    labels = sorted(set(before.get("diagnostics", {})) | set(after.get("diagnostics", {})))
    diagnostics = {
        label: {
            "before": before.get("diagnostics", {}).get(label, 0),
            "after": after.get("diagnostics", {}).get(label, 0),
        }
        for label in labels
    }

    return {
        "pass": {"before": before["pass"], "after": after["pass"]},
        "safe": {"before": before["safe"], "after": after["safe"]},
        "fixed": fixed,
        "broken": broken,
        "still_failing": both_fail,
        "still_passing": unchanged_pass,
        "diagnostics": diagnostics,
        "latency_ms": {
            "before": before["latency_ms"]["mean"],
            "after": after["latency_ms"]["mean"],
        },
        "tokens": {"before": before["tokens"]["mean"], "after": after["tokens"]["mean"]},
    }


def format_comparison(
    cmp: dict[str, Any], before_name: str = "before", after_name: str = "after", width: int = 78
) -> str:
    lines = ["=" * width, before_name + "  ->  " + after_name, "=" * width]
    for key in ("pass", "safe"):
        b, a = cmp[key]["before"], cmp[key]["after"]
        delta = (a["rate"] - b["rate"]) * 100
        # Overlapping intervals mean the suite cannot tell these apart, and
        # saying so is more useful than reporting the sign of the difference.
        overlap = not (a["ci95"][0] > b["ci95"][1] or b["ci95"][0] > a["ci95"][1])
        lines.append(
            key.ljust(6) + _pct(b["rate"]) + "  ->" + _pct(a["rate"])
            + "   " + ("+" if delta >= 0 else "") + str(round(delta, 1)) + " points"
            + ("   (intervals overlap: not distinguishable)" if overlap else "   (intervals clear)")
        )
    lines.append("-" * width)
    for name, key in (
        ("fixed", "fixed"),
        ("broken", "broken"),
        ("still failing", "still_failing"),
    ):
        if cmp[key]:
            lines.append(name.ljust(14) + str(len(cmp[key])) + "  " + ", ".join(cmp[key]))
    lines.append("still passing " + str(len(cmp["still_passing"])))
    lines.append("-" * width)
    lines.append("diagnostics".ljust(36) + before_name[:12].rjust(8) + after_name[:12].rjust(10))
    for label, counts in sorted(
        cmp["diagnostics"].items(), key=lambda kv: -(kv[1]["before"] + kv[1]["after"])
    ):
        lines.append(
            "  " + label.ljust(34) + str(counts["before"]).rjust(8) + str(counts["after"]).rjust(10)
        )
    lines.append("-" * width)
    lines.append(
        "mean latency " + str(cmp["latency_ms"]["before"]) + "ms -> "
        + str(cmp["latency_ms"]["after"]) + "ms"
        + "   mean tokens " + str(cmp["tokens"]["before"]) + " -> " + str(cmp["tokens"]["after"])
    )
    return "\n".join(lines)


def _per_attempt_mismatches(
    now: dict[str, Any], before: dict[str, Any], now_n: int, before_n: int, label: str
) -> list[str]:
    """Counts that should be identical per attempt, compared exactly.

    A replay is deterministic, so any difference in what the graders checked or
    failed is a change in the grading rather than noise. The comparison is per
    attempt because a baseline recorded with three repeats is compared against a
    replay of one, and it is done by cross-multiplying integers rather than
    dividing, so there is no float to be nearly-equal.
    """
    out: list[str] = []
    if not now_n or not before_n:
        return out
    for key in sorted(set(now) | set(before)):
        if key not in before:
            out.append(label + " " + key + " is new; the baseline predates it")
            continue
        if key not in now:
            out.append(label + " " + key + " has disappeared since the baseline")
            continue
        a, b = now[key], before[key]
        for field in ("checked", "failed"):
            if field not in a or field not in b:
                continue
            if a[field] * before_n != b[field] * now_n:
                out.append(
                    label + " " + key + "." + field + ": " + str(a[field]) + "/"
                    + str(now_n) + " attempts, baseline " + str(b[field]) + "/"
                    + str(before_n)
                )
    return out


def gate(
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
    tolerance: float = 0.0,
    strict: bool = False,
):
    """Compare a summary with a baseline. Stage 2 wires this into CI.

    Returns (ok, lines). The comparison is against the baseline's lower
    confidence bound rather than its point estimate, so ordinary sampling noise
    does not fail a build.

    The per-case rule needs the same restraint, and the first version did not
    have it: it failed the build whenever a case that used to pass every repeat
    now failed any of them. Measuring this model showed why that is too strict.
    Three runs on identical input went 75.0%, 75.0%, 70.8% - two byte-identical,
    the third rewriting 9 of 24 replies and flipping one safety case - so a case
    dropping from 3/3 to 2/3 is inside the noise floor rather than a regression.
    (An earlier version of this comment put the spread at 12.5 points, which was
    inflated by a wall-clock leak in the simulation; it is 4.2.) A case
    is only failed here when it goes from clean to failing *every* repeat. One
    that becomes unreliable is reported as newly flaky and does not fail the
    build, because a gate that fires on its own noise gets switched off.

    `strict` additionally requires the per-grader and diagnostic counts to be
    identical per attempt, and exists because the two rules above were tested by
    disabling a critical grader outright and the gate passed. Nothing failed, so
    nothing fired: the rates and the per-case rule only notice grading that got
    *harsher*, while the dangerous direction for an eval platform is grading that
    quietly got weaker. On a replay, where the model's output cannot move, any
    difference at all is a code change and should have to be acknowledged by
    refreshing the baseline on purpose.
    """
    lines: list[str] = []
    if baseline is None:
        lines.append("no baseline to compare against")
        return True, lines

    ok = True
    for key in ("pass", "safe"):
        now, before = summary[key], baseline[key]
        floor = before["ci95"][0] - tolerance
        verdict = "ok" if now["rate"] >= floor else "REGRESSION"
        if now["rate"] < floor:
            ok = False
        lines.append(
            key.ljust(6) + " now " + _pct(now["rate"])
            + "  baseline " + _pct(before["rate"])
            + "  floor " + _pct(floor) + "  " + verdict
        )

    newly_failing: list[str] = []
    newly_flaky: list[str] = []
    for case_id, entry in summary["per_case"].items():
        was = baseline["per_case"].get(case_id)
        if not was or not was["scored"] or not entry["scored"]:
            continue
        if was["passed"] != was["scored"]:
            continue
        if entry["passed"] == 0:
            newly_failing.append(case_id)
        elif entry["passed"] < entry["scored"]:
            newly_flaky.append(case_id)

    if newly_failing:
        ok = False
        lines.append("REGRESSION: cases that used to pass and now fail every repeat: "
                     + ", ".join(sorted(newly_failing)))
    if newly_flaky:
        lines.append("newly flaky (reported, does not fail the build): "
                     + ", ".join(sorted(newly_flaky)))

    if strict:
        now_n = summary.get("totals", {}).get("attempts") or 0
        before_n = baseline.get("totals", {}).get("attempts") or 0
        drift = _per_attempt_mismatches(
            summary.get("per_grader", {}), baseline.get("per_grader", {}),
            now_n, before_n, "grader",
        )
        diag_now = {k: {"failed": v} for k, v in summary.get("diagnostics", {}).items()}
        diag_was = {k: {"failed": v} for k, v in baseline.get("diagnostics", {}).items()}
        drift += _per_attempt_mismatches(
            diag_now, diag_was, now_n, before_n, "diagnostic"
        )
        if drift:
            ok = False
            lines.append("REGRESSION: the grading itself changed against a recording "
                         "that cannot have changed:")
            lines.extend("  " + line for line in drift)
        else:
            lines.append("grading identical to the baseline, grader for grader")
    return ok, lines
