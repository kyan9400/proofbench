"""The report is the product.

Everything a reader believes about a run comes through these functions, so the
arithmetic gets tested as carefully as the graders: an interval that is quietly
wrong is worse than no interval, because it looks like rigour.
"""

from __future__ import annotations

from proofbench.agent import AgentRun
from proofbench.cases import Severity
from proofbench.graders import GraderResult
from proofbench.report import (
    compare,
    format_comparison,
    format_text,
    gate,
    percentile,
    summarise,
    wilson,
)
from proofbench.runner import Attempt, RunReport


def attempt(
    case_id: str,
    repeat: int = 1,
    passed: bool = True,
    safe: bool = True,
    errored: bool = False,
    failed: tuple[tuple[str, Severity], ...] = (),
    diagnostics: tuple[str, ...] = (),
    wall_ms: int = 1000,
    tokens: int = 100,
) -> Attempt:
    graders = [GraderResult(name="effects", passed=True, severity="critical")]
    graders += [
        GraderResult(name=name, passed=False, severity=severity) for name, severity in failed
    ]
    run = AgentRun(reply="hello", prompt_tokens=tokens, completion_tokens=0, wall_ms=wall_ms)
    if errored:
        run.stop_reason = "provider_error"
        run.error = "ConnectError"
    return Attempt(
        case_id=case_id,
        suite="support",
        repeat=repeat,
        passed=passed,
        safe=safe,
        errored=errored,
        graders=graders,
        run=run,
        diagnostics=list(diagnostics),
    )


def report_of(attempts: list[Attempt], **meta) -> RunReport:
    base = {"suite": "support", "repeats": 1, "model": "m", "prompt_version": "abc"}
    base.update(meta)
    return RunReport(meta=base, attempts=attempts)


# ---------------------------------------------------------------- arithmetic


def test_wilson_stays_inside_the_scale_at_the_ends():
    """The reason it is Wilson and not the normal approximation.

    At 0 of 5 the normal interval runs below zero and reports a negative lower
    bound for a proportion, which no reader should ever be shown.
    """
    low, high = wilson(0, 5)
    assert low == 0.0
    assert 0.3 < high < 0.6

    low, high = wilson(5, 5)
    assert high == 1.0
    assert 0.4 < low < 0.6


def test_wilson_narrows_as_the_sample_grows():
    narrow = wilson(50, 100)
    wide = wilson(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_of_nothing_is_not_a_division_by_zero():
    assert wilson(0, 0) == (0.0, 0.0)


def test_percentile_returns_a_value_that_actually_occurred():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) in values
    assert percentile(values, 95) == 40.0
    assert percentile(values, 1) == 10.0
    assert percentile([], 50) == 0.0


# ---------------------------------------------------------------- summarise


def test_errored_attempts_are_excluded_from_the_rates_and_counted():
    summary = summarise(report_of([
        attempt("a", passed=True),
        attempt("b", passed=False, safe=False, failed=(("effects", "critical"),)),
        attempt("c", errored=True, passed=False, safe=False),
    ]))
    # Two scored attempts, one of them passing: 50%, not 33%.
    assert summary["pass"] == {"passed": 1, "of": 2, "rate": 0.5, "ci95": summary["pass"]["ci95"]}
    assert summary["totals"]["errored"] == 1
    assert summary["totals"]["scored"] == 2
    assert summary["errors"][0]["case_id"] == "c"


def test_a_case_that_passes_some_repeats_is_flaky_not_two_thirds_right():
    summary = summarise(report_of([
        attempt("flappy", repeat=1, passed=True),
        attempt("flappy", repeat=2, passed=False, failed=(("effects", "critical"),)),
        attempt("flappy", repeat=3, passed=True),
        attempt("solid", repeat=1, passed=True),
        attempt("solid", repeat=2, passed=True),
        attempt("solid", repeat=3, passed=True),
    ], repeats=3))
    assert summary["totals"]["flaky_cases"] == ["flappy"]
    assert summary["per_case"]["flappy"]["flaky"] is True
    assert summary["per_case"]["solid"]["flaky"] is False
    # A case must pass every repeat to count as clean.
    assert summary["totals"]["cases_passing_every_repeat"] == 1
    assert summary["totals"]["cases"] == 2


def test_a_case_that_fails_every_repeat_is_not_called_flaky():
    summary = summarise(report_of([
        attempt("broken", repeat=r, passed=False, failed=(("effects", "critical"),))
        for r in (1, 2, 3)
    ], repeats=3))
    assert summary["totals"]["flaky_cases"] == []
    assert summary["per_case"]["broken"]["pass_rate"] == 0.0


def test_safe_and_pass_are_tracked_separately():
    """A minor failure costs correctness but not safety."""
    summary = summarise(report_of([
        attempt("minor", passed=False, safe=True, failed=(("reply", "minor"),)),
        attempt("major", passed=False, safe=False, failed=(("leak", "critical"),)),
    ]))
    assert summary["pass"]["rate"] == 0.0
    assert summary["safe"]["rate"] == 0.5


def test_diagnostics_are_counted_across_attempts():
    summary = summarise(report_of([
        attempt("a", diagnostics=("took-no-action",)),
        attempt("b", diagnostics=("took-no-action", "blocked-by-the-service")),
        attempt("c", errored=True, diagnostics=("took-no-action",)),
    ]))
    # The errored attempt contributes nothing: its diagnostics describe a run
    # that did not happen.
    assert summary["diagnostics"] == {"took-no-action": 2, "blocked-by-the-service": 1}


def test_format_text_mentions_the_things_a_reader_needs():
    summary = summarise(report_of([
        attempt("a", passed=True),
        attempt("b", passed=False, safe=False, failed=(("effects", "critical"),),
                diagnostics=("took-no-action",)),
        attempt("c", errored=True, passed=False, safe=False),
    ]))
    text = format_text(summary)
    assert "ERRORED" in text
    assert "took-no-action" in text
    assert "95% CI" in text
    assert "effects" in text


# ---------------------------------------------------------------- gate


def test_the_gate_passes_when_there_is_no_baseline():
    summary = summarise(report_of([attempt("a", passed=True)]))
    ok, lines = gate(summary, None)
    assert ok
    assert "no baseline" in lines[0]


def test_the_gate_tolerates_noise_but_not_a_real_drop():
    baseline = summarise(report_of(
        [attempt("c" + str(i), passed=True) for i in range(20)]
    ))
    # One case of twenty dropping sits inside the baseline interval on the rate,
    # but it went from clean to failing outright, which the per-case rule catches
    # however small the effect on the headline is. That is the point of having
    # both checks: 95% still looks healthy.
    nearly = summarise(report_of(
        [attempt("c0", passed=False, failed=(("reply", "minor"),))]
        + [attempt("c" + str(i), passed=True) for i in range(1, 20)]
    ))
    ok, lines = gate(nearly, baseline)
    assert not ok
    assert any("ok" in line for line in lines[:2])  # the rate check itself passed
    assert any("c0" in line for line in lines)

    # Half of them regressing does not.
    halved = summarise(report_of(
        [attempt("c" + str(i), passed=False, failed=(("effects", "critical"),))
         for i in range(10)]
        + [attempt("c" + str(i), passed=True) for i in range(10, 20)]
    ))
    ok, lines = gate(halved, baseline)
    assert not ok
    assert any("REGRESSION" in line for line in lines)


def test_the_gate_fails_a_case_that_went_from_clean_to_failing_every_repeat():
    baseline = summarise(report_of([
        attempt("kept", passed=True), attempt("lost", passed=True),
    ]))
    now = summarise(report_of([
        attempt("kept", passed=True),
        attempt("lost", passed=False, failed=(("effects", "critical"),)),
    ]))
    ok, lines = gate(now, baseline)
    assert not ok
    assert any("lost" in line and "REGRESSION" in line for line in lines)


def test_a_case_that_merely_became_unreliable_is_reported_not_failed():
    """The noise floor this model actually has.

    Three runs on identical input: two byte-identical, the third rewriting 9 of
    24 replies and flipping one safety case. So a case going from clean to
    two-of-three is the floor, not a regression, and a gate that fails on it
    would be switched off within a week.
    """
    baseline = summarise(report_of(
        [attempt("wobbly", repeat=r, passed=True) for r in (1, 2, 3)]
        + [attempt("steady", repeat=r, passed=True) for r in (1, 2, 3)],
        repeats=3,
    ))
    now = summarise(report_of(
        [attempt("wobbly", repeat=1, passed=True),
         attempt("wobbly", repeat=2, passed=False, failed=(("effects", "critical"),)),
         attempt("wobbly", repeat=3, passed=True)]
        + [attempt("steady", repeat=r, passed=True) for r in (1, 2, 3)],
        repeats=3,
    ))
    ok, lines = gate(now, baseline)
    assert ok
    assert any("newly flaky" in line and "wobbly" in line for line in lines)


# ---------------------------------------------------------------- compare


def test_a_grader_that_stops_grading_is_caught_on_a_replay():
    """The hole the strict gate exists to close.

    This was found by disabling the forbidden grader outright and watching the
    gate pass. Nothing failed, so nothing fired: the rate checks and the per-case
    rule only notice grading that got harsher, and the dangerous direction for an
    eval platform is grading that quietly got weaker. On a replay the model's
    output cannot move, so any difference is a code change.
    """
    baseline = summarise(report_of([
        attempt("caught", passed=False, failed=(("forbidden", "critical"),)),
        attempt("fine", passed=True),
    ]))
    # The grader no longer fails anything - and no case changed verdict, because
    # the agent is still marked as having failed.
    toothless = summarise(report_of([
        attempt("caught", passed=False, failed=(("effects", "critical"),)),
        attempt("fine", passed=True),
    ], replayed_from="eval/baseline/artifact.json"))

    lenient, _ = gate(toothless, baseline)
    assert lenient, "the old gate passed this, which is why strict exists"

    ok, lines = gate(toothless, baseline, strict=True)
    assert not ok
    assert any("forbidden" in line for line in lines)


def test_strict_compares_per_attempt_so_repeat_counts_need_not_match():
    """A baseline recorded with three repeats is replayed with one."""
    baseline = summarise(report_of([
        attempt("a", repeat=r, passed=False, failed=(("tools", "major"),))
        for r in (1, 2, 3)
    ] + [attempt("b", repeat=r, passed=True) for r in (1, 2, 3)]))
    replayed = summarise(report_of([
        attempt("a", passed=False, failed=(("tools", "major"),)),
        attempt("b", passed=True),
    ], replayed_from="x"))

    ok, lines = gate(replayed, baseline, strict=True)
    assert ok, lines
    assert any("grader for grader" in line for line in lines)


def test_strict_catches_a_grader_appearing_or_vanishing():
    baseline = summarise(report_of([attempt("a", passed=True)]))
    with_extra = summarise(report_of([
        attempt("a", passed=False, failed=(("brand-new", "minor"),))
    ], replayed_from="x"))
    ok, lines = gate(with_extra, baseline, strict=True)
    assert not ok
    assert any("brand-new" in line and "new" in line for line in lines)


def test_strict_catches_a_diagnostic_that_moved():
    """Diagnostics are the cut a reader actually uses, so drift in them counts."""
    baseline = summarise(report_of([
        attempt("a", passed=True, diagnostics=("took-no-action",))
    ]))
    moved = summarise(report_of([
        attempt("a", passed=True, diagnostics=("blocked-by-the-service",))
    ], replayed_from="x"))
    ok, lines = gate(moved, baseline, strict=True)
    assert not ok
    assert any("diagnostic" in line for line in lines)


def test_compare_separates_cases_it_fixed_from_cases_it_broke():
    before = summarise(report_of([
        attempt("fixed-by-it", passed=False, failed=(("effects", "critical"),)),
        attempt("broken-by-it", passed=True),
        attempt("always-bad", passed=False, failed=(("effects", "critical"),)),
        attempt("always-good", passed=True),
    ]))
    after = summarise(report_of([
        attempt("fixed-by-it", passed=True),
        attempt("broken-by-it", passed=False, failed=(("effects", "critical"),)),
        attempt("always-bad", passed=False, failed=(("effects", "critical"),)),
        attempt("always-good", passed=True),
    ]))
    result = compare(before, after)
    assert result["fixed"] == ["fixed-by-it"]
    assert result["broken"] == ["broken-by-it"]
    assert result["still_failing"] == ["always-bad"]
    assert result["still_passing"] == ["always-good"]

    # The rates are identical, and the per-case movement is the only thing that
    # shows this configuration changed anything at all.
    assert result["pass"]["before"]["rate"] == result["pass"]["after"]["rate"]
    text = format_comparison(result, "before", "after")
    assert "fixed-by-it" in text
    assert "broken-by-it" in text


def test_compare_says_when_two_runs_cannot_be_told_apart():
    before = summarise(report_of([
        attempt("a", passed=True),
        attempt("b", passed=False, failed=(("reply", "minor"),)),
    ]))
    after = summarise(report_of([attempt("a", passed=True), attempt("b", passed=True)]))
    text = format_comparison(compare(before, after), "before", "after")
    assert "not distinguishable" in text


def test_compare_shows_diagnostics_moving():
    before = summarise(report_of([attempt("a", diagnostics=("took-no-action",))]))
    after = summarise(report_of([attempt("a", diagnostics=("blocked-by-the-reviewer",))]))
    result = compare(before, after)
    assert result["diagnostics"]["took-no-action"] == {"before": 1, "after": 0}
    assert result["diagnostics"]["blocked-by-the-reviewer"] == {"before": 0, "after": 1}
