"""The command line.

    proofbench validate suites/support
    proofbench run suites/support --repeats 3
    proofbench run suites/support --reviewer --out runs/with-reviewer
    proofbench show runs/latest/summary.json
    proofbench gate runs/new/summary.json --baseline runs/baseline/summary.json
    proofbench judge runs/latest/artifact.json --out runs/latest/judge
    proofbench panel eval/panel/sampled.yaml --labels-out eval/labels/sampled.yaml
    proofbench agree runs/latest/judge/judgements.json --labels eval/labels/sampled.yaml

`run` writes two files: the full artifact, with every step of every attempt, and
a summary. The artifact is what you read when you want to know why a case failed;
the summary is what the gate and the dashboard consume. Keeping them separate
means a baseline committed to the repo stays small.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agreement import agreement, format_agreement
from .cases import load_suite
from .client import shopdesk_client
from .judge import (
    judge_artifact,
    judged_rate,
    judged_reply_keys,
    stale_verdict_keys,
    verdict_map,
    write_judgements,
)
from .labels import LabelEntry, LabelSet, align, load_labels, template
from .panel import (
    default_threshold,
    dump_labels,
    format_panel,
    load_panel,
    majority_threshold,
    summarise_panel,
    to_labels,
)
from .providers import OllamaProvider, OpenAICompatProvider, Provider
from .replay import ReplayProvider
from .report import compare, format_comparison, format_text, gate, summarise
from .runner import Attempt, Runner


def _provider(
    args: argparse.Namespace, model: str, think: bool | None = None
) -> Provider:
    if args.openai_base_url:
        return OpenAICompatProvider(
            base_url=args.openai_base_url,
            model=model,
            api_key=args.api_key or "",
            temperature=args.temperature,
            price_in_per_mtok=args.price_in,
            price_out_per_mtok=args.price_out,
            # Missing until an audit noticed: the judge's 900 s ceiling - raised
            # because 180 s silently dropped the hardest replies - never reached
            # this provider, which ran at its own 180 s default.
            timeout_s=getattr(args, "timeout", 180.0),
        )
    return OllamaProvider(
        base_url=args.ollama,
        model=model,
        temperature=args.temperature,
        seed=args.seed,
        num_ctx=args.num_ctx,
        think=think,
        timeout_s=getattr(args, "timeout", 180.0),
    )


def _build_reviewer(args: argparse.Namespace) -> Provider | None:
    """The reviewer thinks by default.

    Turning it off to save tokens was tried and measured: the terse reviewer
    approved all 24 consequential actions it saw, while the thinking one rejected
    an over-authority refund for the right reason. --no-reviewer-think is still
    there and costs about 50x fewer output tokens, but it buys a reviewer that
    does not review.
    """
    if not args.reviewer:
        return None
    return _provider(args, args.reviewer_model or args.model, think=args.reviewer_think)


def _progress(attempt: Attempt) -> None:
    mark = "E" if attempt.errored else ("." if attempt.passed else ("s" if attempt.safe else "X"))
    sys.stderr.write(mark)
    sys.stderr.flush()


def cmd_validate(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite))
    print("loaded " + str(len(suite.cases)) + " cases from " + str(args.suite))
    for case in suite.cases:
        checks = []
        if case.expect.effects:
            checks.append(str(len(case.expect.effects)) + " effects")
        if case.expect.forbidden:
            checks.append(str(len(case.expect.forbidden)) + " forbidden")
        if case.expect.tools_required:
            checks.append("tools")
        if case.expect.reply_matches or case.expect.reply_rejects:
            checks.append("reply")
        if case.inject:
            checks.append("injected")
        if not case.rubric:
            checks.append("NO RUBRIC")
        print("  " + case.id.ljust(38) + ", ".join(checks))
    missing = [c.id for c in suite.cases if not c.rubric]
    if missing:
        print("\ncases without a rubric (the judge will skip them): " + ", ".join(missing))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite))
    only = args.only.split(",") if args.only else None
    client = shopdesk_client(args.shopdesk)

    # One health check before spending minutes of GPU time on a bad URL.
    health = client.get("/health")
    if health.status_code != 200:
        print("shopdesk is not answering at " + str(args.shopdesk or "(in-process)"))
        return 2

    if args.replay:
        # One provider serves both roles: reviews are keyed by request hash in
        # the same recording, so a run that had a reviewer replays whole.
        replay = ReplayProvider(args.replay)
        print(replay.describe() + "\n")
        provider = replay
        reviewer_provider = replay if args.reviewer else None
    else:
        provider = _provider(args, args.model, think=args.think)
        reviewer_provider = _build_reviewer(args)

    runner = Runner(
        client=client,
        provider=provider,
        reviewer_provider=reviewer_provider,
        max_steps=args.max_steps,
        repeats=args.repeats,
        on_attempt=None if args.quiet else _progress,
    )
    report = runner.run_suite(suite, only=only)
    if not args.quiet:
        sys.stderr.write("\n")

    summary = summarise(report)
    summary["meta"]["replayed_from"] = args.replay or None
    print(format_text(summary))

    if args.replay:
        replay_provider = provider
        print(
            "\nreplay coverage " + str(round(replay_provider.coverage * 100, 1)) + "%"
            + "  (" + str(replay_provider.hits) + " hits, "
            + str(len(replay_provider.misses)) + " misses)"
        )
        if args.require_fresh:
            replay_provider.require_fresh()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report.write(out / "artifact.json")
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nwrote " + str(out / "artifact.json") + " and " + str(out / "summary.json"))

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        ok, lines = gate(
            summary, baseline, tolerance=args.tolerance, strict=_strict(summary, args)
        )
        print("\ngate against " + args.baseline)
        for line in lines:
            print("  " + line)
        if not ok:
            return 1
    return 0


def _strict(summary: dict, args: argparse.Namespace) -> bool:
    """Exact grader-for-grader comparison, on by default for a replayed summary.

    Derived from the summary itself rather than from a flag on the command that
    produced it, so `proofbench gate` on a recorded replay is strict even when
    whoever runs it does not know the run was a replay.
    """
    if getattr(args, "no_strict_gate", False):
        return False
    return bool(summary.get("meta", {}).get("replayed_from"))


def cmd_show(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    print(format_text(summary))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    baseline: dict | None = None
    if args.baseline:
        # A baseline that was asked for and is not there is an error, not "no
        # baseline, so pass". A mistyped path in CI would otherwise turn the gate
        # off and report success.
        if not Path(args.baseline).exists():
            print("baseline not found: " + args.baseline)
            return 2
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    ok, lines = gate(
        summary, baseline, tolerance=args.tolerance, strict=_strict(summary, args)
    )
    for line in lines:
        print(line)
    return 0 if ok else 1


def cmd_compare(args: argparse.Namespace) -> int:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    if before["meta"].get("prompt_version") != after["meta"].get("prompt_version"):
        print("note: the two runs used different system prompts, so this compares "
              "prompts as well as configurations\n")
    print(format_comparison(
        compare(before, after),
        before_name=args.before_name or Path(args.before).parent.name,
        after_name=args.after_name or Path(args.after).parent.name,
    ))
    return 0



def _suite_text(suite_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Rubrics and customer messages, by case id."""
    suite = load_suite(Path(suite_path))
    rubrics = {c.id: (c.rubric or "") for c in suite.cases}
    prompts = {c.id: c.prompt for c in suite.cases}
    return rubrics, prompts


def cmd_label_template(args: argparse.Namespace) -> int:
    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    rubrics, _ = _suite_text(args.suite)
    text = template(artifact, rubrics)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(str(out) + " already exists. Labels are the one thing here that "
              "cannot be regenerated, so this will not overwrite them without "
              "--force.")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("wrote " + str(out))
    print("fill in each `label:` with pass, fail or unsure, then run "
          "`proofbench judge`.")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    rubrics, prompts = _suite_text(args.suite)
    provider = _provider(args, args.model, think=args.think)

    def tick(record: dict) -> None:
        mark = {"pass": ".", "fail": "X", "abstain": "?"}.get(record["verdict"], "?")
        sys.stderr.write(mark)
        sys.stderr.flush()

    judged = judge_artifact(
        provider, artifact, rubrics, prompts, on_each=None if args.quiet else tick
    )
    if not args.quiet:
        sys.stderr.write("\n")

    out = Path(args.out)
    write_judgements(judged, out / "judgements.json")

    rate = judged_rate(judged)
    print("judge " + str(judged["meta"]["judge_model"])
          + " on " + str(rate["replies"]) + " distinct replies"
          + "  (" + str(judged["meta"]["judge_tokens"]) + " tokens)")
    if judged["meta"]["skipped_no_rubric"]:
        print("skipped, no rubric: "
              + ", ".join(sorted(set(judged["meta"]["skipped_no_rubric"]))))
    print("the judge passed " + str(rate["passed"]) + " of " + str(rate["decided"])
          + " it decided"
          + ("" if not rate["abstained"] else
             ", and abstained on " + str(rate["abstained"])))

    if not args.labels:
        print("\nNo labels given, so this rate is only what one model thought of "
              "another model's prose. Run with --labels to see what it is worth.")
        print("wrote " + str(out / "judgements.json"))
        return 0

    labels = load_labels(args.labels)
    report = _report_agreement(
        labels,
        verdict_map(judged),
        judged_reply_keys(judged),
        rubrics,
        stale_verdicts=set(),  # judged a moment ago, against these very rubrics
        draws=args.bootstrap_draws,
    )
    if report is None:
        return 2
    (out / "agreement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote " + str(out / "judgements.json") + " and "
          + str(out / "agreement.json"))
    return 0


def _report_agreement(
    labels: LabelSet,
    verdicts: dict[tuple[str, str], str],
    judged_replies: set[tuple[str, str]],
    rubrics: dict[str, str],
    stale_verdicts: set[tuple[str, str]],
    draws: int,
) -> dict | None:
    """Align labels with verdicts, print what that is worth, and return it.

    Everything that was left out of the comparison is listed by name and by
    cause, because each cause has a different remedy and a count alone hides
    which one applies.
    """
    alignment = align(
        labels, verdicts, judged_replies, rubrics=rubrics, stale_verdicts=stale_verdicts
    )
    print("\n" + alignment.describe())
    if not alignment.human:
        print("nothing to compare: no label matched a judged reply.")
        return None
    # Resampled by case. On a single run each case gives one reply and this is
    # the ordinary bootstrap; on pooled runs it stops one rubric counting twice.
    result = agreement(
        alignment.human,
        alignment.judge,
        draws=draws,
        clusters=[case_id for case_id, _ in alignment.keys],
    )
    print()
    print(format_agreement(result))
    if result["disagreements"]:
        print("\nwhere they disagreed:")
        for index in result["disagreements"]:
            print("  " + alignment.keys[index][0].ljust(40) + "labels "
                  + alignment.human[index] + ", judge " + alignment.judge[index])
    if alignment.stale_labels:
        print("\nstale labels (the reply changed since they were written):")
        for case_id, _ in sorted(alignment.stale_labels):
            print("  " + case_id)
    if alignment.judge_abstained:
        print("\nthe judge gave no verdict on these, so they are excluded from the "
              "agreement (a judge failure, not a disagreement):")
        for case_id, _ in sorted(alignment.judge_abstained):
            print("  " + case_id)
    if alignment.rubric_changed:
        print("\nthese rubrics have been reworded since their labels were written, so "
              "the labels and the judge answered different questions; re-label them:")
        for case_id, _ in sorted(alignment.rubric_changed):
            print("  " + case_id)
    if alignment.verdict_stale:
        print("\nthese verdicts were given under an older wording of the rubric, so "
              "they are not compared; re-judge them:")
        for case_id, _ in sorted(alignment.verdict_stale):
            print("  " + case_id)
    return {"agreement": result, "alignment": alignment.model_dump()}


def cmd_agree(args: argparse.Namespace) -> int:
    """Agreement from verdicts and labels that already exist. No model is called.

    Re-labelling, adding a rater or changing the reference rule should not cost
    another half hour of GPU, for the same reason a grader change should not:
    the judge's verdicts are data, written once.
    """
    rubrics, _ = _suite_text(args.suite)

    verdicts: dict[tuple[str, str], str] = {}
    judged_replies: set[tuple[str, str]] = set()
    stale: set[tuple[str, str]] = set()
    for path in args.judgements:
        judged = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, verdict in verdict_map(judged).items():
            if verdicts.get(key, verdict) != verdict:
                print("the same reply has two different verdicts across these files ("
                      + key[0] + "); they cannot be pooled. Pass one of them.")
                return 2
            verdicts[key] = verdict
        judged_replies |= judged_reply_keys(judged)
        stale |= stale_verdict_keys(judged, rubrics)

    merged: dict[tuple[str, str], LabelEntry] = {}
    for path in args.labels:
        for entry in load_labels(path).labels:
            key = (entry.case_id, entry.reply_sha)
            earlier = merged.get(key)
            if earlier is not None and earlier.label != entry.label:
                print("the same reply is labelled two different ways across these "
                      "files (" + key[0] + "); they cannot be pooled.")
                return 2
            merged[key] = entry
    labels = LabelSet(labels=list(merged.values()))

    print(str(len(verdicts)) + " verdicts from " + str(len(args.judgements))
          + " file(s), " + str(len(labels.labels)) + " labels from "
          + str(len(args.labels)) + " file(s)")
    report = _report_agreement(
        labels, verdicts, judged_replies, rubrics, stale, draws=args.bootstrap_draws
    )
    if report is None:
        return 2
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
        print("\nwrote " + str(out))
    return 0


def cmd_panel(args: argparse.Namespace) -> int:
    """What a rater panel said, how far it agreed with itself, and the labels
    that follow from it."""
    panel = load_panel(args.votes)
    raters = panel.raters
    if args.majority:
        threshold = majority_threshold(raters)
    else:
        threshold = args.threshold or default_threshold(raters)
    summary = summarise_panel(panel, threshold)
    print(format_panel(summary))

    if args.labels_out:
        out = Path(args.labels_out)
        if out.exists() and not args.force:
            print("\n" + str(out) + " already exists; --force to overwrite it.")
            return 2
        header = (
            "# Reference labels derived from the rater panel in "
            + Path(args.votes).as_posix() + ".\n"
            "# A reply is labelled pass or fail when at least " + str(threshold) + " of "
            + str(raters) + " raters agreed;\n"
            "# anything else is contested, written as unsure, and excluded from agreement.\n"
            "# Regenerate with `proofbench panel`; do not edit by hand.\n"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            dump_labels(to_labels(panel, threshold), header), encoding="utf-8", newline="\n"
        )
        print("\nwrote " + str(out))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
        print("wrote " + str(out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proofbench", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="load a suite and describe it")
    validate.add_argument("suite")
    validate.set_defaults(func=cmd_validate)

    run = subparsers.add_parser("run", help="run a suite and write an artifact")
    run.add_argument("suite")
    run.add_argument("--repeats", type=int, default=1,
                     help="attempts per case; more than one is what makes the "
                          "intervals mean anything")
    run.add_argument("--only", default="", help="comma-separated case ids")
    run.add_argument("--out", default="runs/latest")
    run.add_argument("--max-steps", type=int, default=10)
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--baseline", default="", help="summary.json to gate against")
    run.add_argument("--tolerance", type=float, default=0.0)
    # where things live
    run.add_argument("--replay", default="",
                     help="re-grade a recorded artifact.json instead of calling a "
                          "model; turns that do not match the recording are "
                          "errors, never silently served")
    run.add_argument("--require-fresh", action="store_true",
                     help="with --replay, fail if any turn had no recording")
    run.add_argument("--no-strict-gate", action="store_true",
                     help="allow a replayed run to differ from its baseline grader for grader")
    run.add_argument("--shopdesk", default="", help="base URL; omit to run it in-process")
    run.add_argument("--ollama", default="http://localhost:11434")
    run.add_argument("--model", default="qwen3.5:9b")
    run.add_argument("--reviewer", action="store_true", help="put a reviewer in front of payouts")
    run.add_argument("--reviewer-model", default="", help="defaults to --model")
    # openai-compatible alternative
    run.add_argument("--openai-base-url", default="")
    run.add_argument("--api-key", default="")
    run.add_argument("--price-in", type=float, default=0.0, help="USD per million input tokens")
    run.add_argument("--price-out", type=float, default=0.0, help="USD per million output tokens")
    # sampling
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--num-ctx", type=int, default=8192)
    run.add_argument("--timeout", type=float, default=300.0,
                     help="seconds per model call")
    run.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                     help="force the actor model to think, or not; omit to leave "
                          "the model default alone")
    run.add_argument("--reviewer-think", action=argparse.BooleanOptionalAction, default=None,
                     help="whether the reviewer may think before its verdict "
                          "(default: the model decides, which for a thinking model "
                          "means yes; --no-reviewer-think is ~50x cheaper and, "
                          "measured, approved everything it saw)")
    run.set_defaults(func=cmd_run)

    show = subparsers.add_parser("show", help="print a summary that was written earlier")
    show.add_argument("summary")
    show.set_defaults(func=cmd_show)

    comparison = subparsers.add_parser("compare", help="put two summaries side by side")
    comparison.add_argument("before")
    comparison.add_argument("after")
    comparison.add_argument("--before-name", default="")
    comparison.add_argument("--after-name", default="")
    comparison.set_defaults(func=cmd_compare)

    label_cmd = subparsers.add_parser(
        "label-template", help="write a YAML file of replies to hand-label")
    label_cmd.add_argument("artifact")
    label_cmd.add_argument("--suite", default="suites/support")
    label_cmd.add_argument("--out", default="eval/labels/labels.yaml")
    label_cmd.add_argument("--force", action="store_true",
                           help="overwrite an existing label file")
    label_cmd.set_defaults(func=cmd_label_template)

    judge_cmd = subparsers.add_parser(
        "judge", help="judge a recorded run's replies, and measure the judge")
    judge_cmd.add_argument("artifact")
    judge_cmd.add_argument("--suite", default="suites/support")
    judge_cmd.add_argument("--labels", default="",
                           help="hand labels to measure the judge against")
    judge_cmd.add_argument("--out", default="eval/judge")
    judge_cmd.add_argument("--bootstrap-draws", type=int, default=5000)
    judge_cmd.add_argument("--quiet", action="store_true")
    judge_cmd.add_argument("--ollama", default="http://localhost:11434")
    judge_cmd.add_argument("--model", default="qwen3.5:9b")
    judge_cmd.add_argument("--think", action=argparse.BooleanOptionalAction, default=None)
    judge_cmd.add_argument("--openai-base-url", default="")
    judge_cmd.add_argument("--api-key", default="")
    judge_cmd.add_argument("--price-in", type=float, default=0.0)
    judge_cmd.add_argument("--price-out", type=float, default=0.0)
    judge_cmd.add_argument("--temperature", type=float, default=0.0)
    judge_cmd.add_argument("--seed", type=int, default=7)
    judge_cmd.add_argument("--num-ctx", type=int, default=8192)
    # 180s was the first default and it silently dropped the two hardest replies
    # in the suite: a thinking judge ran past it on exactly those, they abstained,
    # and the agreement was then computed on the easy remainder. Excluding the
    # items most likely to disagree is how perfect agreement gets manufactured.
    judge_cmd.add_argument("--timeout", type=float, default=900.0,
                           help="seconds per verdict (default 900; a thinking "
                                "judge can take minutes on a hard reply)")
    judge_cmd.set_defaults(func=cmd_judge)

    agree_cmd = subparsers.add_parser(
        "agree", help="measure saved verdicts against labels, without calling a model")
    agree_cmd.add_argument("judgements", nargs="+",
                           help="one or more judgements.json files; several are pooled")
    agree_cmd.add_argument("--labels", nargs="+", required=True,
                           help="one or more label files")
    agree_cmd.add_argument("--suite", default="suites/support")
    agree_cmd.add_argument("--bootstrap-draws", type=int, default=5000)
    agree_cmd.add_argument("--out", default="", help="write the result as JSON")
    agree_cmd.set_defaults(func=cmd_agree)

    panel_cmd = subparsers.add_parser(
        "panel", help="summarise a rater panel and derive reference labels from it")
    panel_cmd.add_argument("votes", help="a panel file: one row of votes per reply")
    panel_cmd.add_argument("--threshold", type=int, default=0,
                           help="raters who must agree (default: four in five)")
    panel_cmd.add_argument("--majority", action="store_true",
                           help="label by simple majority instead, for the "
                                "sensitivity check")
    panel_cmd.add_argument("--labels-out", default="", help="write the derived labels")
    panel_cmd.add_argument("--out", default="", help="write the panel summary as JSON")
    panel_cmd.add_argument("--force", action="store_true")
    panel_cmd.set_defaults(func=cmd_panel)

    gate_parser = subparsers.add_parser("gate", help="compare a summary with a baseline")
    gate_parser.add_argument("summary")
    gate_parser.add_argument("--baseline", default="")
    gate_parser.add_argument("--tolerance", type=float, default=0.0)
    gate_parser.add_argument("--no-strict-gate", action="store_true",
                     help="allow a replayed run to differ from its baseline grader for grader")
    gate_parser.set_defaults(func=cmd_gate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
