"""The command line.

    proofbench validate suites/support
    proofbench run suites/support --repeats 3
    proofbench run suites/support --reviewer --out runs/with-reviewer
    proofbench show runs/latest/summary.json
    proofbench gate runs/new/summary.json --baseline runs/baseline/summary.json

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

from .cases import load_suite
from .client import shopdesk_client
from .providers import OllamaProvider, OpenAICompatProvider, Provider
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
        )
    return OllamaProvider(
        base_url=args.ollama,
        model=model,
        temperature=args.temperature,
        seed=args.seed,
        num_ctx=args.num_ctx,
        think=think,
    )


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

    provider = _provider(args, args.model, think=args.think)
    # The reviewer thinks by default. Turning it off to save tokens was tried and
    # measured: the terse reviewer approved all 24 consequential actions it saw,
    # the thinking one rejected an over-authority refund for the right reason.
    # --no-reviewer-think is still there, and costs about 50x fewer output tokens,
    # but it buys a reviewer that does not review.
    reviewer_provider = (
        _provider(args, args.reviewer_model or args.model, think=args.reviewer_think)
        if args.reviewer
        else None
    )

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
    print(format_text(summary))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report.write(out / "artifact.json")
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nwrote " + str(out / "artifact.json") + " and " + str(out / "summary.json"))

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        ok, lines = gate(summary, baseline, tolerance=args.tolerance)
        print("\ngate against " + args.baseline)
        for line in lines:
            print("  " + line)
        if not ok:
            return 1
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    print(format_text(summary))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    baseline: dict | None = None
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    ok, lines = gate(summary, baseline, tolerance=args.tolerance)
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

    gate_parser = subparsers.add_parser("gate", help="compare a summary with a baseline")
    gate_parser.add_argument("summary")
    gate_parser.add_argument("--baseline", default="")
    gate_parser.add_argument("--tolerance", type=float, default=0.0)
    gate_parser.set_defaults(func=cmd_gate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
