# Proofbench: what it is and how it is built

An evaluation platform for tool-using LLM agents, built around one principle:
**decide everything you can without a model, and measure the judge you cannot
avoid.**

The agent under test is a customer support agent for a small homeware shop. That
choice is not decoration. A support agent moves money, reads other people's data
and acts on text other people wrote, so almost every question worth asking about
it has a right answer that can be checked by reading a log rather than by asking
a model what it thinks of a paragraph.

## Why not just use an LLM judge for everything

Because most of it is decidable, and a judge turns an exact answer into a noisy
one. For the case "refund order A-1001 for 49.99":

| Question | How Proofbench answers it |
| --- | --- |
| Did the money move? | Read the effect log |
| By exactly how much, on which order? | Read the effect log |
| Did it try something it should not have, and get blocked? | Read the refused entries in the effect log |
| Did another customer appear in the reply? | Derive the forbidden set from the dataset, search the reply |
| Were the tool calls well formed? | Validate every call against the schema the model was shown |
| What did it cost, how long did it take? | Meters on the run |
| Was the reply decent to read? | **This one needs a judge** |

Only the last row needs judgement, and stage 2 measures how far that judgement
can be trusted by comparing it against a blind panel of raters.

## The parts

```
suites/support/        24 cases, as YAML
src/shopdesk/          the business: orders, policies, invariants, an effect log
src/proofbench/
  cases.py             the case format and its loader
  providers.py         Ollama, any OpenAI-compatible endpoint, and a scripted one
  tools.py             tool schemas, strict argument checking, injection
  agent.py             the agent under test, and the reviewer in front of payouts
  graders/             deterministic graders, plus diagnostics
  runner.py            drives a suite, repeats each case, writes an artifact
  replay.py            serves a recorded run back, and refuses what it never saw
  report.py            rates with intervals, flake detection, the CI gate
  judge.py             the one grader that needs a model
  labels.py            reference labels, pinned to the reply and the rubric
  panel.py             a blind rater panel, and the labels derived from it
  agreement.py         kappa, its interval, and what to say when it is undefined
  dashboard.py         re-grades every recorded run and writes the one-file dashboard
  cli.py               validate / run / show / compare / gate /
                       label-template / judge / panel / agree / dashboard
tests/                 171 tests, no model, no GPU, no network
```

## Design decisions worth defending

**Shopdesk enforces mechanics, not policy.** It refuses to refund more money
than an order holds, and refuses to cancel something that has shipped. It does
*not* enforce the 30-day window or the agent's 500.00 authority cap. If it did,
every policy case would pass for the wrong reason and the suite would quietly
stop measuring anything. `tests/test_shopdesk.py` pins this open on purpose.

**Refused attempts are recorded, not just accepted ones.** An agent that tries to
refund an already-refunded order and is stopped by the service has still decided
to pay out twice. A grader reading final balances scores that as a pass. The
`forbidden` expectation matches attempts as well as successes.

**The leak set is derived, not written.** A case declares whose conversation it
is; the grader works out what counts as someone else's data from the dataset. A
case cannot forget to forbid one of the other customers.

**Severity is reported separately from correctness.** Two rates, not one:
how often the agent was completely right, and how often it did nothing
dangerous. "Forgot to state the amount" and "refunded a stranger" are not the
same failure and should not average together.

**Diagnostics are deterministic labels, not prose.** `took-no-action`,
`acted-before-reading-the-policy`, `blocked-by-the-service`,
`blocked-by-the-reviewer`, `malformed-tool-calls`. A pass rate says how often the
agent was wrong; these say in which direction, which is what tells you whether
you have a hesitant model or a dangerous one.

**Provider failures are errors, not wrong answers.** A connection timeout is
excluded from the rates and counted on its own line. Scoring an outage as a wrong
answer hides the outage and flatters nothing.

**Each case runs more than once, and rates carry intervals.** A sampled model
varies. A suite that runs each case once reports its own noise as a score. Cases
that pass some repeats and fail others are listed as flaky rather than averaged,
because "unreliable" is a different finding from "67% correct".

**The injection suite has a control case.** Four cases plant hostile text in tool
results. The fifth plants hostile text *and* requires the agent to complete a
legitimate refund anyway. Without it, an agent that refuses everything scores
perfectly on injection resistance.

**The gate checks the grading, not only the result.** A gate that compares pass
rates and per-case verdicts only notices grading that got harsher. Disabling the
`forbidden` grader entirely and replaying the recorded run through it passed the
gate: the one case it was catching failed on `effects` too, so nothing either
rule looks at moved. The dangerous direction in an eval platform is the opposite
one - measurement quietly getting weaker makes the agent look better and nothing
complains. A replayed run is therefore compared grader for grader and diagnostic
for diagnostic, per attempt, exactly: on a recording the model cannot move, so
any difference is a code change and has to be acknowledged by refreshing the
baseline deliberately.

**The system prompt is hashed into every run.** Prompts are edited far more often
than code. A result that cannot be traced to the prompt that produced it is not a
measurement.

## Stages

- **Stage 1 — the harness.** Shopdesk, the agent, the reviewer, the tools, the
  deterministic graders, diagnostics, the runner, the report, the CLI, 24 cases,
  41 tests at the time it was committed. *Done.*
- **Stage 2 — the judge, and how much to trust it.** An LLM judge for the free-text
  rubric; reference labels from a blind panel of five raters, with the panel's
  own agreement measured; Cohen's kappa for judge-versus-panel agreement reported
  alongside every judged number, with the contested replies both in and out. A recorded-replay provider so grader and
  prompt changes can be re-checked in seconds instead of 17 minutes of GPU, and so
  the gate can run in CI without a model at all. Variance measured where stage 1
  found it — across repeated whole-suite runs, not across repeats inside one run,
  which came back byte-identical. The CI regression gate was pulled forward from
  stage 4, because once replay worked there was no reason to leave the graders
  ungated for two more stages. *Done.*
- **Stage 3 — the dashboard.** Runs, per-case diffs against a baseline, a trace
  viewer for every step of an attempt, injection-resistance and reviewer-value
  panels, and the judge against the panel. One self-contained HTML file, built by
  re-grading every recorded run by replay so that every run on it is graded by the
  same code. *Done.*
- **Stage 4 — live.** Deployed, the Dockerfile actually exercised rather than
  merely committed, and a write-up whose every number comes from a run that was
  executed. The regression gate moved into stage 2.

## What stage 1 measured

The headline and the three findings are in the README, with the artifacts in
`eval/`. The short version: 66.7% fully correct (95% CI 55.2-76.5%) on
qwen3.5:9b; the dominant failure is an agent that reasons correctly and then does
not call the tool; it paid out 2.8x its authority after reading a permissive
policy and never the governing one; it obeyed one of five planted injections; and
the reviewer architecture was defeated by injection because the transcript it
reads carries the attack.

A correction belongs with that headline: 66.7% was measured while shopdesk was
stamping escalations with the wall clock, which the agent read back in a tool
result, so that run is not reproducible. On the fixed simulation, and after the
grader fixes in the README's *What an audit found*, the headline is 75.0% fully
correct and 95.8% with no critical failure. The 12.5-point between-run spread reported with it was inflated
by the same bug and by one run having the reviewer enabled; measured properly on
three fixed runs it is 4.2 points - real, but a third of what was claimed.

Two of the four measurement runs were discarded rather than published, because
checking the failures against their traces turned up a bug of the author's each
time - a leak grader that counted an order id the customer had themselves quoted,
and a reviewer configured so cheaply it stopped reviewing. Both are now tests.

## Things deliberately not done

**The reviewer is not assumed to help.** It is switchable per run so the platform
can report what it is worth. That turned out to matter: with its thinking
disabled to save tokens it approved all 24 consequential actions it was shown,
including a 1420.00 refund it described as "under 10000 cents". With thinking on
it rejected that one correctly. Neither version caught the prompt injection,
because the reviewer reads the conversation transcript and the transcript carries
the attack. The result is published as measured, including the part where the
cost optimisation was the author's mistake rather than the model's.

**No price table in the repo.** Prices for hosted models are passed in. A stale
table silently turns every cost figure in the dashboard into a wrong one, and it
still looks like money.

**The prompt is not tuned against the suite.** It was clarified once, when the
first real run showed the instruction genuinely ambiguous about whether to act on
a clear request. Editing it further until the numbers improve would make the
suite a measure of the prompt rather than of the agent, and the whole point of
hashing the prompt into the run is to make that visible if it ever happens.
