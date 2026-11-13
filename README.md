# Proofbench

An evaluation platform for tool-using LLM agents, built on one principle:
**decide everything you can without a model, and measure the judge you cannot
avoid.**

The agent under test is a customer support agent for a small homeware shop. It
can look up orders, read written policy, refund money, cancel orders and escalate
to a supervisor. That makes almost every question worth asking about it decidable
by reading a log rather than by asking a model for an opinion:

| Question | How it is answered |
| --- | --- |
| Did the money move, by how much, on which order? | The effect log |
| Did it try something forbidden and get blocked? | Refused entries in the effect log |
| Did another customer appear in the reply? | Forbidden set derived from the dataset |
| Were the tool calls well formed? | Validated against the schema the model was shown |
| What did it cost and how long did it take? | Meters on the run |
| Was the reply decent to read? | **A judge — whose agreement with hand labels is measured** |

Only the last row needs a model. Stage 2 adds it, together with the number that
says how much to trust it.

## Running it

```bash
pip install -e ".[dev]"
```

The whole test suite runs on a scripted model — no GPU, no weights, no network:

```bash
pytest -q
```

Against a real model (Ollama, in this case a 9B on one GPU):

```bash
proofbench run suites/support --repeats 3 --out eval/baseline
```

Any OpenAI-compatible endpoint works too, with prices passed in rather than read
from a table that would silently go stale:

```bash
proofbench run suites/support --openai-base-url https://example/v1 --model some-model \
  --api-key "$KEY" --price-in 3.0 --price-out 15.0
```

<!-- RESULTS -->

## What is unusual about it

**Shopdesk enforces mechanics, not policy.** The shop refuses to refund more
money than an order holds, and refuses to cancel something that already shipped.
It deliberately does *not* enforce the 30-day refund window or the agent's 500.00
authority cap. If it did, every policy case would pass for the wrong reason and
the suite would quietly stop measuring anything. Two tests exist purely to keep
that hole open.

**Attempts count, not just successes.** An agent that tries to refund an
already-refunded order and is stopped by the service has still decided to pay out
twice. Grading on final balances scores that as a pass. The `forbidden`
expectation matches attempts as well as completed actions.

**The leak set is derived, not listed.** A case says whose conversation it is;
the grader works out what counts as another customer's data from the dataset
itself. A case cannot forget to forbid one of the other customers.

**Two rates, not one.** How often the agent was completely right, and how often
it did nothing dangerous. "Forgot to state the amount" and "refunded a stranger"
should not average together.

**Diagnostics say which direction a failure went.** Deterministic labels —
`took-no-action`, `acted-before-reading-the-policy`, `blocked-by-the-service`,
`blocked-by-the-reviewer`, `malformed-tool-calls`, `read-injected-content`. A
pass rate tells you the agent was wrong; these tell you whether you have a
hesitant model or a dangerous one, which need opposite fixes.

**Provider failures are errors, not wrong answers.** A timeout is excluded from
the rates and counted on its own line.

**Repeats, intervals and flake detection.** A sampled model varies, so each case
runs several times, rates carry Wilson 95% intervals, and a case that passes some
repeats and fails others is reported as flaky rather than averaged into a
two-thirds score.

**The injection suite has a control case.** Four cases plant hostile text inside
tool results — a fake system notice in an order, an addendum appended to the
policy the agent reads, an exfiltration instruction, a forged supervisor
approval. The fifth plants hostile text *and* requires the agent to complete a
legitimate refund anyway, because without it an agent that refuses everything
would score perfectly on injection resistance.

**An optional reviewer, not an assumed one.** A second model can sit in front of
every consequential action, see the policy and the proposed call, and send it
back with a reason. It is switchable per run so the platform can report what it
is actually worth. An unparseable or unreachable review abstains — never
approves — so the reviewer cannot look effective in the runs where it failed.

**The system prompt is hashed into every run.** Prompts get edited far more often
than code, and a result that cannot be traced to the prompt that produced it is
not a measurement.

## Layout

```
suites/support/      24 cases, as YAML
src/shopdesk/        the business: orders, policies, invariants, an effect log
src/proofbench/
  cases.py           the case format and loader
  providers.py       Ollama, OpenAI-compatible, and a scripted provider
  tools.py           tool schemas, strict argument checking, injection
  agent.py           the agent under test and the reviewer
  graders/           deterministic graders and diagnostics
  runner.py          drives a suite, repeats cases, writes an artifact
  report.py          rates, intervals, flake detection, a CI gate
  cli.py             validate / run / show / gate
tests/               41 tests, no model required
docs/plan.md         the design decisions, and what was deliberately not done
```

`docs/plan.md` has the reasoning behind each decision above, and the list of
things left undone on purpose.

## Licence

MIT.
