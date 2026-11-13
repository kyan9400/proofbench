"""The runner: drive a suite, grade every attempt, write an artifact.

Three choices here that the rest of the platform depends on:

* Cases run one at a time against one shopdesk. The effect log is the ground
  truth for grading, and it is process-wide, so parallelism would mix two cases
  together and produce results that look fine.

* Each case runs `repeats` times. One attempt per case reads as a precise number
  and is not one: a sampled model varies, and a suite that runs each case once
  reports its own noise as a score. Stage 2 turns the repeats into intervals.

* A provider failure marks an attempt `errored`, not `failed`, and errored
  attempts are excluded from pass rates and counted separately. Scoring a
  connection timeout as a wrong answer flatters nothing and hides an outage.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .agent import ACTOR_PROMPT_VERSION, Agent, AgentRun, Reviewer
from .cases import Case, Suite
from .graders import GraderResult, diagnose, grade_programmatic
from .graders.programmatic import Observed
from .providers import Provider
from .tools import Toolbox


@dataclass
class Attempt:
    case_id: str
    suite: str
    repeat: int
    passed: bool
    safe: bool
    errored: bool
    graders: list[GraderResult]
    run: AgentRun
    labels: dict[str, str] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def failed_graders(self) -> list[str]:
        return [g.name for g in self.graders if g.applicable and not g.passed]

    @property
    def critical_failures(self) -> list[str]:
        return [
            g.name for g in self.graders
            if g.applicable and not g.passed and g.severity == "critical"
        ]


@dataclass
class RunReport:
    meta: dict[str, Any]
    attempts: list[Attempt]

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "attempts": [
                {
                    "case_id": a.case_id,
                    "suite": a.suite,
                    "repeat": a.repeat,
                    "passed": a.passed,
                    "safe": a.safe,
                    "errored": a.errored,
                    "labels": a.labels,
                    "diagnostics": a.diagnostics,
                    "graders": [asdict(g) for g in a.graders],
                    "run": {
                        "reply": a.run.reply,
                        "stop_reason": a.run.stop_reason,
                        "error": a.run.error,
                        "prompt_tokens": a.run.prompt_tokens,
                        "peak_prompt_tokens": a.run.peak_prompt_tokens,
                        "completion_tokens": a.run.completion_tokens,
                        "cost_usd": a.run.cost_usd,
                        "wall_ms": a.run.wall_ms,
                        "actor_model": a.run.actor_model,
                        "reviewer_model": a.run.reviewer_model,
                        "prompt_version": a.run.prompt_version,
                        "tool_names": a.run.tool_names,
                        "steps": [asdict(s) for s in a.run.steps],
                    },
                }
                for a in self.attempts
            ],
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
        return path


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _observe(client: httpx.Client) -> Observed:
    """What the world looks like now, from the grader's side of the fence."""
    effects = client.get("/_admin/effects").json()
    world = client.get("/_admin/world").json()
    return Observed(
        effects=effects["effects"],
        orders=effects["orders"],
        escalations=effects["escalations"],
        customers=world["customers"],
        all_orders=world["orders"],
    )


class Runner:
    def __init__(
        self,
        client: httpx.Client,
        provider: Provider,
        reviewer_provider: Provider | None = None,
        max_steps: int = 10,
        repeats: int = 1,
        on_attempt: Callable[[Attempt], None] | None = None,
    ) -> None:
        self.client = client
        self.provider = provider
        self.reviewer_provider = reviewer_provider
        self.max_steps = max_steps
        self.repeats = repeats
        self.on_attempt = on_attempt

    def _policy_text(self) -> str:
        topics = self.client.get("/policies").json()["topics"]
        parts = []
        for topic in topics:
            body = self.client.get("/policies/" + topic).json()
            parts.append(topic + ": " + body["text"])
        return "\n\n".join(parts)

    def run_case(self, case: Case, repeat: int) -> Attempt:
        self.client.post("/_admin/reset", json={"now": case.now})
        toolbox = Toolbox(self.client, injections=case.inject)
        reviewer = (
            Reviewer(self.reviewer_provider, self._policy_text())
            if self.reviewer_provider is not None
            else None
        )
        agent = Agent(
            self.provider, toolbox, max_steps=self.max_steps, reviewer=reviewer
        )
        run = agent.run(case.prompt)
        obs = _observe(self.client)
        graders = grade_programmatic(case, run, obs)
        diagnostics = diagnose(case, run, obs)

        errored = run.stop_reason == "provider_error"
        applicable = [g for g in graders if g.applicable]
        passed = (not errored) and all(g.passed for g in applicable)
        safe = (not errored) and not any(
            not g.passed for g in applicable if g.severity == "critical"
        )

        attempt = Attempt(
            case_id=case.id,
            suite=case.suite,
            repeat=repeat,
            passed=passed,
            safe=safe,
            errored=errored,
            graders=graders,
            run=run,
            labels=dict(case.labels),
            diagnostics=diagnostics,
        )
        if self.on_attempt:
            self.on_attempt(attempt)
        return attempt

    def run_suite(self, suite: Suite, only: list[str] | None = None) -> RunReport:
        cases = [c for c in suite.cases if not only or c.id in set(only)]
        started = time.time()
        attempts: list[Attempt] = []
        for case in cases:
            for repeat in range(1, self.repeats + 1):
                attempts.append(self.run_case(case, repeat))

        meta = {
            "suite": suite.name,
            "cases": len(cases),
            "repeats": self.repeats,
            "attempts": len(attempts),
            "provider": self.provider.name,
            "model": self.provider.model,
            "reviewer_model": (
                self.reviewer_provider.model if self.reviewer_provider else None
            ),
            "reviewer": self.reviewer_provider is not None,
            "actor_think": getattr(self.provider, "think", None),
            "reviewer_think": getattr(self.reviewer_provider, "think", None),
            "prompt_version": ACTOR_PROMPT_VERSION,
            "max_steps": self.max_steps,
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "started_at": datetime.fromtimestamp(started, UTC).isoformat(),
            "duration_s": round(time.time() - started, 1),
        }
        return RunReport(meta=meta, attempts=attempts)
