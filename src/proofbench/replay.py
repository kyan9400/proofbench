"""Replaying a recorded run, so grader work does not cost GPU time.

Stage 1 ran the suite against the model four times and threw two of those runs
away, because checking the failures against their traces turned up a bug in the
graders each time. Each round trip was seventeen minutes of GPU for a change that
touched no model input at all. That is the wrong price for fixing a grader.

A recorded run already contains every model turn. If the conversation handed to
the provider is byte-for-byte the one that was recorded, the recorded reply is
the reply, and the whole suite re-grades in about a second.

The part that makes this trustworthy rather than convenient is what happens when
the conversation does *not* match. Every model turn is keyed by a hash of exactly
what the model was shown - the messages and the tool schemas - so editing the
system prompt, a tool description, or the dataset changes the key and the
recording misses. A miss is reported as an error, which the runner counts as an
errored attempt and excludes from the rates. It is never silently served from a
recording made under different inputs, because a replay that quietly answers the
wrong question is worse than no replay: it would let a prompt change look free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .providers import Completion, ToolCall


class StaleRecording(Exception):
    """Raised by `require_fresh` when a replay misses more than it is allowed to."""


class ReplayProvider:
    """Serves model turns out of a recorded artifact, keyed by request hash."""

    def __init__(self, artifact_path: str | Path, model_suffix: str = " (replayed)") -> None:
        self.path = Path(artifact_path)
        artifact = json.loads(self.path.read_text(encoding="utf-8"))
        self.name = "replay"
        self.recorded_model = str(artifact.get("meta", {}).get("model", "unknown"))
        self.model = self.recorded_model + model_suffix
        self.recorded_meta: dict[str, Any] = artifact.get("meta", {})

        self._turns: dict[str, Completion] = {}
        self._no_hash = 0
        for attempt in artifact.get("attempts", []):
            for step in attempt.get("run", {}).get("steps", []):
                if step.get("kind") not in {"model", "review"}:
                    continue
                key = step.get("request_hash")
                if not key:
                    # Recorded before turns were hashed. Counted and reported
                    # rather than guessed at by position: replaying by index
                    # would silently drift the moment a run took a different
                    # number of steps.
                    self._no_hash += 1
                    continue
                self._turns.setdefault(key, _completion_from(step))

        self.hits = 0
        self.misses: list[str] = []

    # -- the provider interface ----------------------------------------

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        # Replaying costs nothing. The recorded run's own cost is in its summary;
        # reporting it again here would double-count it.
        return 0.0

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Completion:
        from .agent import request_hash

        key = request_hash(messages, tools)
        found = self._turns.get(key)
        if found is None:
            self.misses.append(key)
            return Completion(
                error=(
                    "no recorded turn for this conversation (" + key + "). The "
                    "recording in " + self.path.name + " was made under different "
                    "model inputs - a changed system prompt, tool schema or "
                    "dataset - so it cannot answer this. Re-record against the "
                    "model."
                )
            )
        self.hits += 1
        return found

    # -- reporting -----------------------------------------------------

    @property
    def coverage(self) -> float:
        total = self.hits + len(self.misses)
        return (self.hits / total) if total else 0.0

    def describe(self) -> str:
        parts = [
            "replayed " + str(self.hits) + " turns from " + str(self.path),
            "recorded against " + self.recorded_model,
        ]
        if self.misses:
            parts.append(str(len(self.misses)) + " turns had no recording")
        if self._no_hash:
            parts.append(
                str(self._no_hash) + " recorded turns carry no request hash and "
                "were skipped (recorded before hashing existed)"
            )
        return "; ".join(parts)

    def require_fresh(self) -> None:
        """Fail loudly if anything was missing.

        Used by CI: a replay that half matched is not a cheaper measurement, it
        is a measurement of a different thing, and a gate must not pass on one.
        """
        if self.misses:
            raise StaleRecording(
                str(len(self.misses)) + " of " + str(self.hits + len(self.misses))
                + " model turns had no recording in " + str(self.path)
                + ". The recording predates the current prompt, tools or dataset."
            )


def _completion_from(step: dict[str, Any]) -> Completion:
    calls = [
        ToolCall(
            name=call.get("name") or "",
            arguments=call.get("arguments") or {},
            raw_arguments=call.get("raw") or "",
            parse_error=call.get("parse_error"),
        )
        for call in step.get("tool_calls") or []
    ]
    return Completion(
        text=step.get("text") or "",
        tool_calls=calls,
        prompt_tokens=int(step.get("prompt_tokens") or 0),
        completion_tokens=int(step.get("completion_tokens") or 0),
        latency_ms=int(step.get("latency_ms") or 0),
        error=step.get("error"),
    )
