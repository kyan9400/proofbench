"""Proofbench: an evaluation platform for tool-using LLM agents.

The shape of it:

    suites/        cases, as YAML
    shopdesk       the business the agent works for, with real invariants
    agent.py       the agent under test, and an optional reviewer in front of it
    tools.py       the tools, their schemas, and strict argument checking
    graders/       deterministic graders first; a measured judge second
    runner.py      drives a suite, repeats each case, writes an artifact
    report.py      rates with confidence intervals, flake detection, a CI gate
"""

__version__ = "0.1.0"
