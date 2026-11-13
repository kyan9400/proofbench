"""Shopdesk: the small business the agent under test works for.

It is a real HTTP service with real invariants, not a mock. The agent reaches it
only through its tools, and every mutation is recorded so a grader can ask what
the agent actually did rather than what it said it did.
"""
