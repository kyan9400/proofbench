"""Graders.

Everything in `programmatic` decides without a model: it reads the effect log,
the tool calls, the reply text and the meters. `judge` (stage 2) adds the one
grader that does use a model, and reports its agreement with hand labels so a
reader can discount it appropriately.
"""

from .programmatic import (
    PROGRAMMATIC_GRADERS,
    GraderResult,
    diagnose,
    grade_programmatic,
)

__all__ = ["GraderResult", "PROGRAMMATIC_GRADERS", "diagnose", "grade_programmatic"]
