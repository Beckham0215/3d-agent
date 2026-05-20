"""Evaluate activity-to-location mapping (pure Python, no external API)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from eval.datasets import ACTIVITY_MAP_CASES


@dataclass
class ActivityMapResult:
    name: str = "activity_map"
    total: int = 0
    correct: int = 0
    overall_accuracy: float = 0.0
    cases: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def run() -> ActivityMapResult:
    from app.services.groq_service import get_location_for_activity

    cases_out: List[Dict[str, Any]] = []
    correct = 0

    for case in ACTIVITY_MAP_CASES:
        predicted = get_location_for_activity(case.activity)
        ok = predicted == case.expected_location
        if ok:
            correct += 1
        cases_out.append({
            "activity": case.activity,
            "expected": case.expected_location,
            "predicted": predicted,
            "pass": ok,
        })

    total = len(ACTIVITY_MAP_CASES)
    return ActivityMapResult(
        total=total,
        correct=correct,
        overall_accuracy=correct / total if total else 0.0,
        cases=cases_out,
    )
