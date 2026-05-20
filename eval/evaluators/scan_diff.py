"""Evaluate scan change detection correctness (pure Python, no external API)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from eval.datasets import SCAN_DIFF_CASES


@dataclass
class ScanDiffResult:
    name: str = "scan_diff"
    total: int = 0
    correct: int = 0
    overall_accuracy: float = 0.0
    cases: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def run() -> ScanDiffResult:
    from app.routes.api import _compute_scan_diff

    cases_out: List[Dict[str, Any]] = []
    correct = 0

    for case in SCAN_DIFF_CASES:
        predicted = _compute_scan_diff(case.current, case.previous)
        # Compare as sorted-by-item lists so order doesn't matter
        predicted_sorted = sorted(predicted, key=lambda x: x["item"])
        expected_sorted  = sorted(case.expected_changes, key=lambda x: x["item"])
        ok = predicted_sorted == expected_sorted
        if ok:
            correct += 1
        cases_out.append({
            "current":  case.current,
            "previous": case.previous,
            "expected": expected_sorted,
            "predicted": predicted_sorted,
            "pass": ok,
        })

    total = len(SCAN_DIFF_CASES)
    return ScanDiffResult(
        total=total,
        correct=correct,
        overall_accuracy=correct / total if total else 0.0,
        cases=cases_out,
    )
