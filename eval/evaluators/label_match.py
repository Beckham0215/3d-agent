"""Evaluate label resolution accuracy (pure Python, no external API)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from eval.datasets import LABEL_MATCH_CASES
from eval.metrics import accuracy


class _MockAsset:
    """Minimal stand-in satisfying resolve_asset's .label_name interface."""
    def __init__(self, label_name: str):
        self.label_name = label_name


@dataclass
class LabelMatchResult:
    name: str = "label_match"
    total: int = 0
    correct: int = 0
    overall_accuracy: float = 0.0
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def run() -> LabelMatchResult:
    from app.label_match import resolve_asset

    cases_out: List[Dict[str, Any]] = []
    correct = 0
    cat_stats: Dict[str, Dict[str, int]] = {}

    for case in LABEL_MATCH_CASES:
        assets = [_MockAsset(lbl) for lbl in case.available_labels]
        result = resolve_asset(assets, case.label_guess)
        predicted = result.label_name if result else None
        ok = predicted == case.expected_match
        if ok:
            correct += 1

        cat = case.category
        if cat not in cat_stats:
            cat_stats[cat] = {"correct": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        if ok:
            cat_stats[cat]["correct"] += 1

        cases_out.append({
            "input": case.label_guess,
            "available": case.available_labels,
            "expected": case.expected_match,
            "predicted": predicted,
            "pass": ok,
            "category": cat,
        })

    total = len(LABEL_MATCH_CASES)
    by_cat: Dict[str, Dict[str, Any]] = {}
    for cat, s in cat_stats.items():
        by_cat[cat] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": s["correct"] / s["total"] if s["total"] else 0.0,
        }

    return LabelMatchResult(
        total=total,
        correct=correct,
        overall_accuracy=correct / total if total else 0.0,
        by_category=by_cat,
        cases=cases_out,
    )
