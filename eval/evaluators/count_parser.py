"""Evaluate BLIP answer count extraction (pure Python, no external API)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from eval.datasets import COUNT_PARSER_CASES
from eval.metrics import mean_absolute_error, accuracy


@dataclass
class CountParserResult:
    name: str = "count_parser"
    total: int = 0
    exact_matches: int = 0
    exact_match_rate: float = 0.0
    mae: float = 0.0
    cases: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def run() -> CountParserResult:
    from app.routes.api import _extract_count_from_answer

    preds: List[int] = []
    truth: List[int] = []
    cases_out: List[Dict[str, Any]] = []

    for case in COUNT_PARSER_CASES:
        predicted = _extract_count_from_answer(case.answer)
        ok = predicted == case.expected_count
        preds.append(predicted)
        truth.append(case.expected_count)
        cases_out.append({
            "answer": case.answer,
            "expected": case.expected_count,
            "predicted": predicted,
            "pass": ok,
        })

    total = len(COUNT_PARSER_CASES)
    exact = sum(1 for p, g in zip(preds, truth) if p == g)

    return CountParserResult(
        total=total,
        exact_matches=exact,
        exact_match_rate=exact / total if total else 0.0,
        mae=mean_absolute_error(preds, truth),
        cases=cases_out,
    )
