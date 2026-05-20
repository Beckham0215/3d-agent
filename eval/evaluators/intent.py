"""Evaluate intent classification accuracy (requires GROQ_API_KEY).

Runs each test message through route_intent and compares the returned
intent label against the ground truth.  Computes per-class precision /
recall / F1 and a macro-F1 summary.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from eval.datasets import INTENT_CASES, INTENT_CLASSES
from eval.metrics import (
    accuracy,
    precision_recall_f1_per_class,
    macro_f1,
    confusion_matrix,
)


@dataclass
class IntentResult:
    name: str = "intent"
    total: int = 0
    correct: int = 0
    overall_accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    conf_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    cases: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def run() -> IntentResult:
    """Call Groq API for each test case and evaluate intent routing."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return IntentResult(
            error="GROQ_API_KEY not set — skipping intent evaluation."
        )

    from flask import Flask
    from app.services.groq_service import route_intent

    app = Flask(__name__)
    app.config["GROQ_API_KEY"] = api_key
    app.config["GROQ_MODEL"] = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    predictions: List[str] = []
    ground_truth: List[str] = []
    cases_out: List[Dict[str, Any]] = []
    total_latency = 0.0

    with app.app_context():
        for case in INTENT_CASES:
            try:
                t0 = time.perf_counter()
                result = route_intent(case.message, case.labels)
                elapsed_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                result = {"intent": "conversational"}
                elapsed_ms = 0.0
                cases_out.append({
                    "message": case.message,
                    "expected": case.expected_intent,
                    "predicted": "ERROR",
                    "pass": False,
                    "latency_ms": 0.0,
                    "error": str(exc),
                })
                predictions.append("conversational")
                ground_truth.append(case.expected_intent)
                continue

            predicted = result.get("intent", "conversational")
            ok = predicted == case.expected_intent
            total_latency += elapsed_ms
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            cases_out.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "pass": ok,
                "latency_ms": round(elapsed_ms, 1),
                "destination_label": result.get("destination_label"),
                "asset_name": result.get("asset_name"),
                "query_area": result.get("query_area"),
            })

    total = len(INTENT_CASES)
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    per_cls = precision_recall_f1_per_class(predictions, ground_truth, INTENT_CLASSES)
    mf1 = macro_f1(per_cls)
    cm = confusion_matrix(predictions, ground_truth, INTENT_CLASSES)

    return IntentResult(
        total=total,
        correct=correct,
        overall_accuracy=correct / total if total else 0.0,
        macro_f1=mf1,
        per_class=per_cls,
        conf_matrix=cm,
        avg_latency_ms=total_latency / total if total else 0.0,
        cases=cases_out,
    )
