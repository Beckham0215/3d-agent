"""Evaluate vision response parsers (pure Python, no external API).

Tests two parsers:
  1. parse_groq_vision_response()  — primary path (Llama 4 Scout output)
  2. _parse_blip_detection_response() — fallback path (BLIP free-form text)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from eval.datasets import GROQ_VISION_PARSE_CASES, BLIP_PARSE_CASES
from eval.metrics import accuracy


@dataclass
class VisionParserResult:
    name: str = "vision_parser"
    # Groq (Llama 4 Scout) parser
    groq_total: int = 0
    groq_correct: int = 0
    groq_accuracy: float = 0.0
    groq_cases: List[Dict[str, Any]] = field(default_factory=list)
    # BLIP fallback parser
    blip_total: int = 0
    blip_correct: int = 0
    blip_accuracy: float = 0.0
    blip_cases: List[Dict[str, Any]] = field(default_factory=list)
    # Combined
    overall_accuracy: float = 0.0
    error: Optional[str] = None


def run() -> VisionParserResult:
    from flask import Flask
    from app.services.groq_service import parse_groq_vision_response
    from app.routes.api import _parse_blip_detection_response

    # Minimal app context so current_app.logger inside _parse_blip_detection_response works.
    app = Flask(__name__)
    app.config["GROQ_API_KEY"] = ""  # no API calls made here

    # ── Groq vision parser (parse_groq_vision_response is pure — no context needed) ──
    groq_cases_out: List[Dict[str, Any]] = []
    groq_correct = 0

    for case in GROQ_VISION_PARSE_CASES:
        predicted = parse_groq_vision_response(case.raw_response)
        ok = predicted == case.expected_counts
        if ok:
            groq_correct += 1
        groq_cases_out.append({
            "input":       case.raw_response,
            "expected":    case.expected_counts,
            "predicted":   predicted,
            "pass":        ok,
            "description": case.description,
        })

    groq_total = len(GROQ_VISION_PARSE_CASES)
    groq_acc = groq_correct / groq_total if groq_total else 0.0

    # ── BLIP fallback parser (uses current_app.logger — needs app context) ───
    blip_cases_out: List[Dict[str, Any]] = []
    blip_correct = 0

    with app.app_context():
        for case in BLIP_PARSE_CASES:
            predicted = _parse_blip_detection_response(case.response)
            ok = predicted == case.expected_counts
            if ok:
                blip_correct += 1
            blip_cases_out.append({
                "input":       case.response,
                "expected":    case.expected_counts,
                "predicted":   predicted,
                "pass":        ok,
                "description": case.description,
            })

    blip_total = len(BLIP_PARSE_CASES)
    blip_acc = blip_correct / blip_total if blip_total else 0.0

    # Combined
    total = groq_total + blip_total
    correct = groq_correct + blip_correct
    overall = correct / total if total else 0.0

    return VisionParserResult(
        groq_total=groq_total,
        groq_correct=groq_correct,
        groq_accuracy=groq_acc,
        groq_cases=groq_cases_out,
        blip_total=blip_total,
        blip_correct=blip_correct,
        blip_accuracy=blip_acc,
        blip_cases=blip_cases_out,
        overall_accuracy=overall,
    )
