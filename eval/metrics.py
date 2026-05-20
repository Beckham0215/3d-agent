"""Metric computation utilities for 3DAgent evaluation."""

from typing import List, Dict, Optional


def accuracy(predictions: List, ground_truth: List) -> float:
    if not ground_truth:
        return 0.0
    return sum(p == g for p, g in zip(predictions, ground_truth)) / len(ground_truth)


def mean_absolute_error(predictions: List[int], ground_truth: List[int]) -> float:
    if not ground_truth:
        return 0.0
    return sum(abs(p - g) for p, g in zip(predictions, ground_truth)) / len(ground_truth)


def precision_recall_f1_per_class(
    predictions: List[str],
    ground_truth: List[str],
    classes: List[str],
) -> Dict[str, Dict[str, float]]:
    """Return per-class precision, recall, F1 and raw counts."""
    results: Dict[str, Dict[str, float]] = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != cls and g == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[cls] = {"precision": prec, "recall": rec, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn}
    return results


def macro_f1(per_class: Dict[str, Dict[str, float]]) -> float:
    scores = [v["f1"] for v in per_class.values()]
    return sum(scores) / len(scores) if scores else 0.0


def confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    classes: List[str],
) -> Dict[str, Dict[str, int]]:
    """cm[actual][predicted] = count."""
    cm: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for pred, actual in zip(predictions, ground_truth):
        if actual in cm and pred in cm[actual]:
            cm[actual][pred] += 1
    return cm


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (p / 100.0) * (len(sv) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sv) - 1)
    return sv[lo] * (1 - (idx - lo)) + sv[hi] * (idx - lo)


def score_badge(score: float) -> str:
    """Return a CSS class name based on a 0-1 score."""
    if score >= 0.9:
        return "pass"
    if score >= 0.7:
        return "warn"
    return "fail"
