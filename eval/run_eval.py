"""3DAgent Evaluation Runner.

Usage
-----
  # Unit tests only (no external API calls)
  python eval/run_eval.py

  # Include Groq API-dependent tests
  python eval/run_eval.py --integration

  # Run a single evaluator
  python eval/run_eval.py --only label_match

  # Write the HTML report to a custom path
  python eval/run_eval.py --output my_report.html

  # Skip HTML report (console output only)
  python eval/run_eval.py --no-report

Available evaluators
--------------------
  Unit (always available):
    label_match   - resolve_asset() fuzzy label matching
    vision_parser - parse_groq_vision_response() + _parse_blip_detection_response()
    activity_map  - get_location_for_activity() dict lookup
    scan_diff     - _compute_scan_diff() change detection
    latency       - Flask test-client API endpoint latency

  Integration (require GROQ_API_KEY):
    intent        - route_intent() 8-class intent classification
    react_parser  - parse_react_request() planning query extraction
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── colour helpers (ANSI) ─────────────────────────────────────────────────────

_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _col(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _status(score: float) -> str:
    if score >= 0.9:
        return _col("PASS", _GREEN)
    if score >= 0.7:
        return _col("WARN", _YELLOW)
    return _col("FAIL", _RED)


def _bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {score * 100:.1f}%"


# ── runner ────────────────────────────────────────────────────────────────────

UNIT_EVALUATORS = ["label_match", "vision_parser", "activity_map", "scan_diff", "latency"]
INTEGRATION_EVALUATORS = ["intent", "react_parser"]


def run_evaluator(name: str):
    if name == "label_match":
        from eval.evaluators.label_match import run
    elif name == "vision_parser":
        from eval.evaluators.vision_parser import run
    elif name == "activity_map":
        from eval.evaluators.activity_map import run
    elif name == "scan_diff":
        from eval.evaluators.scan_diff import run
    elif name == "intent":
        from eval.evaluators.intent import run
    elif name == "react_parser":
        from eval.evaluators.react_parser import run
    elif name == "latency":
        from eval.evaluators.latency import run
    else:
        raise ValueError(f"Unknown evaluator: {name!r}")
    return run()


def _print_result(name: str, result, elapsed: float) -> None:
    print(f"\n{_col('-' * 60, _CYAN)}")
    print(f"  {_col(name.upper().replace('_', ' '), _BOLD)}  [{elapsed:.1f}s]")
    print(_col("-" * 60, _CYAN))

    if hasattr(result, "error") and result.error:
        print(f"  {_col('SKIPPED', _YELLOW)}: {result.error}")
        return

    # ── label_match ──
    if name == "label_match":
        print(f"  Overall accuracy : {_bar(result.overall_accuracy)}"
              f"  {_status(result.overall_accuracy)}")
        print(f"  Correct / Total  : {result.correct}/{result.total}")
        print()
        for cat, v in result.by_category.items():
            print(f"    {cat:<20} {_bar(v['accuracy'], 12)}")

    # ── vision_parser ──
    elif name == "vision_parser":
        print(f"  Groq (Llama 4 Scout) : {_bar(result.groq_accuracy)}"
              f"  {_status(result.groq_accuracy)}"
              f"  ({result.groq_correct}/{result.groq_total})")
        print(f"  BLIP fallback        : {_bar(result.blip_accuracy)}"
              f"  {_status(result.blip_accuracy)}"
              f"  ({result.blip_correct}/{result.blip_total})")
        print(f"  Combined             : {_bar(result.overall_accuracy)}"
              f"  {_status(result.overall_accuracy)}")

    # ── activity_map ──
    elif name == "activity_map":
        print(f"  Accuracy         : {_bar(result.overall_accuracy)}"
              f"  {_status(result.overall_accuracy)}")
        print(f"  Correct / Total  : {result.correct}/{result.total}")

    # ── scan_diff ──
    elif name == "scan_diff":
        print(f"  Accuracy         : {_bar(result.overall_accuracy)}"
              f"  {_status(result.overall_accuracy)}")
        print(f"  Correct / Total  : {result.correct}/{result.total}")

    # ── intent ──
    elif name == "intent":
        print(f"  Accuracy         : {_bar(result.overall_accuracy)}"
              f"  {_status(result.overall_accuracy)}")
        print(f"  Macro F1         : {_bar(result.macro_f1)}")
        print(f"  Correct / Total  : {result.correct}/{result.total}")
        print(f"  Avg latency      : {result.avg_latency_ms:.0f} ms")
        print()
        header = f"  {'Intent':<18} {'Prec':>6} {'Rec':>6} {'F1':>6}"
        print(header)
        print("  " + "-" * 38)
        for cls, v in result.per_class.items():
            print(f"  {cls:<18} {v['precision']:>6.2f} {v['recall']:>6.2f} {v['f1']:>6.2f}")

    # ── react_parser ──
    elif name == "react_parser":
        print(f"  Asset accuracy   : {_bar(result.asset_accuracy)}"
              f"  {_status(result.asset_accuracy)}")
        print(f"  Count MAE        : {result.count_mae:.2f}")
        print(f"  Fully correct    : {result.exact_matches}/{result.total}")

    # ── latency ──
    elif name == "latency":
        header = f"  {'Endpoint':<42} {'p50':>8} {'p95':>8} {'p99':>8} {'OK%':>6}"
        print(header)
        print("  " + "-" * 74)
        for ep in result.endpoints:
            short = ep.endpoint[-40:] if len(ep.endpoint) > 40 else ep.endpoint
            ok_flag = _col("100%", _GREEN) if ep.ok_rate == 1.0 else _col(f"{ep.ok_rate*100:.0f}%", _RED)
            print(f"  {short:<42} {ep.p50_ms:>7.1f}  {ep.p95_ms:>7.1f}  {ep.p99_ms:>7.1f}  {ok_flag}")

    # failed cases summary for unit evaluators
    if hasattr(result, "cases"):
        failed = [c for c in result.cases if not c.get("pass")]
        if failed:
            print(f"\n  {_col(f'{len(failed)} failing case(s):', _RED)}")
            for c in failed[:5]:
                # pick a sensible label for each evaluator
                label = (
                    c.get("message") or c.get("answer") or
                    c.get("activity") or c.get("request") or
                    str(c.get("current", c.get("input", "?")))
                )
                print(f"    • {label[:70]}")
            if len(failed) > 5:
                print(f"    … and {len(failed) - 5} more")


def _print_final_summary(results: dict, total_elapsed: float) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {_col('EVALUATION SUMMARY', _BOLD)}")
    print("=" * 60)

    score_map = {
        "label_match":  ("overall_accuracy", "Accuracy"),
        "vision_parser": ("overall_accuracy", "Accuracy"),
        "activity_map": ("overall_accuracy", "Accuracy"),
        "scan_diff":    ("overall_accuracy", "Accuracy"),
        "intent":       ("overall_accuracy", "Accuracy"),
        "react_parser": ("asset_accuracy",   "Asset Acc"),
    }

    for name, (attr, label) in score_map.items():
        r = results.get(name)
        if r is None:
            continue
        if hasattr(r, "error") and r.error:
            print(f"  {name:<18} {_col('SKIPPED', _YELLOW)}")
            continue
        score = getattr(r, attr, 0.0)
        print(f"  {name:<18} {_bar(score)}  {_status(score)}")

    lat = results.get("latency")
    if lat and not (hasattr(lat, "error") and lat.error):
        p50s = [e.p50_ms for e in lat.endpoints]
        avg = sum(p50s) / len(p50s) if p50s else 0
        print(f"  {'latency':<18} avg p50 = {avg:.1f} ms")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print(_col("=" * 60, _BOLD))


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run 3DAgent evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--integration", action="store_true",
        help="Also run Groq API-dependent evaluators (intent, react_parser)",
    )
    parser.add_argument(
        "--only", metavar="NAME",
        help="Run a single named evaluator",
    )
    parser.add_argument(
        "--output", metavar="FILE", default="eval_report.html",
        help="Path for the HTML report (default: eval_report.html)",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip HTML report generation",
    )
    args = parser.parse_args()

    to_run = UNIT_EVALUATORS[:]
    if args.integration:
        to_run += INTEGRATION_EVALUATORS
    if args.only:
        to_run = [args.only]

    print(_col("3DAgent Evaluation Suite", _BOLD))
    print(f"Evaluators: {', '.join(to_run)}")
    if args.integration:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            print(_col("GROQ_API_KEY detected — integration tests will run.", _GREEN))
        else:
            print(_col("GROQ_API_KEY not set — integration tests will be skipped.", _YELLOW))

    results: dict = {}
    total_start = time.perf_counter()

    for name in to_run:
        print(f"\nRunning {_col(name, _CYAN)} …", end="", flush=True)
        t0 = time.perf_counter()
        try:
            result = run_evaluator(name)
        except Exception as exc:
            import traceback
            print(f"\n  {_col('ERROR', _RED)}: {exc}")
            traceback.print_exc()
            result = None
        elapsed = time.perf_counter() - t0
        print(f" done ({elapsed:.1f}s)")
        if result is not None:
            results[name] = result
            _print_result(name, result, elapsed)

    total_elapsed = time.perf_counter() - total_start
    _print_final_summary(results, total_elapsed)

    if not args.no_report:
        from eval.report import generate
        html = generate(results)
        out_path = os.path.abspath(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n{_col('HTML report saved to:', _GREEN)} {out_path}")


if __name__ == "__main__":
    main()
