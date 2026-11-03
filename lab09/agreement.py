
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    """Summarize finite, nonnegative durations in milliseconds."""
    samples = [_finite(value, f"ms[{i}]") for i, value in enumerate(ms)]
    if not samples:
        raise AgreementError("no timing samples.")
    if any(value < 0 for value in samples):
        raise AgreementError("durations must be nonnegative.")

    report = {"n": len(samples), "mean": sum(samples) / len(samples),
              "worst": max(samples)}
    for percentile in LATENCY_PERCENTILES:
        report[f"p{percentile}"] = _percentile(samples, percentile)
    report["jitter"] = report["p95"] - report["p50"]
    return report


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    """Compare positive durations; negative savings mean a slowdown."""
    base = _finite(base_ms, "base_ms")
    other = _finite(other_ms, "other_ms")
    if base <= 0 or other <= 0:
        raise AgreementError("speedup requires positive durations.")

    factor = base / other
    return {"factor": factor, "faster": factor > 1,
            "saved_ms": base - other}


def _paired_values(a: Sequence[float], b: Sequence[float]):
    """Validate every entry before comparing equally sized outputs."""
    if not a or len(a) != len(b):
        raise AgreementError("output sequences must be nonempty and equal in length.")
    left = [_finite(v, f"a[{i}]") for i, v in enumerate(a)]
    right = [_finite(v, f"b[{i}]") for i, v in enumerate(b)]
    return left, right


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Find the largest absolute error, keeping the first index on ties."""
    left, right = _paired_values(a, b)
    errors = [abs(x - y) for x, y in zip(left, right)]
    index = max(range(len(errors)), key=errors.__getitem__)
    return {"max_abs": errors[index], "index": index,
            "a": left[index], "b": right[index], "n": len(left)}


def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    """Measure error relative to reference b and disclose clamped denominators."""
    floor = _finite(floor, "floor")
    if floor <= 0:
        raise AgreementError("relative-error floor must be positive.")
    left, right = _paired_values(a, b)
    magnitudes = [abs(value) for value in right]
    errors = [abs(x - y) / max(magnitude, floor)
              for x, y, magnitude in zip(left, right, magnitudes)]
    index = max(range(len(errors)), key=errors.__getitem__)
    return {"max_rel": errors[index], "index": index,
            "floored": sum(value < floor for value in magnitudes),
            "denom_min": min(magnitudes), "n": len(left)}


def _score_rows(rows: Sequence[Sequence[float]], name: str):
    """Reject missing rows, empty classes, and non-finite scores."""
    if not rows:
        raise AgreementError(f"{name} has no input rows.")
    validated = []
    for i, row in enumerate(rows):
        if not row:
            raise AgreementError(f"{name}[{i}] has no classes.")
        validated.append([_finite(v, f"{name}[{i}][{j}]")
                          for j, v in enumerate(row)])
    return validated


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Compare class indices for corresponding rows; first index wins ties."""
    left = _score_rows(rows_a, "rows_a")
    right = _score_rows(rows_b, "rows_b")
    if len(left) != len(right):
        raise AgreementError("backends must have equal row counts.")

    mismatches = []
    for i, (a, b) in enumerate(zip(left, right)):
        if len(a) != len(b):
            raise AgreementError(f"row {i} has different class counts.")
        if max(range(len(a)), key=a.__getitem__) != max(range(len(b)), key=b.__getitem__):
            mismatches.append(i)
    n = len(left)
    return {"n": n, "agree": n - len(mismatches),
            "disagree": len(mismatches), "rate": (n - len(mismatches)) / n,
            "first_disagreement": mismatches[0] if mismatches else None}


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    """Compare top-two gaps with the supplied tolerance (inclusive boundary)."""
    tol = _finite(tol, "tol")
    if tol < 0:
        raise AgreementError("tolerance must be nonnegative.")
    scores = _score_rows(rows, "rows")
    margins = []
    for row in scores:
        if len(row) < 2:
            raise AgreementError("a decision margin requires at least two classes.")
        first, second = sorted(row, reverse=True)[:2]
        margins.append(first - second)

    argmin = min(range(len(margins)), key=margins.__getitem__)
    below_tol = sum(margin <= tol for margin in margins)
    return {"n": len(margins), "min": margins[argmin],
            "median": _percentile(margins, 50),
            "mean": sum(margins) / len(margins), "argmin": argmin,
            "below_tol": below_tol, "safe": below_tol == 0}

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()