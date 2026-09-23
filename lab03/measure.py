from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from bench import Bench, measured, read_text, unknown


WARMUP_TOL = 0.5
MIN_SAMPLES_ABOVE = 5
PERCENTILES = (50, 95, 99)
MULTIMODAL_GAP_RATIO = 20.0
MIN_MODE_FRACTION = 0.10
MIN_SAMPLES_FOR_MODALITY = 20
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"
POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)
GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
    "sys/devices/platform/17000000.gpu/load",
    "sys/devices/platform/bus@0/17000000.gpu/load",
)
CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


def _valid(samples: list[float]) -> bool:
    """Latencies must be finite, nonnegative numbers."""
    return all(isinstance(x, (int, float)) and not isinstance(x, bool)
               and math.isfinite(x) and x >= 0 for x in samples)


def _number(root: Path, path: str) -> float | None:
    """Read a numeric sensor without inventing a missing value."""
    try:
        raw = read_text(root, path)
        value = float(raw)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        # Some Jetson thermal sensors return no data to Python's reader.
        return None


def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    """Time complete workload executions in milliseconds."""

    # 1. Check the requested number of runs.
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0:
        raise ValueError("repeats must be a nonnegative integer")

    samples = []

    # 2. Finish earlier work before starting each timer.
    for _ in range(repeats):
        bench.workload.synchronize()
        start = bench.clock()

        # 3. Wait for this run to finish before stopping the timer.
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()

        # 4. Convert nanoseconds to milliseconds.
        elapsed = (end - start) / 1_000_000
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("clock returned an invalid elapsed time")
        samples.append(elapsed)

    return samples


def is_stationary(samples: list[float]) -> dict[str, Any]:
    """Compare early and late medians to flag a drifting run."""
    source = "first-third vs last-third median, relative to whole-run median"

    if not _valid(samples) or len(samples) < MIN_SAMPLES_FOR_STATIONARITY:
        return unknown(source, "need at least 12 valid samples to assess drift")

    third = len(samples) // 3
    first = statistics.median(samples[:third])
    last = statistics.median(samples[-third:])
    baseline = statistics.median(samples)
    difference = abs(last - first)
    drift = difference / baseline if baseline else (0.0 if difference == 0 else None)

    return measured(
        drift is not None and drift <= STATIONARITY_TOL, source,
        first_median_ms=first, last_median_ms=last,
        relative_drift=drift, tolerance=STATIONARITY_TOL,
    )


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    """Discard only a slow leading prefix, and only with a stable tail."""
    source = "leading prefix above (1 + 0.5) x median of the run's second half"

    # 1. Use the second half as the possible settled baseline.
    if not samples or not _valid(samples):
        return unknown(source, "need nonempty, finite, nonnegative samples")

    tail = samples[len(samples) // 2:]
    rate = statistics.median(tail)
    threshold = (1 + WARMUP_TOL) * rate

    # 2. Keep everything if the tail never establishes a stable baseline.
    stationary = is_stationary(tail)
    boundary = 0
    if stationary["value"] is True:
        while boundary < len(samples) and samples[boundary] > threshold:
            boundary += 1

    # 3. Record how much data remains.
    result = measured(
        boundary, source, settled_rate_ms=round(rate, 4),
        threshold_ms=round(threshold, 4), tolerance=WARMUP_TOL,
        retained=len(samples) - boundary,
    )
    if stationary["value"] is not True:
        result["detail"] = "no confirmed steady tail; no samples discarded"
    return result


def summarize(samples: list[float]) -> dict[str, Any]:
    """Summarize samples with linear interpolation between sorted values."""

    # 1. Reject missing or invalid measurements.
    if not samples or not _valid(samples):
        return unknown("latency samples", "need nonempty, finite, nonnegative samples")
    values = sorted(samples)
    n = len(values)

    # 2. Use sample standard deviation; one observation cannot estimate it.
    result = {
        "n": n,
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if n > 1 else None,
        "min": round(values[0], 4),
        "max": round(values[-1], 4),
    }

    # 3. Interpolate at position (n - 1) times the requested fraction.
    for percentile in PERCENTILES:
        position = (n - 1) * percentile / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        value = values[lower] + fraction * (values[upper] - values[lower])
        result[f"p{percentile}"] = round(value, 4)

    return result


def is_multimodal(samples: list[float]) -> dict[str, Any]:
    """Look for a large gap with enough observations on both sides."""
    source = (
        "widest trimmed gap >= 20.0x the median gap, "
        "with >= 10% of samples on each side"
    )

    # 1. Work on sorted, already warm-up-trimmed measurements.
    if not _valid(samples) or len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown(source, "need at least 20 valid retained samples")
    values = sorted(samples)
    n = len(values)

    # 2. Only consider gaps leaving at least 10% on each side.
    minimum = math.ceil(n * MIN_MODE_FRACTION)
    splits = range(minimum, n - minimum + 1)
    split = max(splits, key=lambda i: values[i] - values[i - 1])
    widest = values[split] - values[split - 1]
    gaps = [values[i] - values[i - 1] for i in range(minimum, n - minimum + 1)]
    typical = statistics.median(gaps)

    # 3. Handle identical measurements without dividing by zero.
    ratio = widest / typical if typical else (0.0 if widest == 0 else None)
    multimodal = widest > 0 and (typical == 0 or ratio >= MULTIMODAL_GAP_RATIO)
    modes = [
        {"n": len(group), "share": len(group) / n,
         "median_ms": round(statistics.median(group), 4)}
        for group in (values[:split], values[split:])
    ]

    return measured(
        multimodal, source,
        gap_ratio=round(ratio, 2) if ratio is not None else None,
        widest_gap_ms=round(widest, 5), typical_gap_ms=round(typical, 5),
        modes=modes,
    )


def probe_power_state(bench: Bench) -> dict[str, Any]:
    """Read the power profile and use CPU0 limits as a clock-pinning hint."""

    # 1. Ask nvpmodel for the active profile, without changing it.
    source = "nvpmodel -q"
    reply = bench.runner(["nvpmodel", "-q"])
    match = re.search(r"^NV Power Mode:[ \t]*([^\r\n]+)\r?\n\s*(\d+)\s*$",
                      reply.stdout, re.M) if reply.ok and reply.returncode == 0 else None
    result = (measured(match.group(1).strip(), source, mode_index=int(match.group(2)))
              if match else unknown(source, reply.error or "power profile query failed or was not parseable"))

    # 2. Compare the CPU minimum and maximum scaling limits.
    low = _number(bench.telemetry, CPUFREQ_MIN)
    high = _number(bench.telemetry, CPUFREQ_MAX)
    clock_source = CPUFREQ_MIN + " vs " + CPUFREQ_MAX

    if low is None or high is None or low <= 0 or high < low:
        result["jetson_clocks"] = None
        result["jetson_clocks_source"] = unknown(clock_source, "CPU frequency limits unavailable or invalid")
    else:
        result["jetson_clocks"] = low == high
        result["jetson_clocks_source"] = measured(
            f"scaling_min_freq={low:.0f}, scaling_max_freq={high:.0f}", clock_source
        )

    return result


def probe_telemetry(bench: Bench) -> dict[str, Any]:
    """Read temperatures, board input power, and GPU load without changing them."""
    root = Path(bench.telemetry)

    # 1. Read available thermal zones and convert millidegrees to degrees C.
    zones = []
    for directory in (THERMAL_ZONES, "sys/class/thermal"):
        for zone in sorted((root / directory).glob("thermal_zone*")):
            relative = zone.relative_to(root).as_posix()
            value = _number(root, relative + "/temp")
            if value is not None:
                name = read_text(root, relative + "/type") or zone.name
                zones.append((value / 1000, name))
        if zones:
            break
    thermal_source = directory + "/*/temp"
    if zones:
        peak, name = max(zones)
        temperature = measured(round(peak, 2), thermal_source, zone=name, zones_read=len(zones))
    else:
        temperature = unknown(thermal_source, "no readable thermal zone temperatures")

    # 2. Legacy INA3221 power nodes report milliwatts directly.
    power = unknown(" | ".join(POWER_RAIL_CANDIDATES),
                    "no readable board power rail with known units")
    for path in POWER_RAIL_CANDIDATES:
        if "/in_power" not in path:
            continue
        value = _number(root, path)
        if value is not None and value >= 0:
            power = measured(value, path)
            break

    # 3. Modern hwmon in*_input is VOLTAGE, not power.
    #    Combine board-input millivolts and milliamps to obtain milliwatts.
    if power["status"] == "unknown":
        for directory in sorted((root / "sys/bus/i2c/drivers/ina3221/1-0040/hwmon").glob("hwmon*")):
            for label_path in sorted(directory.glob("in*_label")):
                label_rel = label_path.relative_to(root).as_posix()
                label = read_text(root, label_rel)
                if label not in ("VDD_IN", "VIN_SYS_5V0"):
                    continue
                channel = label_path.name[2:-6]
                voltage_path = directory.relative_to(root).as_posix() + f"/in{channel}_input"
                current_path = directory.relative_to(root).as_posix() + f"/curr{channel}_input"
                voltage = _number(root, voltage_path)
                current = _number(root, current_path)
                if voltage is not None and current is not None and voltage >= 0 and current >= 0:
                    power = measured(voltage * current / 1000,
                                     voltage_path + " * " + current_path + " / 1000")
                    break
            if power["status"] == "ok":
                break

    # 4. GPU load is in per-mille, so divide by ten for a percentage.
    gpu = unknown(" | ".join(GPU_LOAD_CANDIDATES), "no valid GPU load reading")
    for path in GPU_LOAD_CANDIDATES:
        value = _number(root, path)
        if value is not None and 0 <= value <= 1000:
            gpu = measured(value / 10, path, units="per-mille / 10")
            break

    return {"temperature_c": temperature, "power_mw": power,
            "gpu_utilization_percent": gpu}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Jetson inference latency")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    # 1. Construct the real workload. Never silently replace GPU timing with CPU timing.
    try:
        bench = Bench.real(device=args.device)
    except ImportError as error:
        parser.exit(1, f"Benchmark unavailable: {error}. Activate the course PyTorch/torchvision environment first.\n")
    samples = run_timed_iterations(bench, args.repeats)

    # 2. Preserve raw samples and analyze the settled portion separately.
    warmup = find_warmup_boundary(samples)
    retained = samples[warmup["value"] or 0:]
    report = {
        "warmup_boundary": warmup,
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(retained),
        "probe_power_state": probe_power_state(bench),
        "probe_telemetry": probe_telemetry(bench),
    }
    analysis = {
        "device": bench.device, "model": bench.model, "precision": bench.precision,
        "input_shape": list(bench.input_shape), "batch_size": bench.batch_size,
        "timing_boundary": "input already on device; synchronized forward pass",
        "retained_summary": summarize(retained),
        "stationarity": is_stationary(retained),
        "percentile_support": {
            f"p{p}": sum(x > _quantile_unrounded(retained, p) for x in retained) >= MIN_SAMPLES_ABOVE
            for p in PERCENTILES
        },
    }

    # 3. Save real measurements; simulated tests never write these reports.
    for name, data in (("samples_analysis.json", samples),
                       ("system_report.json", report),
                       ("benchmark_analysis.json", analysis)):
        Path(name).write_text(json.dumps(data, indent=4, allow_nan=False) + "\n", encoding="utf-8")


def _quantile_unrounded(samples: list[float], percentile: int) -> float:
    values = sorted(samples)
    position = (len(values) - 1) * percentile / 100
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


if __name__ == "__main__":
    main()
