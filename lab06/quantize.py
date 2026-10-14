from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from pathlib import Path
import json

from tensors import (
    SCALE_BYTES,
    ZERO_BYTES,
    Sample,
    Tensor,
    TensorError,
    load_model,
    dtype_bytes,
    elements,
    total_parameters,
    materialise_samples,
)

from quantize_helper import (
    _channel_slice,
    _qmin_qmax,
    _range_of,
    _round_half_up,
    _scale_zero,
    CALIBRATIONS,
    CLIP_METHODS,
    DEFAULT_BITS,
    DEFAULT_PERCENTILE,
    GRANULARITIES,
    MSE_CANDIDATES,
    PRECISIONS,
)


# helper class and functions to carry the quantized tensor and its metadata around together
@dataclass(frozen=True)
class Quantized:
    name: str
    shape: tuple[int, ...]
    codes: tuple[int, ...]
    scales: tuple[float, ...]
    zeros: tuple[int, ...]
    bits: int
    granularity: str
    symmetric: bool
    clipped: int = 0

    @property
    def parameters(self) -> int:
        return elements(self.shape)

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def channel_stride(self) -> int:
        return elements(self.shape[1:]) if len(self.shape) > 1 else 1

    @property
    def scale_count(self) -> int:
        """How many (scale, zero) pairs this tensor carries on disk."""
        return len(self.scales)

    @property
    def clipped_fraction(self) -> float:
        return round(self.clipped / len(self.codes), 8) if self.codes else 0.0

def to_fp16(t: Tensor) -> Tensor:
    out = []
    for v in t.data:
        if not -65504.0 <= v <= 65504.0:
            raise TensorError(
                f"{t.name}: {v} is outside binary16's range; FP16 would store "
                "infinity and every number after this one would be a lie"
            )
        out.append(round(struct.unpack("e", struct.pack("e", float(v)))[0], 8))
    return Tensor(name=t.name, shape=t.shape, data=tuple(out), dtype="fp16")


def as_tensor(name: str, values: Sequence[float]) -> Tensor:
    vals = tuple(float(v) for v in values)
    if not vals:
        raise TensorError(f"{name}: no values")
    return Tensor(name=name, shape=(len(vals),), data=vals, dtype="fp32")


def select_calibration(activations: Sequence[Sample],
                       policy: str = "all") -> tuple[float, ...]:
    acts = tuple(activations)
    if policy == "all":
        chosen = acts
    elif policy == "two_class":
        chosen = tuple(s for s in acts if s.cls in (0, 1))
    elif policy == "first_8":
        chosen = acts[:8]
    elif policy == "one_class":
        chosen = tuple(s for s in acts if s.cls == 0)
    else:
        raise TensorError(
            f"unknown calibration policy {policy!r}; known: {list(CALIBRATIONS)}"
        )
    if not chosen:
        raise TensorError(f"calibration policy {policy!r} selected no samples")
    return tuple(v for s in chosen for v in s.data)


########################################
# start filling in the code from here.
#########################################

def quantize_tensor(t: Tensor, bits: int = DEFAULT_BITS,
                    granularity: str = "per_tensor", symmetric: bool = True,
                    clip: str = "minmax",
                    percentile: float = DEFAULT_PERCENTILE,
                    given_range: tuple[float, float] | None = None) -> Quantized:
    """Encode weights using one scale per tensor or output channel."""
    # 1. Validate the policy and choose the available integer codes.
    if granularity not in GRANULARITIES:
        raise TensorError(f"unsupported granularity: {granularity!r}")
    if clip not in CLIP_METHODS:
        raise TensorError(f"unsupported clipping method: {clip!r}")
    lower_code, upper_code = _qmin_qmax(bits, symmetric)
    slices = ([_channel_slice(t, channel) for channel in range(t.channels)]
              if granularity == "per_channel" else [(0, t.parameters)])


    # 2. Encode each group in order so the original tensor layout is preserved.
    encoded = []
    group_scales = []
    group_zeros = []
    overflow_count = 0
    for first, stop in slices:
        values = t.data[first:stop]
        limits = (tuple(float(v) for v in given_range) if given_range is not None
                  else _range_of(values, clip, percentile, bits))
        scale, zero = _scale_zero(*limits, bits, symmetric)
        group_scales.append(scale)
        group_zeros.append(zero)

        for value in values:
            candidate = _round_half_up(value / scale) + zero
            if candidate < lower_code:
                encoded.append(lower_code)
                overflow_count += 1
            elif candidate > upper_code:
                encoded.append(upper_code)
                overflow_count += 1
            else:
                encoded.append(candidate)


    # 3. Store the codes together with the metadata needed to decode them.
    return Quantized(
        name=t.name, shape=tuple(t.shape), codes=tuple(encoded),
        scales=tuple(group_scales), zeros=tuple(group_zeros), bits=int(bits),
        granularity=granularity, symmetric=bool(symmetric), clipped=overflow_count,
    )


def dequantize(qt: Quantized) -> Tensor:
    """Reconstruct floats from codes; rounding cannot restore lost precision."""
    # 1. Determine how many consecutive values share a scale.
    group_width = qt.channel_stride if qt.granularity == "per_channel" else len(qt.codes)
    if group_width <= 0:
        raise TensorError("quantized groups must contain values")


    # 2. Decode one group at a time with value = scale * (code - zero).
    reconstructed = []
    for first in range(0, len(qt.codes), group_width):
        group = first // group_width
        if group >= len(qt.scales) or group >= len(qt.zeros):
            raise TensorError(f"{qt.name}: missing scale or zero point for group {group}")
        scale, zero = qt.scales[group], qt.zeros[group]
        reconstructed.extend(
            round(scale * (int(code) - zero), 8)
            for code in qt.codes[first:first + group_width]
        )
    return Tensor(qt.name, qt.shape, tuple(reconstructed), "fp32")


def quantization_error(before: Sequence[Tensor],
                       after: Sequence[Tensor]) -> dict[str, float]:
    """Compare original and reconstructed values using three error measures."""
    # 1. Corresponding tensors must have matching numbers of values.
    original, restored = list(before), list(after)
    if len(original) != len(restored):
        raise TensorError("original and reconstructed tensor counts differ")
    squared_error = 0.0
    squared_signal = 0.0
    largest_error = 0.0


    # 2. Accumulate across the whole model, before taking a square root.
    for original_tensor, restored_tensor in zip(original, restored):
        if len(original_tensor.data) != len(restored_tensor.data):
            raise TensorError(f"{original_tensor.name}: tensor value counts differ")
        for value, approximation in zip(original_tensor.data, restored_tensor.data):
            difference = value - approximation
            squared_error += difference ** 2
            squared_signal += value ** 2
            if abs(difference) > largest_error:
                largest_error = abs(difference)


    # 3. Follow the handout's zero-signal convention and round the report.
    error_norm = math.sqrt(squared_error)
    relative_error = error_norm / math.sqrt(squared_signal) if squared_signal > 0 else 0.0
    return {"frobenius": round(error_norm, 8),
            "relative": round(relative_error, 8),
            "max_abs": round(largest_error, 8)}


def bytes_stored(tensors: Sequence[Tensor], precision: str = "fp32",
                 granularity: str = "per_tensor") -> int:
    """Account for the weight payload and INT8 scale/zero metadata."""
    # 1. Validate both choices even when a floating format needs no metadata.
    if precision not in PRECISIONS:
        raise TensorError(f"unsupported precision: {precision!r}")
    if granularity not in GRANULARITIES:
        raise TensorError(f"unsupported granularity: {granularity!r}")


    # 2. Floating formats need only weights; INT8 needs one pair per group.
    stored = list(tensors)
    payload = total_parameters(stored) * dtype_bytes(precision)
    metadata = 0
    if precision == "int8":
        pair_count = (sum(t.channels for t in stored)
                      if granularity == "per_channel" else len(stored))
        metadata = pair_count * (SCALE_BYTES + ZERO_BYTES)
    return int(payload + metadata)


def effective_bits(tensors: Sequence[Tensor], precision: str = "fp32",
                   granularity: str = "per_tensor") -> float:
    """Express total storage as average bits per model parameter."""
    stored = list(tensors)
    parameter_count = total_parameters(stored)
    if parameter_count <= 0:
        raise TensorError("effective bits requires at least one parameter")
    storage_bits = 8 * bytes_stored(stored, precision, granularity)
    return round(storage_bits / parameter_count, 4)


def precision_row(model_name: str, config: Mapping[str, Any],
                  tensors: Sequence[Tensor],
                  calibration_values: Sequence[float],
                  all_values: Sequence[float]) -> dict[str, Any]:
    """Evaluate one weight policy and its activation calibration range."""
    # 1. Require the label explicitly, as stated in the handout.
    raw_label = config.get("label")
    if raw_label is None or not str(raw_label).strip():
        raise TensorError("each configuration needs a nonempty label")
    label = str(raw_label)
    precision = str(config.get("precision", "fp32"))
    if precision not in PRECISIONS:
        raise TensorError(f"unsupported precision: {precision!r}")
    grouping = str(config.get("granularity", "per_tensor"))
    symmetric = bool(config.get("symmetric", True))
    clipping = str(config.get("clip", "minmax"))
    percentile = float(config.get("percentile", DEFAULT_PERCENTILE))
    calibration = str(config.get("calibration", "all"))
    original_weights = list(tensors)


    # 2. Reconstruct weights at the selected precision to measure their error.
    restored_weights = []
    scale_count = 0
    for weight in original_weights:
        if precision == "int8":
            encoded = quantize_tensor(weight, DEFAULT_BITS, grouping,
                                      symmetric, clipping, percentile)
            restored_weights.append(dequantize(encoded))
            scale_count += encoded.scale_count
        elif precision == "fp16":
            restored_weights.append(to_fp16(weight))
        else:
            restored_weights.append(weight)


    # 3. Learn the activation range on calibration data; evaluate all values.
    original_activations = as_tensor("activations", all_values)
    clipped_fraction = 0.0
    if precision == "int8":
        limits = _range_of(calibration_values, clipping, percentile, DEFAULT_BITS)
        encoded_activations = quantize_tensor(
            original_activations, bits=DEFAULT_BITS, granularity="per_tensor",
            symmetric=False, clip=clipping, percentile=percentile, given_range=limits,
        )
        restored_activations = dequantize(encoded_activations)
        clipped_fraction = encoded_activations.clipped_fraction
    else:
        limits = (min(original_activations.data), max(original_activations.data))
        restored_activations = (to_fp16(original_activations) if precision == "fp16"
                                else original_activations)


    # 4. Keep storage, weight error, and activation error in the same record.
    is_integer = precision == "int8"
    return {
        "model": str(model_name), "label": label, "precision": precision,
        "granularity": grouping if is_integer else "n/a",
        "symmetric": symmetric if is_integer else None,
        "calibration": calibration if is_integer else "not calibrated",
        "clip": clipping if is_integer else "n/a",
        "parameters": total_parameters(original_weights),
        "weight_error": quantization_error(original_weights, restored_weights)["relative"],
        "activation_error": quantization_error([original_activations], [restored_activations])["relative"],
        "clipped_fraction": clipped_fraction,
        "activation_range": [round(float(bound), 8) for bound in limits],
        "scale_count": scale_count,
        "bytes_stored": bytes_stored(original_weights, precision, grouping),
        "effective_bits": effective_bits(original_weights, precision, grouping),
    }


def pareto_front(points: Sequence[Mapping[str, Any]],
                 objectives: Mapping[str, float]) -> tuple[int, ...]:
    """Keep indices with no strictly dominating alternative; preserve ties."""
    # 1. Convert each objective to a score where higher always means better.
    candidates = list(points)
    if not candidates or not objectives:
        raise TensorError("Pareto comparison needs points and objectives")
    scores = []
    for index, point in enumerate(candidates):
        row = []
        for name, sense in objectives.items():
            if point.get(name) is None:
                raise TensorError(f"point {index} is missing objective {name!r}")
            row.append(float(point[name]) * float(sense))
        scores.append(tuple(row))


    # 2. An alternative must be no worse everywhere and better somewhere.
    survivors = []
    for index, candidate in enumerate(scores):
        dominated = False
        for other in scores:
            no_worse = all(b >= a for a, b in zip(candidate, other))
            strictly_better = any(b > a for a, b in zip(candidate, other))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            survivors.append(index)
    return tuple(survivors)


if __name__ == "__main__":
    json_path = Path("model.json")

    model_spec = load_model(json_path)
    model_tensors = model_spec["tensors"]
    model_name = model_spec["name"]
    total_params = total_parameters(model_tensors)

    test_tensor = model_tensors[0]  # conv1: shape (20, 3, 3, 3)

    raw_samples = materialise_samples(layer="conv3", width=40, count=32, seed=model_spec["seed"] + 100)
    all_act_vals = tuple(v for s in raw_samples for v in s.data)
    calib_all = select_calibration(raw_samples, policy="all")
    calib_one_cls = select_calibration(raw_samples, policy="one_class")

    # -----------------------------------------------------------------
    # 1. Test quantize_tensor
    # -----------------------------------------------------------------
    print("\n[1/7] Testing quantize_tensor...")
    q_pt = quantize_tensor(test_tensor, bits=8, granularity="per_tensor", symmetric=True)
    q_pc = quantize_tensor(test_tensor, bits=8, granularity="per_channel", symmetric=True)

    assert len(q_pt.codes) == test_tensor.parameters
    assert q_pt.scale_count == 1
    assert q_pc.scale_count == test_tensor.channels
    print(f"  ✓ Per-tensor quantized:  {len(q_pt.codes)} codes, {q_pt.scale_count} scale")
    print(f"  ✓ Per-channel quantized: {len(q_pc.codes)} codes, {q_pc.scale_count} scales")

    # -----------------------------------------------------------------
    # 2. Test dequantize
    # -----------------------------------------------------------------
    print("\n[2/7] Testing dequantize...")
    deq_pt = dequantize(q_pt)
    deq_pc = dequantize(q_pc)

    assert deq_pt.shape == test_tensor.shape
    assert len(deq_pt.data) == len(test_tensor.data)
    assert deq_pc.shape == test_tensor.shape
    print(f"  ✓ Reconstructed tensor shape: {deq_pt.shape}, dtype: {deq_pt.dtype}")

    # -----------------------------------------------------------------
    # 3. Test quantization_error
    # -----------------------------------------------------------------
    print("\n[3/7] Testing quantization_error...")
    err_pt = quantization_error([test_tensor], [deq_pt])
    err_pc = quantization_error([test_tensor], [deq_pc])

    assert err_pt["relative"] >= 0.0 and err_pc["relative"] >= 0.0
    print(f"  ✓ Per-tensor relative error:  {err_pt['relative']:.6f} (max_abs: {err_pt['max_abs']:.6f})")
    print(f"  ✓ Per-channel relative error: {err_pc['relative']:.6f} (max_abs: {err_pc['max_abs']:.6f})")

    # -----------------------------------------------------------------
    # 4. Test bytes_stored
    # -----------------------------------------------------------------
    print("\n[4/7] Testing bytes_stored...")
    bytes_fp32 = bytes_stored(model_tensors, precision="fp32")
    bytes_fp16 = bytes_stored(model_tensors, precision="fp16")
    bytes_int8_pt = bytes_stored(model_tensors, precision="int8", granularity="per_tensor")
    bytes_int8_pc = bytes_stored(model_tensors, precision="int8", granularity="per_channel")

    assert bytes_fp32 > bytes_fp16 > bytes_int8_pt
    assert bytes_int8_pc >= bytes_int8_pt  # per-channel carries scale/zero per output channel
    print(f"  ✓ FP32 bytes:            {bytes_fp32} B")
    print(f"  ✓ FP16 bytes:            {bytes_fp16} B")
    print(f"  ✓ INT8 per-tensor bytes: {bytes_int8_pt} B")
    print(f"  ✓ INT8 per-channel bytes:{bytes_int8_pc} B")

    # -----------------------------------------------------------------
    # 5. Test effective_bits
    # -----------------------------------------------------------------
    print("\n[5/7] Testing effective_bits...")
    eff_fp32 = effective_bits(model_tensors, precision="fp32")
    eff_fp16 = effective_bits(model_tensors, precision="fp16")
    eff_int8_pt = effective_bits(model_tensors, precision="int8", granularity="per_tensor")
    eff_int8_pc = effective_bits(model_tensors, precision="int8", granularity="per_channel")

    assert eff_fp32 == 32.0
    assert eff_fp16 == 16.0
    assert eff_int8_pc > 8.0  # accounts for scales and zeros overhead
    print(f"  ✓ FP32 effective bits:            {eff_fp32:.4f}")
    print(f"  ✓ FP16 effective bits:            {eff_fp16:.4f}")
    print(f"  ✓ INT8 per-tensor effective bits: {eff_int8_pt:.4f}")
    print(f"  ✓ INT8 per-channel effective bits:{eff_int8_pc:.4f}")

    # -----------------------------------------------------------------
    # 6. Test precision_row
    # -----------------------------------------------------------------
    print("\n[6/7] Testing precision_row...")
    configs = [
        {"label": "FP32 Baseline", "precision": "fp32"},
        {"label": "FP16", "precision": "fp16"},
        {
            "label": "INT8 Per-Tensor (All)",
            "precision": "int8",
            "granularity": "per_tensor",
            "symmetric": False,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (All)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (Class 0 Only)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "one_class",
        },
    ]

    rows = []
    for cfg in configs:
        calib_vals = calib_one_cls if cfg.get("calibration") == "one_class" else calib_all
        row = precision_row(
            model_name=model_name,
            config=cfg,
            tensors=model_tensors,
            calibration_values=calib_vals,
            all_values=all_act_vals,
        )
        rows.append(row)
        print(f"  ✓ {row['label']:<32} | W_Err: {row['weight_error']:<8.6f} | Act_Err: {row['activation_error']:<8.6f} | Bits: {row['effective_bits']:<6.3f}")

    # -----------------------------------------------------------------
    # 7. Test pareto_front & Save Output
    # -----------------------------------------------------------------
    print("\n[7/7] Testing pareto_front...")
    # Minimize effective_bits and minimize weight_error (sense = -1.0)
    objectives = {
        "effective_bits": -1.0,
        "weight_error": -1.0,
    }
    front_indices = pareto_front(rows, objectives)

    assert len(front_indices) > 0
    print(f"  ✓ Pareto optimal candidate indices: {front_indices}")
    for idx in front_indices:
        p = rows[idx]
        print(f"    * Index {idx}: {p['label']} (Bits: {p['effective_bits']}, Weight Err: {p['weight_error']})")

    # Output to JSON
    report_path = Path("quantization_test_results.json")
    results = {
        "model": model_name,
        "total_parameters": total_params,
        "objectives": objectives,
        "pareto_front_indices": list(front_indices),
        "configurations": rows,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] Full output written to {report_path.resolve()}")


