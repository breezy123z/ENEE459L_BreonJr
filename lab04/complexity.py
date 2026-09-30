from __future__ import annotations

import math
from typing import Any

from graph import Graph, Layer, computed, dtype_bytes, is_answered, unknown


FLOPS_PER_MAC = 2
FLOP_CONVENTIONS = {"mac_is_two_flops": FLOPS_PER_MAC, "mac_is_one_flop": 1}
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2
BUFFER_DTYPE = "fp32"
MIN_MODELS_FOR_FIT = 3
TIE_EXACT = True


def _problems(graph: Graph) -> list[str]:
    """Reject incomplete descriptions before returning an exact count."""
    names = [layer.name for layer in graph]
    if len(names) != len(set(names)):
        return ["layer names must be unique"]

    for shape in [graph.input_shape] + [
        shape for layer in graph for shape in (layer.in_shape, layer.out_shape)
    ]:
        if not shape or any(type(d) is not int or d <= 0 for d in shape):
            return ["tensor dimensions must be positive integers"]

    for layer in graph:
        if layer.kind == "conv":
            if len(layer.in_shape) != 3 or len(layer.out_shape) != 3:
                return [f"{layer.name}: convolution needs C,H,W shapes"]
            if (layer.kernel is not None and (len(layer.kernel) != 2
                    or any(type(d) is not int or d <= 0 for d in layer.kernel))):
                return [f"{layer.name}: missing or invalid kernel"]
            if type(layer.groups) is not int or layer.groups <= 0:
                return [f"{layer.name}: groups must be a positive integer"]
        if layer.kind == "linear" and (
            len(layer.in_shape) != 1 or len(layer.out_shape) != 1
        ):
            return [f"{layer.name}: linear layer needs feature-vector shapes"]
        if layer.kind == "add" and len(layer.reads) != 2:
            return [f"{layer.name}: add must name two input tensors"]

    problems = graph.validate()
    if graph.layers and not graph.layers[0].reads:
        if graph.layers[0].in_shape != graph.input_shape:
            problems.append("first layer input does not match network input")
    if graph.precision not in ("fp32", "fp16", "bf16", "int8", "int4"):
        problems.append("unknown network input precision")
    return problems


def _layer_parameters(layer: Layer) -> int:
    """Count stored trainable scalars for one supported layer."""
    if layer.kind == "conv":
        c_in, c_out = layer.in_shape[0], layer.out_shape[0]
        kh, kw = layer.kernel or (1, 1)
        return c_out * (c_in // layer.groups) * kh * kw + (c_out if layer.bias else 0)

    if layer.kind == "linear":
        inputs, outputs = layer.in_shape[0], layer.out_shape[0]
        return inputs * outputs + (outputs if layer.bias else 0)

    if layer.kind == "bn":
        return BN_PARAMS_PER_CHANNEL * layer.out_shape[0]

    return 0


def _bytes(value: float) -> int | float:
    """Preserve fractional-byte int4 accounting instead of truncating it."""
    return int(value) if value.is_integer() else value


def count_parameters(graph: Graph) -> dict[str, Any]:
    """Count weights and biases, excluding nontrainable BN statistics."""
    source = f"{graph.name}: {len(graph)} layers, shapes from the description"

    # 1. Check the graph before using its dimensions.
    problems = _problems(graph)
    if problems:
        return unknown(source, "; ".join(problems))


    # 2. Convolutions divide input channels by groups.
    #    Linear layers multiply input features by output features.
    #    Batch norm has one scale and one shift per channel.
    per_layer = {layer.name: _layer_parameters(layer) for layer in graph}


    # 3. Keep each layer's count so the total can be checked.
    return computed(
        sum(per_layer.values()), source, per_layer=per_layer,
        includes_bias=True, excludes_bn_buffers=True,
        bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
    )


def model_size_bytes(graph: Graph) -> dict[str, Any]:
    """Count parameter and persistent-buffer storage at their own dtypes."""
    source = f"{graph.name}: per-layer dtypes, buffers at fp32"

    # 1. Reuse the checked parameter counts.
    params = count_parameters(graph)
    if not is_answered(params):
        return unknown(source, params["detail"])

    per_layer = {}
    per_dtype = {}
    buffer_bytes = 0.0


    # 2. Multiply each layer's parameters by that layer's byte width.
    for layer in graph:
        weight_bytes = params["per_layer"][layer.name] * float(dtype_bytes(layer.weight_dtype))
        per_dtype[layer.weight_dtype] = per_dtype.get(layer.weight_dtype, 0.0) + weight_bytes

        # BN running mean and variance remain FP32, even with FP16 weights.
        buffers = 0.0
        if layer.kind == "bn":
            buffers = BN_BUFFERS_PER_CHANNEL * layer.out_shape[0] * float(dtype_bytes(BUFFER_DTYPE))
            per_dtype[BUFFER_DTYPE] = per_dtype.get(BUFFER_DTYPE, 0.0) + buffers
            buffer_bytes += buffers

        per_layer[layer.name] = _bytes(weight_bytes + buffers)


    # 3. This counts tensors, not serialization headers or archive metadata.
    return computed(
        _bytes(float(sum(per_layer.values()))), source,
        per_layer=per_layer, per_dtype=per_dtype, buffer_bytes=buffer_bytes,
        container_overhead_excluded=True,
        note="not the size of the file on disk; see the handout, Stage A step 3",
    )


def _elements(shape: tuple[int, ...]) -> int:
    """Multiply dimensions to count the scalars in a tensor."""
    return math.prod(shape)


def _last_use(graph: Graph) -> dict[str, int]:
    """Find the final consumer of every output and the network input."""
    last = {}
    names = [layer.name for layer in graph]
    for i, layer in enumerate(graph):
        reads = layer.reads or ((names[i - 1],) if i else ("__input__",))
        for name in reads:
            last[name] = i
        last.setdefault(layer.name, i)

    if graph.layers:
        last[graph.layers[-1].name] = len(graph) - 1
    return last


def _peak_elements(graph: Graph, last_use: dict[str, int]) -> int:
    """Count coexisting elements before releasing each layer's inputs."""
    live = {"__input__": _elements(graph.input_shape)}
    peak = sum(live.values())
    for i, layer in enumerate(graph):
        live[layer.name] = layer.out_elements
        peak = max(peak, sum(live.values()))
        for name, last_index in last_use.items():
            if last_index == i:
                live.pop(name, None)
    return peak


def count_activations(graph: Graph) -> dict[str, Any]:
    """Track tensors until their last consumer, including residual branches."""
    source = f"{graph.name}: liveness over {len(graph)} layers, input included"

    # 1. Check shapes and determine when each tensor can be released.
    problems = _problems(graph)
    if problems:
        return unknown(source, "; ".join(problems))
    last_use = _last_use(graph)
    live = {"__input__": _elements(graph.input_shape) * float(dtype_bytes(graph.precision))}
    peak_bytes = sum(live.values())
    peak_at = "input"
    total_elements = 0
    total_bytes = 0.0


    # 2. Allocate each output while its inputs are still in memory.
    for i, layer in enumerate(graph):
        out_b = layer.out_elements * float(dtype_bytes(layer.act_dtype))
        live[layer.name] = out_b
        total_elements += layer.out_elements
        total_bytes += out_b

        resident = sum(live.values())
        if resident > peak_bytes:
            peak_bytes = resident
            peak_at = layer.name

        # Release tensors only after recording their overlapping memory.
        for name, last_index in last_use.items():
            if last_index == i:
                live.pop(name, None)


    # 3. Total counts outputs; peak also includes the network input.
    return computed(
        _bytes(peak_bytes), source, peak_at=peak_at,
        peak_elements=_peak_elements(graph, last_use), total_elements=total_elements,
        total_bytes=total_bytes, includes_network_input=True,
        note="peak is the resident set, not the largest single tensor",
    )


def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops") -> dict[str, Any]:
    """Convert an operation count once, retaining the stated convention."""
    source = macs.get("source", "MAC finding") if isinstance(macs, dict) else "MAC finding"

    # 1. Unknown inputs and unsupported conventions are not zero operations.
    if not isinstance(macs, dict) or not is_answered(macs):
        return unknown(source, "No valid MAC count was provided")
    if convention not in FLOP_CONVENTIONS:
        return unknown(source, f"unrecognized FLOP convention: {convention}; allowed: {list(FLOP_CONVENTIONS)}")

    def valid(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0 and int(value) == value)

    if not valid(macs.get("value")):
        return unknown(source, "MAC count must be a finite nonnegative integer")
    layers = macs.get("per_layer", {})
    if not isinstance(layers, dict) or not all(valid(value) for value in layers.values()):
        return unknown(source, "invalid per-layer MAC counts")


    # 2. Multiply both the total and the breakdown by the chosen factor.
    factor = FLOP_CONVENTIONS[convention]
    return computed(
        macs["value"] * factor, source,
        convention=convention, flops_per_mac=factor,
        per_layer={name: value * factor for name, value in layers.items()},
        note="a count of operations contains no unit of time",
    )
