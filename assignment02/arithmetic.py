"""Assignment 2: arithmetic intensity under the handout's DRAM model."""
from __future__ import annotations
from typing import Any
from graph import Graph, Layer, dtype_bytes

FLOPS_PER_MAC = 2
BN_BUFFERS_PER_CHANNEL = 2
BUFFER_DTYPE = 'fp32'


def computed(value: Any, message: str, **metadata: Any) -> dict[str, Any]:
    """Assignment 2 uses type/message, unlike Lab 4's status/source wrapper."""
    return {'type': 'computed', 'value': value, 'message': message, **metadata}


def _layer_parameters(layer: Layer) -> int:
    """Standalone parameter helper; includes biases, excludes BN buffers."""
    if layer.kind == 'conv':
        kh, kw = layer.kernel or (1, 1)
        return layer.out_shape[0] * (layer.in_shape[0] // layer.groups) * kh * kw + (layer.out_shape[0] if layer.bias else 0)
    if layer.kind == 'linear':
        return layer.in_shape[0] * layer.out_shape[0] + (layer.out_shape[0] if layer.bias else 0)
    if layer.kind == 'bn':
        return 2 * layer.out_shape[0]
    return 0


def _layer_macs(layer: Layer) -> int:
    """Conv/linear MAC convention for bundled examples; no bias-add MACs.

    The PDF names this helper but does not supply its implementation. If the
    course provides another helper, use that in the assignment environment.
    Other supported operators contribute zero MACs under this convention.
    """
    if layer.kind == 'conv':
        kh, kw = layer.kernel or (1, 1)
        return layer.out_elements * (layer.in_shape[0] // layer.groups) * kh * kw
    if layer.kind == 'linear':
        return layer.in_shape[0] * layer.out_shape[0]
    return 0


def _input_dtype(layer: Layer, producers: dict[str, Layer], previous: Layer | None,
                 external_dtype: str) -> str:
    if layer.reads:
        try:
            return producers[layer.reads[0]].act_dtype
        except KeyError:
            raise ValueError(f'{layer.name}: first input producer must precede this layer') from None
    return previous.act_dtype if previous is not None else external_dtype


def arithmetic_intensity(graph: Graph) -> dict[str, Any]:
    """Return per-layer and whole-graph FLOPs/byte using the PDF's rules."""
    per_layer: dict[str, dict[str, Any]] = {}
    producers: dict[str, Layer] = {}
    previous = None
    total_flops = 0
    total_bytes = 0.0

    for layer in graph:
        if layer.name in producers:
            raise ValueError(f'duplicate layer name: {layer.name}')

        # 1. A producer determines its tensor's dtype, not its consumer.
        input_dtype = _input_dtype(layer, producers, previous, graph.precision)
        input_streams = 2 if layer.kind == 'add' and len(layer.reads) == 2 else 1
        input_bytes = input_streams * layer.in_elements * dtype_bytes(input_dtype)


        # 2. Read learnable weights and persistent BatchNorm statistics.
        weight_bytes = _layer_parameters(layer) * dtype_bytes(layer.weight_dtype)
        buffer_bytes = 0
        if layer.kind == 'bn':
            buffer_bytes = (BN_BUFFERS_PER_CHANNEL * layer.out_shape[0]
                            * dtype_bytes(BUFFER_DTYPE))


        # 3. Write the output and price this layer without inter-layer reuse.
        output_bytes = layer.out_elements * dtype_bytes(layer.act_dtype)
        moved = float(input_bytes + weight_bytes + buffer_bytes + output_bytes)
        flops = int(_layer_macs(layer) * FLOPS_PER_MAC)
        per_layer[layer.name] = {
            'flops': flops,
            'bytes': moved,
            'intensity': round(flops / moved, 4) if moved else 0.0,
        }
        total_flops += flops
        total_bytes += moved
        producers[layer.name] = layer
        previous = layer


    # 4. Divide aggregate costs, retaining zero-FLOP layers' memory traffic.
    overall = round(total_flops / total_bytes, 4) if total_bytes else 0.0
    return computed(
        overall, f'{graph.name}: FLOPs over bytes, every byte counted once',
        per_layer=per_layer, total_flops=total_flops, total_bytes=float(total_bytes),
        flops_per_mac=FLOPS_PER_MAC, cache_reuse_assumed=False,
        note='no fusion and no cache reuse; both would lower the byte column',
    )
