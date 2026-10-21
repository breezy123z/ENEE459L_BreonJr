"""ENEE459L Assignment 2 - single-file submission.

Required function: arithmetic_intensity(graph).
Includes Graph/Layer support and the helpers named by the handout so no
companion Python files or third-party packages are needed.
Graph utilities are adapted from the instructor's Lab 4 starter.

Run: python Assignment2.py
Optional: python Assignment2.py --model model.json --output results.json
The default graph is a synthetic demonstration, not a measured benchmark.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable



DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}

PARAMETRIC = frozenset({"conv", "linear", "bn"})


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    kernel: tuple[int, int] | None = None
    stride: int = 1
    groups: int = 1
    bias: bool = False
    weight_dtype: str = "fp32"
    act_dtype: str = "fp32"
    reads: tuple[str, ...] = ()

    def elements(self, shape: tuple[int, ...]) -> int:
        n = 1
        for d in shape:
            n *= d
        return n

    @property
    def in_elements(self) -> int:
        return self.elements(self.in_shape)

    @property
    def out_elements(self) -> int:
        return self.elements(self.out_shape)

    @property
    def out_spatial(self) -> int:
        if len(self.out_shape) == 3:
            return self.out_shape[1] * self.out_shape[2]
        return 1


@dataclass
class Graph:
    name: str
    input_shape: tuple[int, ...]
    layers: list[Layer]
    precision: str = "fp32"
    source: str = "unattributed"
    notes: str = ""
    _by_name: dict[str, Layer] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {ly.name: ly for ly in self.layers}

    def __iter__(self) -> Iterable[Layer]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def get(self, name: str) -> Layer | None:
        return self._by_name.get(name)

    def conditions(self) -> dict[str, Any]:
        """The fields that must match before two graphs may be ranked together."""
        return {
            "input_shape": list(self.input_shape),
            "precision": self.precision,
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        prev: Layer | None = None
        seen: set[str] = set()

        for ly in self.layers:
            if ly.kind not in {
                "conv", "linear", "pool", "relu", "add", "bn", "flatten",
            }:
                problems.append(f"{ly.name}: unknown layer kind {ly.kind!r}")
            for label, dt in (("weight", ly.weight_dtype), ("act", ly.act_dtype)):
                if dt not in DTYPE_BYTES:
                    problems.append(f"{ly.name}: unknown {label} dtype {dt!r}")

            if not ly.reads and prev is not None and ly.in_shape != prev.out_shape:
                problems.append(
                    f"{ly.name}: input {ly.in_shape} does not match "
                    f"{prev.name} output {prev.out_shape}"
                )

            for target in ly.reads:
                if target not in seen:
                    problems.append(
                        f"{ly.name}: reads {target!r}, which is not an earlier layer"
                    )

            if ly.kind == "add" and len(ly.reads) == 2:
                shapes = {self.get(t).out_shape for t in ly.reads if self.get(t)}
                if len(shapes) > 1:
                    problems.append(
                        f"{ly.name}: add over tensors of different shapes {sorted(shapes)}"
                    )

            if ly.kind == "conv":
                c_in = ly.in_shape[0]
                c_out = ly.out_shape[0]
                if ly.groups < 1 or c_in % ly.groups or c_out % ly.groups:
                    problems.append(
                        f"{ly.name}: groups={ly.groups} does not divide "
                        f"C_in={c_in} / C_out={c_out}"
                    )
                if len(ly.in_shape) == 3 and len(ly.out_shape) == 3 and ly.stride:
                    want = -(-ly.in_shape[1] // ly.stride)
                    if abs(ly.out_shape[1] - want) > 1:
                        problems.append(
                            f"{ly.name}: output H={ly.out_shape[1]} is not "
                            f"consistent with input H={ly.in_shape[1]} at "
                            f"stride {ly.stride}"
                        )

            seen.add(ly.name)
            prev = ly

        return problems


# --- loading ----------------------------------------------------------------


def load_graph(path: str | Path) -> Graph:
    """Read one model description from JSON."""
    raw = json.loads(Path(path).read_text())
    layers = [
        Layer(
            name=d["name"],
            kind=d["kind"],
            in_shape=tuple(d["in_shape"]),
            out_shape=tuple(d["out_shape"]),
            kernel=tuple(d["kernel"]) if d.get("kernel") else None,
            stride=int(d.get("stride", 1)),
            groups=int(d.get("groups", 1)),
            bias=bool(d.get("bias", False)),
            weight_dtype=d.get("weight_dtype", raw.get("precision", "fp32")),
            act_dtype=d.get("act_dtype", raw.get("precision", "fp32")),
            reads=tuple(d.get("reads", ())),
        )
        for d in raw["layers"]
    ]
    return Graph(
        name=raw["name"],
        input_shape=tuple(raw["input_shape"]),
        layers=layers,
        precision=raw.get("precision", "fp32"),
        source=raw.get("source", "unattributed"),
        notes=raw.get("notes", ""),
    )


def load_graphs(directory: str | Path) -> list[Graph]:
    """Every `*.json` in a directory, in sorted filename order."""
    return [load_graph(p) for p in sorted(Path(directory).glob("*.json"))]


def load_bench(path: str | Path) -> dict[str, Any] | None:
    """Read one Lab 03 record, or None if it is absent or not JSON.

    Returning None rather than raising is the whole reason this is a function.
    A missing benchmark record is the normal state of the world in Stage A —
    you have not run anything yet — and every function downstream has to
    produce a report anyway, with the latency column honestly unknown. A loader
    that raises turns "I have not measured this yet" into a crash.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None




def dtype_bytes(name: str) -> float:
    """Byte width of each stored scalar."""
    if name not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype: {name!r}")
    return DTYPE_BYTES[name]


"""Assignment 2: arithmetic intensity under the handout's DRAM model."""
from typing import Any

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


DEMO_MODEL = {'name': 'toy_resnet', 'input_shape': [3, 32, 32], 'precision': 'fp32', 'source': 'Synthetic Lab 4 toy_resnet layer descriptions; Assignment 2 computed MACs (not Lab 4 mock MACs)', 'notes': 'Eight-layer demonstration, not a measured runtime or trained accuracy result.', 'layers': [{'name': 'conv1', 'kind': 'conv', 'in_shape': [3, 32, 32], 'out_shape': [16, 32, 32], 'kernel': [3, 3], 'stride': 1, 'groups': 1, 'bias': False, 'weight_dtype': 'fp32', 'act_dtype': 'fp32', 'reads': []}, {'name': 'bn1', 'kind': 'bn', 'in_shape': [16, 32, 32], 'out_shape': [16, 32, 32], 'kernel': None, 'stride': 1, 'groups': 1, 'bias': False, 'weight_dtype': 'fp32', 'act_dtype': 'fp32', 'reads': []}, {'name': 'relu1', 'kind': 'relu', 'in_shape': [16, 32, 32], 'out_shape': [16, 32, 32], 'kernel': None, 'stride': 1, 'groups': 1, 'bias': False, 'weight_dtype': 'fp32', 'act_dtype': 'fp32', 'reads': []}, {'name': 'res_conv', 'kind': 'conv', 'in_shape': [16, 32, 32], 'out_shape': [16, 32, 32], 'kernel': [3, 3], 'stride': 1, 'groups': 1, 'bias': False, 'weight_dtype': 'fp32', 'act_dtype': 'fp32', 'reads': ['relu1']}, {'name': 'add1', 'kind': 'add', 'in_shape': [16, 32, 32], 'out_shape': [16, 32, 32], 'kernel': None, 'stride': 1, 'groups': 1, 'bias': False, 'weight_dtype': 'fp32', 'act_dtype': 'fp32', 'reads': ['relu1', 'res_conv']}, {'name': 'pool', 'kind': 'pool', 'in_shape': [16, 32, 32], 'out_shape': [16, 1, 1], 'kernel': None, 'stride': 1, 'groups': 1, 'bias': False, 'weight_dtype': 'fp32', 'act_dtype': 'fp32', 'reads': []}, {'name': 'flatten', 'kind': 'flatten', 'in_shape': [16, 1, 1], 'out_shape': [16], 'kernel': None, 'stride': 1, 'groups': 1, 'bias': False, 'weight_dtype': 'fp32', 'act_dtype': 'fp32', 'reads': []}, {'name': 'fc', 'kind': 'linear', 'in_shape': [16], 'out_shape': [10], 'kernel': None, 'stride': 1, 'groups': 1, 'bias': True, 'weight_dtype': 'fp32', 'act_dtype': 'fp32', 'reads': []}]}


def demo_graph() -> Graph:
    """Load the embedded Lab 4 toy graph; no model.json is needed."""
    layers = []
    for description in DEMO_MODEL['layers']:
        fields = dict(description)
        for name in ('in_shape', 'out_shape', 'kernel', 'reads'):
            if fields.get(name) is not None:
                fields[name] = tuple(fields[name])
        layers.append(Layer(**fields))
    return Graph(DEMO_MODEL['name'], tuple(DEMO_MODEL['input_shape']), layers,
                 precision=DEMO_MODEL['precision'], source=DEMO_MODEL['source'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Assignment 2 arithmetic intensity')
    parser.add_argument('--model', type=Path, help='Optional compatible graph JSON')
    parser.add_argument('--output', type=Path, help='Optional JSON output path')
    args = parser.parse_args()
    graph = load_graph(args.model) if args.model else demo_graph()
    result = arithmetic_intensity(graph)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
