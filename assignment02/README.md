# Assignment 2 - Arithmetic intensity

## What the assignment asks

Implement arithmetic_intensity(graph) to divide operation count by pessimistic
DRAM traffic, per layer and across the model. Each layer pays for its own input
reads, parameters/buffers, and output write. There is no cross-layer cache reuse
or fusion. Return the exact type/message finding described in the PDF.

## Submitted implementation

- arithmetic.py: required function, explicit producer dtype resolution, and
  standalone equivalents of the helpers named by the handout.
- graph.py: instructor-provided Lab 4 graph classes and loader, unchanged.
- model.json: explicit eight-layer Lab 4 toy_resnet demonstration.
- main.py: portable runner producing JSON and CSV.
- test_arithmetic.py: 15 independent tests, all passing.
- Assignment2_Report.pdf: six-page explanation with four computed charts.
- charts/: PNG and vector SVG exports.
- results/: exact numerical outputs and hypothetical FP16 sensitivity analysis.
- build_report.py and requirements-report.txt: reproduce figures and report.

## Run

    python main.py
    python -m unittest -v test_arithmetic.py

To analyze another compatible JSON description:

    python main.py --model path/to/model.json --output-dir results/custom

The computation and tests use standard Python 3.10+; no Jetson or GPU is needed.
To regenerate the report and charts, install the optional plotting dependencies:

    python -m pip install -r requirements-report.txt
    python build_report.py

## Results and meaning

The example produces 5,603,648 FLOPs and 745,360 bytes, or 7.5180 FLOPs/byte.
The residual convolution has 33.6350 FLOPs/byte. Activation reads/writes account
for 98.41% of traffic; zero-MAC operators account for 70.40% of bytes.
The hypothetical FP16 scenario uses 372,744 bytes and reaches 15.0335 FLOPs/byte.
All figures are calculated model costs, not measured bus usage or speed.

## Requirements checklist

- [x] First explicit producer determines input dtype.
- [x] Implicit input uses immediate predecessor dtype.
- [x] External input uses graph.precision.
- [x] Two-input add doubles first-producer input bytes per PDF formula.
- [x] Trainable parameters include biases and BN scale/shift.
- [x] BN running mean/variance are additional FP32 buffer bytes.
- [x] Zero-FLOP layers stay present with intensity 0.0 and nonzero traffic.
- [x] Whole-model result uses sum(FLOPs) / sum(bytes).
- [x] Ratios round to four decimals only at reporting time.
- [x] Required type/message schema, source string, note and flags are returned.
- [x] Tests cover fractional INT4 bytes, empty/zero-byte graphs, invalid
      dependency names and duplicate layer names as well as the main rules.

## Important conventions and provenance

The supplied PDF names _layer_macs but does not supply its implementation or
an Assignment 2 model file. The included helper counts conv MACs as output
values * (input channels // groups) * kernel area, and linear MACs as input
features * output features. Bias adds are not MACs. Other operators return zero
under this demonstration's MAC-only convention. Use course-provided helpers if
they are supplied separately. ReLU comparisons are not counted as FLOPs here.
The example uses real shape-derived MAC arithmetic, not Lab 4 main.py's mock count.

The PDF's add rule prices both inputs using reads[0]'s dtype. That convention
is followed exactly, including a mixed-branch test. Actual unequal-width branches
would require independent input accounting. No shape validation is required by
the assignment function; valid topological graph descriptions are assumed.

computed() in this assignment returns type/message as the PDF requests. The
older graph.py computed helper uses status/source, so arithmetic.py supplies a
separate assignment-specific wrapper. Both the required function and its local
input-dtype resolver can be integrated with another starter's existing helpers.

No cache/fusion savings, workspaces, intermediate padding, real hardware ceiling,
accuracy, or measured execution time are claimed. A low ratio indicates more
bytes per operation but cannot alone establish a device bottleneck.

The implementation and report were produced using coding assistance; the Graph
utilities are instructor-provided. All calculations and claims are reproducible.
