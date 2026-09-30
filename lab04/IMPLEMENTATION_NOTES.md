# Lab 4 — implementation and checks

## What was implemented

The four requested functions are in complexity.py, with simple numbered
comments and space between each step. graph.py and main.py remain the
instructor's files.

1. count_parameters: convolution weights are Cout * (Cin / groups) * Kh * Kw,
   plus Cout biases when enabled. Linear weights are inputs * outputs plus
   optional output biases. Batch normalization has two parameters per channel.
   Pool, ReLU, add, and flatten layers have no parameters.

2. model_size_bytes: each layer's parameters use its own weight dtype. Batch
   normalization adds running mean and variance buffers at FP32. The count
   excludes file-container overhead. int4 uses the supplied 0.5 bytes per
   element; no unprovided byte-packing or alignment overhead is assumed.

3. count_activations: explicit reads identify dependencies; otherwise a layer
   reads its predecessor (or the network input for the first layer). The pass
   records each tensor's last consumer, allocates the output before releasing
   inputs, and retains skip tensors until their final use. Total activation
   elements count layer outputs; resident peak includes the network input.
   Each layer output is separately allocated, including flatten and ReLU, as
   in the course's abstract graph model. This does not model runtime aliasing,
   kernel workspace, allocator caching, or training gradients.

4. to_flops: known MAC counts are multiplied once by the selected factor,
   including the per-layer breakdown. Both conventions are explicitly named.
   Unknown inputs or conventions return unknown, not zero.

## Validation

Run from lab04:

    python3 main.py
    python3 -m unittest -v test_complexity.py

All 18 tests passed locally and on the Jetson on September 30, 2026. The entire generated
complexity_results.json equals sample_complexity_results.json after parsing.
Additional independent examples check grouped and depthwise convolutions,
linear biases, FP16 weights with FP32 buffers, int4 storage, chain liveness,
residual lifetimes, mixed activation precision, invalid inputs, and conversions.

The instructor's synthetic toy_resnet has:
- 2,938 parameters
- 11,880 bytes of model tensor storage
- 81,962 total generated activation elements
- 196,608 bytes of peak resident activation memory, at add1
- 3,000,000 FLOPs for the supplied mock 1,500,000 MACs under the two-FLOP convention

These are static calculations for the provided synthetic graph. The mock MAC
count is supplied by main.py, not measured or independently inferred. This lab
does not measure inference latency, and these counts alone cannot establish it.

## Repository and original starter

The solution4 branch incorporates the new instructor commits while preserving
the user's actual Lab 3 measurement reports over conflicting instructor copies.
The unedited Lab 4 starter is available in the upstream merge parent in Git and
in the local archive work/lab04-original.tar in the task workspace.

The Jetson repository at /home/student/ENEE459L was updated to solution4.
Both python3 main.py and python3 -m unittest -v test_complexity.py succeeded
on the Jetson (18 tests passed). No GPU is required for this static analysis.

Reviewed against the supplied five-page Lab 4 handout. The named helpers
_layer_parameters, _elements, _last_use, and _peak_elements are implemented.
An omitted convolution kernel defaults to (1, 1). Invalid FLOP convention
messages list both allowed options. Tests cover these handout requirements
as well as the complete sample report.
