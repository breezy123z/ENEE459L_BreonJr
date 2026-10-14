# Lab 6: steps, explanation, and results

October 14, 2026. Instructor revision: 9400d1f.

## Steps taken and authorship

1. Fetched instructor main and inspected the six new Lab 6 files.
2. Created solution6 from solution5, merged the instructor update, and preserved
   the original Lab 6 starter in Git and work/lab06-original.tar locally.
3. Confirmed the latest quantize.py already implements all seven assignment
   functions. These algorithms are supplied by the instructor, not authored
   by us. We added explanatory comments without changing their calculations.
4. Added main.py to run the existing experiment, matching the README command.
   The original executable entry point remains python3 quantize.py.
5. Added 17 independent tests covering arithmetic examples, calibration,
   clipping, range choices, storage metadata, Pareto dominance, and errors.
6. Ran the experiment and tests locally: all 17 passed. The entire generated
   quantization_test_results.json exactly equals sample_quantization_results.json.
7. Added activation_pareto_analysis.json to explain a limitation of the default
   objective choice while preserving the instructor's required report.
8. Connected to the Jetson, confirmed the existing repository was clean on
   solution5, fetched fork/solution6, and switched to solution6 in place.
9. Ran python3 main.py and python3 -m unittest -v test_quantize.py on the
   Jetson: all 17 tests passed in 0.217 seconds, including exact reference
   equality. No dependencies were installed. The output is at
   /home/student/ENEE459L/lab06/quantization_test_results.json.
10. Confirmed the quantize.py syntax tree is identical to instructor revision
    9400d1f: only explanatory comments were added to the supplied algorithms.

## What each stage does

- quantize_tensor chooses integer bounds (-127..127 symmetric, -128..127
  asymmetric for INT8), finds a range, and derives scale and zero point.
  Code = clamp(round_half_up(value / scale) + zero_point).
  Per-tensor uses one pair per tensor; per-channel uses one per output channel.
  Out-of-range integer codes are counted before clamping.
- dequantize reconstructs value = scale * (code - zero_point), using the
  correct channel's scale. Reported reconstructed values are rounded to 8 decimals.
- quantization_error computes sqrt(sum((original-reconstructed)^2)), divides
  that norm by the original norm for relative error, and records worst absolute
  difference. This measures numerical reconstruction, not classification accuracy.
- bytes_stored counts FP32 at 4 bytes/value, FP16 at 2, and INT8 at 1 plus
  one 4-byte scale and one 4-byte zero point for each quantization group.
- effective_bits computes 8 * total_bytes / parameters, including metadata.
- precision_row applies the weight policy, calibrates activations using selected
  values, evaluates all activations, and combines size/error/clipping results.
  Activations use asymmetric per-tensor INT8 even for symmetric weight policies.
- pareto_front retains a configuration unless another is at least as good on
  every selected objective and strictly better on one. Equal points survive.

## Results

| Configuration | Bytes | Bits/parameter | Weight relative error | Activation relative error | Clipped fraction |
|---|---:|---:|---:|---:|---:|
| FP32 | 46,960 | 32 | 0 | 0 | 0 |
| FP16 | 23,480 | 16 | 0.00020098 | 0.00020777 | 0 |
| INT8 per-tensor, all calibration | 11,772 | 8.0218 | 0.00593672 | 0.00297452 | 0 |
| INT8 per-channel, all calibration | 12,460 | 8.4906 | 0.00389339 | 0.00297452 | 0 |
| INT8 per-channel, class 0 calibration | 12,460 | 8.4906 | 0.00389339 | 0.64920162 | 0.5109375 |

Good calibration covers activations up to 4.99642334; class 0 reaches only
0.99689778. The biased policy therefore clips 51.09375% of evaluated values.
The two per-channel configurations have identical weight errors and storage,
but radically different activation reconstruction errors.

The default frontier minimizes bits and WEIGHT error only, so all five indices
survive; the two per-channel configurations tie on both selected metrics.
Adding activation_error as an objective removes index 4 (biased calibration),
leaving [0, 1, 2, 3]. This extra report explains why objective choice matters.
The provided per-tensor/per-channel configurations also differ in weight
symmetry, so their report comparison does not isolate granularity alone. Our
small per-channel test holds symmetry fixed to check that mechanism separately.

## Running and limits

From lab06:

    python3 main.py
    python3 -m unittest -v test_quantize.py

No third-party packages or GPU are needed. Data are deterministic synthetic
weights and activation vectors. The experiment does not train a model, run
inference, measure accuracy/latency, serialize a real INT8 artifact, or measure
actual disk file size. Bytes are calculated payload/metadata accounting.
FP16 conversion uses binary16 rounding. The MSE clipping helper uses a finite
candidate grid and an approximate error objective supplied by the instructor.
ELMS slides were not supplied; checks follow the source contracts and sample.
