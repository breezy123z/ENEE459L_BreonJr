# Lab 6: steps, explanation, and results

October 14, 2026. Instructor revision: 9400d1f.

## Steps taken and authorship

1. Initially fetched instructor revision 9400d1f and created solution6. That
   revision already contained working solutions; our initial validation-only
   version remains preserved at commit b7822d9.
2. After receiving the six-page handout and the request to remake the code,
   rewrote all seven assignment functions in quantize.py from its requirements.
   The earlier solution had already been read; this is a fresh implementation
   of the stated algorithms, not a claim of a blind or independent discovery.
3. Did not fetch, pull, or merge subsequent instructor changes for this rewrite.
   Retained the supplied Tensor/Quantized types, quantization helpers, FP16
   helper, calibration selection, and executable sample experiment.
4. Added simple numbered comments and spacing. The rewrite encodes groups in
   sequence, decodes groups by slices, separates payload and metadata storage,
   and uses objective scores to identify Pareto domination.
5. Required an explicit nonempty configuration label as the handout states.
   Added an explanatory TensorError for a missing decoding zero point.
6. Expanded the independent arithmetic and edge-case suite from 17 to 19 tests.
   All 19 passed locally; the complete generated report exactly matches the
   existing sample_quantization_results.json.
7. Kept main.py as the README-compatible entry point and saved the latest local
   test run in test-results.txt. The extra activation-aware Pareto report is
   unchanged because the required numerical outputs match.
8. Rewritten code is awaiting a fresh Jetson verification at this commit.

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
