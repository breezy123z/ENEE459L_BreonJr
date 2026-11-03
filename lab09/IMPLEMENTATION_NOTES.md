# Lab 09 implementation notes

## Requirements and scope

The instructor's README asks for six functions in agreement.py and comparison with sample_benchmark_results.json. All six starter bodies were `pass`. This lab processes supplied synthetic timing and score arrays. It does not build or benchmark a real TensorRT engine, install dependencies, or capture images. The broader TensorRT/session constants in schema.py are supplied infrastructure, not additional requested implementations.

The README references ELMS slides that are not included in the public repository. This implementation follows the README and sample; check any additional slide-specific requirements before submission.

## Steps taken

1. Fetched instructor main at f63f33b (adding lab 9).
2. Created solution9 from the completed solution8 branch and merged the new starter, preserving earlier lab work.
3. Saved the untouched lab09 starter in ../lab09-original.tar locally. The merge commit also preserves the starter in Git history.
4. Implemented the six required functions with simple comments and separate validation/calculation blocks.
5. Added two shared validation helpers to avoid silently truncating mismatched arrays and to reject non-finite scores, booleans, and empty rows.
6. Added 15 unittest cases covering the full supplied report, error handling, interpolation, ties, denominator floors, mismatches, and margin boundaries.
7. Tested locally. All sample fields match; floating-point fields allow a 1e-12 absolute tolerance because Python versions can sum floating-point values differently.
8. Published solution9, updated the existing Jetson checkout, and ran the tests and supplied example there. Board evidence is saved in test-results.txt and run-results.txt.

## Functions written

| Function | How it works | Purpose |
|---|---|---|
| latency_report | Validates and materializes the input iterable; calculates count, arithmetic mean, maximum, interpolated p50/p95/p99, and p95 minus p50. | Compare typical latency, tail latency, and timing jitter. |
| speedup | Divides baseline by optimized positive duration; subtracts optimized from baseline; faster means factor > 1. | Report acceleration and signed milliseconds saved. |
| max_abs_dev | Checks equal nonempty lengths, calculates absolute differences, and takes the first maximum. | Find the largest numerical output discrepancy. |
| max_rel_dev | Divides each absolute difference by max(abs(reference), floor); reports first maximum and clamping information. | Describe proportional error without dividing by zero or hiding denominator flooring. |
| top1_agreement | Finds each paired row's maximum class index, choosing the lowest index on ties; counts matches and first mismatch. | Check actual classification parity. Rate is a fraction from 0 to 1, as in the sample. |
| decision_margin | Sorts each score row and subtracts runner-up from winner; aggregates gaps and compares them with tolerance. | Identify decisions close to the supplied threshold. |
| _paired_values | Validates equal lengths and every numeric entry. | Shared flat-output validation. |
| _score_rows | Validates nonempty rows and every numeric score. | Shared classification validation. |

## Already provided

AgreementError, _finite, _percentile, deviation_sentence, decision_sentence, and main were supplied. Their behavior is preserved. schema.py, input JSON, sample JSON, and the instructor README are unchanged. test_agreement.py is new student-side verification, not instructor-provided testing.

## Boundary choices and limitations

- Empty inputs and mismatched output shapes raise AgreementError rather than returning misleading statistics. Timing zero is allowed, but speedup requires both durations strictly positive.
- Relative denominators count as floored only when abs(reference) < floor, not at equality.
- A margin equal to tolerance is counted in below_tol. This conservative equality choice is explicit because the public sample does not exercise the boundary. Tied top classes at zero tolerance are therefore unsafe.
- The supplied main compares margins against one maximum absolute deviation. If each of two competing scores can change by epsilon independently, a general prediction-preservation guarantee requires margin > 2*epsilon. Also, flat logits and batched scores in the example are separate arrays. Consequently the provided decision_sentence is a lab-format summary, not proof for arbitrary inputs or an independently measured model. We retain its sample-compatible wording and contract. The sample minimum gap 0.1 exceeds both epsilon=0.03 and 2*epsilon=0.06, and its three observed predictions agree.
- Finite ordinary Python int/float values are supported as specified; this is not an arbitrary-precision numeric or NumPy-array interface.

## Supplied example findings

| Metric | Baseline | Optimized |
|---|---:|---:|
| Median latency (ms) | 14.45 | 10.25 |
| Mean latency (ms) | 14.72 | 10.48 |
| p95 latency (ms) | 15.82 | 11.365 |
| p99 latency (ms) | 15.964 | 11.473 |
| Worst latency (ms) | 16.0 | 11.5 |
| Jitter p95-p50 (ms) | 1.37 | 1.115 |

Median speedup is 1.409756x, saving 4.2 ms. Maximum absolute deviation is approximately 0.03 at index 2; maximum relative deviation is 0.5 at index 5, with no floored denominators. All 3 top-1 predictions agree. Minimum baseline decision margin is 0.1 at row 1. These are calculations from synthetic instructor inputs, not measured Jetson inference performance.
