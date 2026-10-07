# Lab 5 implementation and steps taken

Date: October 7, 2026.

1. Fetched instructor main at 829e14a (adding lab5 code).
2. Created solution5 from the completed solution4 and merged the instructor
   update. Earlier lab solutions remain in this branch.
3. Preserved the unedited starter in Git and work/lab05-original.tar.
4. Read README.md, prune.py docstrings, tensors.py, main.py, model.json,
   compare_json.py, and the full reference report.
5. Implemented the eight missing functions in prune.py:
   - magnitude_mask ranks absolute values with deterministic index ties.
   - channel_keep ranks L^p channel norms and keeps at least one channel.
   - apply_mask returns a new tensor with zeros and unchanged shape.
   - drop_channels copies surviving axis-0 slices in original order.
   - bytes_stored accounts for dense weights, framework/bitmap masks,
     or nonzero values plus four-byte sparse indices.
   - sparsity_row records nominal ratio, actual zeros, shape reduction,
     storage bytes, and removal classification separately.
   - classify_removal prioritizes structural changes, sparse storage,
     consecutive 2:4 patterns, masking, then dense baseline.
   - sweep_model starts from original tensors for every sorted ratio.
6. Added 12 tests covering the full reference plus independent small tensors,
   ties, boundaries, invalid inputs, immutability, ordering, channel clamping,
   storage arithmetic, classification precedence, and sweep independence.
7. Ran main.py and compare_json.py locally: 100% match. All 12 tests passed.
8. Reconnected to the Jetson on October 7, 2026. Confirmed the existing
   /home/student/ENEE459L checkout was clean on solution4, fetched fork/solution5,
   and switched to solution5 without making another repository copy.
9. Ran python3 main.py, the instructor JSON comparator, and the test suite on
   the Jetson: 100% reference match and all 12 tests passed (0.136 seconds).
   Generated report: /home/student/ENEE459L/lab05/prune_outputs.json.

## Run from lab05

    python3 main.py
    python3 compare_json.py sample_prune_outputs.json prune_outputs.json
    python3 -m unittest -v test_prune.py

## Results and limits

The original tinyconv tensors contain 11,740 parameters (46,960 FP32 bytes).
At 50% fine pruning: 5,870 zeros, 11,740 parameter slots, 46,960 dense bytes,
93,920 framework-masked bytes, 48,428 bitmap bytes, and 46,960 sparse bytes.
At 50% channel pruning: 5,870 remaining parameters, 23,480 dense bytes.

Channel rounding makes requested and achieved ratios differ in some sweep
rows. Removed channels are absent, not values set to zero: the supplied sample
reports values_zeroed=0 for channel pruning, despite an inconsistent docstring
sentence suggesting zeroed_fraction equals achieved_reduction.
Bitmap masks round up independently per tensor. The 2:4 classifier examines
consecutive flat groups (including a partial final group) as the starter states;
it reports a value pattern, not measured hardware acceleration.

These are deterministic generated weights, not a trained network. No accuracy,
latency, GPU acceleration, actual serialized file size, or working pruned network
is established. Sparse accounting omits row pointers; channel removal does not
propagate shape changes into the next layer. No dependencies were installed.

The separate ELMS slides were not supplied; this implementation follows the
published starter contracts and complete sample. The README has copied lab
references: use sample_prune_outputs.json and branch solution5 for this lab.
