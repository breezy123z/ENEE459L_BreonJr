# Lab 3 implementation and validation

## Current status

All six requested functions are implemented in `measure.py` with numbered
comments and spacing. Twenty deterministic tests pass on the Jetson after the handout corrections.
The original starter is preserved at:
`/home/student/ENEE459L-backups/lab03-original-5dd86b3.tar`.

The real inference run is blocked: `/usr/bin/python3` cannot import `torch`.
No virtual environment was found under /home/student or /opt to depth four,
and the Docker image inventory was empty. The instructor's intended compatible
PyTorch and torchvision environment is still needed. No packages were installed.
No inference timing report has been fabricated or copied from the sample.

## Run

```bash
cd ~/ENEE459L/lab03
python3 -m unittest -v test_measure.py
python3 measure.py
```

The benchmark defaults to CUDA and 100 repetitions. After activating the course
Python environment, the last command writes:

- `samples_analysis.json`: raw measured latencies, matching the sample list format.
- `system_report.json`: the five top-level entries in the instructor's sample.
- `benchmark_analysis.json`: model and device context, retained statistics,
  stationarity, and whether five samples sit strictly above each percentile.

`--device cpu` explicitly selects CPU measurements; there is no automatic CPU
fallback. Both modes need PyTorch and torchvision.

## Interpretation choices

- Timing synchronizes once before the loop, then uses clock, workload,
  synchronization, clock for every repetition. Input
  allocation and transfer are outside the measured boundary in the given helper.
- Warm-up uses the median of the second half and the provided 50% tolerance.
  It requires four samples and a positive second-half median, then removes
  only a consecutive leading prefix, exactly as Problem 2 specifies.
  This conflicts with the README statement that drifting runs discard nothing;
  the detailed handout takes precedence. Drift is checked separately.
- Stationarity compares first-third and last-third medians using the provided
  10% tolerance. Modality receives the retained samples, not cold-start samples.
- Summary uses sample standard deviation and linear percentile interpolation.
  As specified, standard deviation is 0.0 for one or two samples. Empty
  input returns n=0 and all metrics null.
- Modality trims 5% from each end, calculates all adjacent gaps, and uses
  their median. A nonpositive median gap returns unknown (coarse timer).
  The widest gap determines the split; shares use the trimmed sample count.
- `summarize_setup` describes raw samples as in the starter. The additional
  analysis file describes the retained samples and warns about sparse tails.
- CPU0 min=max is a clock-pinning hint; it does not prove every GPU/EMC clock
  is pinned or that the jetson_clocks command was run.
- INA3221 `in*_input` is millivolts, not milliwatts. For a rail labelled VDD_IN
  or VIN_SYS_5V0, we multiply millivolts by milliamps and divide by 1000.
  Legacy in_power nodes are read as milliwatts. Sensor paths and units are
  recorded, missing readings are unknown, and GPU per-mille is divided by ten.

## Actual hardware snapshot

`hardware_snapshot.json` records a separate live reading on September 23, 2026.
It is not telemetry from an inference benchmark. The board reported 25W mode,
CPU0 scaling limits not pinned, six readable thermal zones, peak 45.91 C,
board input power 4461.312 mW, and GPU load 0% at that instant.

Tests cover timing order and units, warm-up lengths, threshold equality,
drifting samples, percentiles, constant and split populations, isolated outliers,
invalid inputs, power modes, sensor fallback, voltage/current conversion, and
sample report keys. The supplied three-page handout was checked; hidden grading tests were unavailable;
the implementation follows the handout, with the documented power-unit correction.
