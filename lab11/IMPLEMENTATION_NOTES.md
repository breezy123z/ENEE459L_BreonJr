# Lab 11 implementation notes

## Objective

Convert sliding-window audio scores into decisions and event onsets, check configuration, and quantify mistakes and timing. The supplied test_voice.wav is used because microphones were unavailable.

## Steps taken

1. Fetched instructor main at f2c1d87, branched solution11 from solution10, and saved the untouched lab11 starter in Git and ../lab11-original.tar locally.
2. Read the seven-page Lab 11 instructions and instructor README.
3. Implemented seven functions and incorporated an existing Jetson edit implementing _validate_plan with detailed handout error messages. Preserved that file locally at ../jetson-lab11-0-stream-preserved.py and on the board in its backup directory.
4. Updated 2_events.py to preserve the four required null stage keys when writing its session; the supplied starter removed them even though the sample includes them and schema.py requires them.
5. Added test_lab11.py for invalid configurations, inclusive thresholds, undefined rates, rare-event arithmetic, partial vote windows, hysteresis, accepted-onset timing, event matching and the three sample reports.
6. Ran all three scripts on the supplied WAV and compared their generated reports with the three provided samples. Values match after excluding generated timestamps and absolute source paths. All 9 test groups passed on the Jetson in 0.021 seconds. See test-results.txt and output0/1/2_session.json for board verification.

## Functions written

| Function | What it does |
|---|---|
| _validate_plan (existing Jetson edit) | Validates plan keys, counts, durations, scores and the constraints k <= n, leave <= enter and hop <= window. |
| confusion | Counts TP, FP, TN and FN using score >= threshold and checks score/label lengths. |
| rates | Computes TPR, FPR, precision, recall and F1; returns None for undefined ratios. |
| precision_at_base_rate | Estimates expected TP, FP and precision given hourly event frequency and decision hop. |
| k_of_n | Counts positive flags across trailing windows, including partial prefixes. |
| hysteresis | Enters at or above the enter threshold; remains active at the leave threshold and exits below it. |
| refractory | Keeps rising edges outside the mute period measured from the last accepted edge. |
| match_events | Matches truth onsets in order to nearest unused predictions inside the collar; reports hits, misses, false alarms and signed delays. |

Other helpers and command-line scripts were supplied. The event script's output serialization was corrected as noted above.

## Results on the supplied audio

The sample is 10 seconds and produces 91 overlapping RMS-score windows. With threshold 0.5, the demonstration has TP=10, FP=5, TN=76 and FN=0; precision=0.6667, recall=1.0 and F1=0.8. The sample event example has one hit, no misses and no false alarms before and after smoothing; the matched smoothed onset is about 100 ms after the derived reference. The calculated 3-of-5 vote delay is 200 ms, a modeled quantity, not a measured runtime.

Important: 1_decide.py and 2_events.py normalize per-window RMS energy as a score. The supplied labels and truth onsets are derived from those same scores, rather than independently annotated speech. These numbers illustrate the code; they are not estimates of real keyword recognition accuracy. The starter also writes example device/software conditions and modeled inference time, which are not independently measured Jetson results. Unknown stages remain null in the output.

## Run

```bash
cd ~/ENEE459L/lab11
python3 0_stream.py test_voice.wav -o output0_session.json
python3 1_decide.py test_voice.wav -o output1_session.json
python3 2_events.py test_voice.wav -o output2_session.json
python3 -m unittest -v test_lab11.py
```

Run the three programs before the tests so the comparison has fresh reports. Only Python standard library modules are required.
