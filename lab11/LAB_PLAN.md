# Lab 11 preparation and requirements

## Status

Fetched instructor commit f2c1d87 (adding lab 11) on November 18, 2026. Created solution11 from completed solution10, merged lab11, and preserved the untouched starter at ../lab11-original.tar locally and in Git history. Read all seven pages of Lab11_instructions.pdf and the repository README. The eight functions are now implemented and tested; see IMPLEMENTATION_NOTES.md for final results.

## Objective

Turn window-level scores into reliable event detections: reject impossible streaming plans, measure decision errors under rare-event conditions, reduce repeated triggers, and match detected onsets against reference events. Lab 10 checked the recording; Lab 11 checks the decisions made over it.

## Required implementations

| File | Function | Required behavior |
|---|---|---|
| 0_stream.py | _validate_plan | Require all plan/stated fields; validate sample rate, k/n, durations, budgets, thresholds and buffer count; enforce k <= n, leave <= enter, hop_s <= window_s. |
| 1_decide.py | confusion | Compare score >= threshold against labels; count tp/fp/tn/fn and total windows. Reject invalid labels and mismatched lengths. |
| 1_decide.py | rates | Compute TPR, FPR, precision, recall and F1. Return None for undefined ratios rather than inventing zero. |
| 1_decide.py | precision_at_base_rate | Use 3600/hop_s decisions per hour, expected positive/negative windows, TPR and FPR to calculate expected TP/FP and real-world precision without rounding intermediate values. |
| 2_events.py | k_of_n | At each index, output whether at least k of the last n boolean flags are true. Evaluate partial startup windows too. |
| 2_events.py | hysteresis | Start on score >= enter; remain active until score < leave. Equality at leave stays active. |
| 2_events.py | refractory | Accept rising edges unless they occur strictly within refractory_s of the last accepted edge. Suppressed edges do not restart the timer. |
| 2_events.py | match_events | Process truth onsets in order, pairing each with the nearest unused prediction within an inclusive collar. Break equal-distance ties by earlier prediction; report signed delta, hits, misses and false alarms. |

## Work sequence

1. Implement and test plan validation, including invalid domain values and contradictory settings.
2. Implement confusion/rates/base-rate arithmetic. Test threshold equality, empty denominators, rare events, and impossible hourly counts.
3. Implement event smoothing and matching. Test startup prefixes, threshold boundaries, refractory equality, suppressed-edge timing, ties, and one-to-one matching.
4. Run all three supplied examples using test_voice.wav.
5. Compare meaningful output fields against the three sample_output*_session.json files. Account for generated timestamps, absolute source paths, and any supplied metadata differences instead of forcing byte-for-byte equality.
6. Save actual test/run logs and output JSON, and document which functions were written versus supplied.
7. Verify the solution11 branch and publish the final submission only after implementation and checks.

## Commands after implementation

```bash
cd ~/ENEE459L/lab11
python3 0_stream.py test_voice.wav -o session.json
python3 1_decide.py test_voice.wav -o decide_session.json
python3 2_events.py test_voice.wav -o events_session.json
```

Python standard-library modules are used in the starters. The provided test_voice.wav permits an offline workflow; a microphone is needed only for a new recording.

## Interpretation cautions from inspecting the supplied drivers

- 1_decide.py and 2_events.py compute scores by sliding a window over the audio, measuring RMS energy, and dividing by the largest window RMS. They are not running a trained speech/keyword recognizer.
- The decision driver generates labels from score >= 0.7. The event driver derives reference onsets from score >= 0.75, including a fallback onset for an empty derived reference. These are demo reference labels, not independently annotated ground truth.
- Device identity, software versions, and inference timing include hard-coded example values. They are not measurements of the Jetson; final documentation must distinguish them from observed conditions.
- Smoothing trades responsiveness for fewer triggers. At a 0.1-second hop, 3-of-5 voting adds a modeled (k-1)*hop = 0.2 seconds for an initial consecutive-positive burst. It can also miss short events.
- An FPR of 0.001 at a 0.1-second hop corresponds to 36 positive decisions per hour if all decisions are negative. Window-level false positives and distinct false-alarm events are different counts.
