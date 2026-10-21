# Lab 7

- [Implemented functions](frames.py)
- [Requirements and work log](IMPLEMENTATION_NOTES.md)
- [Pre-lab predictions](prelab_predictions.json) and [generator](prelab.py)
- [Tests](test_frames.py) and [local results](test-results.txt)
- [Instructor reference artifacts](reference_output/)

The original camera pipeline, helpers, schema, and selfie example were provided.
The eight frame/accounting/diagnostic functions are implemented in this branch.
Actual capture artifacts belong in output/ once a connected camera is available.

All 18 tests pass locally and on the Jetson. Camera capture remains pending:
Argus reports no cameras available. See [hardware status](hardware_status.json).
