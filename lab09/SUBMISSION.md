# Lab 09 submission

The six required functions are implemented in [agreement.py](agreement.py). See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for the function table, steps, results, and interpretation limits.

Run on the Jetson:

```bash
cd /home/student/ENEE459L/lab09
python3 agreement.py
python3 -m unittest -v test_agreement.py
```

Only the Python standard library is needed. The first command writes benchmark_results.json from the supplied synthetic benchmark_input_data.json. It does not time a live model. The second command runs 15 tests, including a complete comparison against the supplied sample.

Submission branch: https://github.com/breezy123z/ENEE459L_BreonJr/tree/solution9/lab09

Expected summary: 1.41x median speedup, 4.20 ms saved, maximum absolute deviation 0.03, and 3/3 matching top-1 decisions. See saved board run and test logs for actual execution evidence.
