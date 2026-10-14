# Lab 6 submission

- [Quantization functions and explanatory comments](quantize.py)
- [Generated report](quantization_test_results.json)
- [Independent tests](test_quantize.py)
- [Steps, formulas, results, and authorship](IMPLEMENTATION_NOTES.md)
- [Additional Pareto analysis including activation error](activation_pareto_analysis.json)

Run from lab06:

    python3 main.py
    python3 -m unittest -v test_quantize.py

The current instructor revision already supplied the seven algorithms. Our
contributions are explanatory comments, the README-compatible entry point,
independent validation, saved results, and analysis of the calibration tradeoff.
