# Lab 6 submission

- [Quantization functions and explanatory comments](quantize.py)
- [Generated report](quantization_test_results.json)
- [Independent tests](test_quantize.py)
- [Steps, formulas, results, and authorship](IMPLEMENTATION_NOTES.md)
- [Additional Pareto analysis including activation error](activation_pareto_analysis.json)

Run from lab06:

    python3 main.py
    python3 -m unittest -v test_quantize.py

The seven assignment functions were rewritten from the supplied handout after
initially reviewing the provided implementation. The original supplied version
remains in Git history at b7822d9. Helpers, data types, and sample experiment
remain instructor-provided. All 19 local tests pass, including exact comparison
with the sample output. No later instructor changes were pulled for the rewrite.
