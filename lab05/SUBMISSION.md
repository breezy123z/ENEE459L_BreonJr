# Lab 5 submission

- [Implemented pruning functions](prune.py)
- [Generated results](prune_outputs.json)
- [Tests](test_prune.py) and [saved local test run](test-results.txt)
- [Steps and Jetson validation](IMPLEMENTATION_NOTES.md)
- [Eight-chart report](charts/README.md)
- [Download charts and data](lab05-chart-bundle.zip)

All 12 tests passed locally and on the Jetson. The instructor comparator
reported a 100% reference match on both. The original starter is preserved
in this branch's Git history before the implementation commit.

Run from lab05:

    python3 main.py
    python3 compare_json.py sample_prune_outputs.json prune_outputs.json
    python3 -m unittest -v test_prune.py

Submission branch: solution5.
