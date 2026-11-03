"""Boundary tests and complete comparison with the instructor's sample."""
import json
import tempfile
import unittest
from pathlib import Path

import agreement as a


class AgreementTests(unittest.TestCase):
    def test_sample_report(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'results.json'
            a.main(Path(__file__).with_name('benchmark_input_data.json'), out)
            actual = json.loads(out.read_text(encoding='utf-8'))
            expected = json.loads(Path(__file__).with_name('sample_benchmark_results.json').read_text(encoding='utf-8'))
            self.assert_nested_equal(actual, expected)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, float):
            self.assertAlmostEqual(actual, expected, places=12)
        else:
            self.assertEqual(actual, expected)

    def test_latency_generator_and_interpolation(self):
        report = a.latency_report(x for x in [4, 0, 2, 1])
        self.assertEqual(report['p50'], 1.5)
        self.assertAlmostEqual(report['p95'], 3.7)
        self.assertEqual(report['worst'], 4)

    def test_single_timing(self):
        self.assertEqual(a.latency_report([0])['jitter'], 0)

    def test_bad_timings(self):
        for values in ([], [-1], [True], ['2'], [float('nan')], [float('inf')]):
            with self.subTest(values=values), self.assertRaises(a.AgreementError):
                a.latency_report(values)

    def test_speedup(self):
        self.assertEqual(a.speedup(10, 5), {'factor': 2, 'faster': True, 'saved_ms': 5})
        self.assertFalse(a.speedup(5, 5)['faster'])
        self.assertEqual(a.speedup(5, 10)['saved_ms'], -5)

    def test_bad_speedup(self):
        for base, other in [(0, 1), (1, 0), (-1, 1), (True, 1), (1, float('inf'))]:
            with self.subTest(base=base, other=other), self.assertRaises(a.AgreementError):
                a.speedup(base, other)

    def test_absolute_tie(self):
        self.assertEqual(a.max_abs_dev([3, -3], [1, -1])['index'], 0)

    def test_pair_validation(self):
        for func in (a.max_abs_dev, a.max_rel_dev):
            for left, right in [([], []), ([1], [1, 2]), ([1, float('nan')], [2, 3]), ([False], [0])]:
                with self.subTest(func=func.__name__), self.assertRaises(a.AgreementError):
                    func(left, right)

    def test_relative_floor(self):
        result = a.max_rel_dev([.002, -.002, .001], [0, -.001, .001])
        self.assertEqual(result['max_rel'], 2)
        self.assertEqual(result['floored'], 1)
        self.assertEqual(result['denom_min'], 0)
        self.assertEqual(result['index'], 0)

    def test_invalid_floor(self):
        for floor in [0, -1, True, float('nan')]:
            with self.subTest(floor=floor), self.assertRaises(a.AgreementError):
                a.max_rel_dev([1], [1], floor)

    def test_top1_ties_and_first_mismatch(self):
        result = a.top1_agreement([[2, 2], [0, 1], [1, 0]], [[2, 1], [1, 0], [0, 1]])
        self.assertEqual(result, {'n': 3, 'agree': 1, 'disagree': 2, 'rate': 1/3, 'first_disagreement': 1})

    def test_top1_invalid_shapes(self):
        for left, right in [([], []), ([[]], [[]]), ([[1]], [[1, 2]]), ([[1]], [[1], [1]]), ([[1]], [[float('inf')]])]:
            with self.subTest(left=left), self.assertRaises(a.AgreementError):
                a.top1_agreement(left, right)

    def test_margin_boundary(self):
        result = a.decision_margin([[2, 1], [3, 3], [5, 3]], 1)
        self.assertEqual(result['below_tol'], 2)
        self.assertEqual(result['argmin'], 1)
        self.assertEqual(result['median'], 1)
        self.assertFalse(result['safe'])

    def test_margin_negative_scores(self):
        self.assertTrue(a.decision_margin([[-3, -1, -4]], .5)['safe'])
        self.assertEqual(a.decision_margin([[3, 2], [4, 3]], 0)['argmin'], 0)

    def test_margin_invalid(self):
        for rows, tol in [([], 0), ([[1]], 0), ([[1, 2]], -1), ([[1, float('nan')]], 1), ([[1, 2]], True)]:
            with self.subTest(rows=rows, tol=tol), self.assertRaises(a.AgreementError):
                a.decision_margin(rows, tol)


if __name__ == '__main__':
    unittest.main()
