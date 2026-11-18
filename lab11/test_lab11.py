"""Tests for Lab 11 decision, timing, and event boundaries."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

stream = load('stream', '0_stream.py')
decide = load('decide', '1_decide.py')
events = load('events', '2_events.py')

class Lab11Tests(unittest.TestCase):
    def setUp(self):
        self.plan = stream.build_session_from_wav(ROOT / 'test_voice.wav')['plan']

    def test_plan(self):
        stream._validate_plan(self.plan)
        stream._validate_plan({**self.plan, 'collar_s': 0, 'refractory_s': 0})
        for changes in ({'k': 6, 'n': 5}, {'leave': .8, 'enter': .6},
                        {'hop_s': 2, 'window_s': 1}, {'sample_rate': True},
                        {'ring_blocks': 0}, {'threshold': 1.1}, {'refractory_s': -1},
                        {'stated': {'ring_ms': 1, 'latency_ms': 1, 'decisions_per_hour': 0}}):
            with self.subTest(changes=changes), self.assertRaises(stream.SchemaError):
                stream._validate_plan({**self.plan, **changes})

    def test_confusion(self):
        self.assertEqual(decide.confusion([.5, .6, .4, .1], [1, 0, 1, 0], .5),
                         {'tp': 1, 'fp': 1, 'tn': 1, 'fn': 1, 'windows': 4})
        for scores, labels in [([.5], []), ([.5], [True]), ([1.1], [1]), ([], [])]:
            with self.subTest(scores=scores, labels=labels), self.assertRaises(decide.DecideError):
                decide.confusion(scores, labels, .5)

    def test_rates(self):
        result = decide.rates({'tp': 0, 'fp': 0, 'tn': 10, 'fn': 0})
        self.assertIsNone(result['precision'])
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['f1'])
        self.assertEqual(result['fpr'], 0)
        self.assertEqual(decide.rates({'tp': 0, 'fp': 2, 'tn': 3, 'fn': 1})['f1'], 0)
        self.assertAlmostEqual(decide.rates({'tp': 2, 'fp': 1, 'tn': 3, 'fn': 1})['f1'], 2/3)

    def test_base_rate(self):
        result = decide.precision_at_base_rate(.9, .001, 2, 1, .1)
        self.assertEqual(result['positives'], 2)
        self.assertEqual(result['negatives'], 35998)
        self.assertAlmostEqual(result['precision'], 1.8 / (1.8 + 35.998))
        self.assertIsNone(decide.precision_at_base_rate(.9, 0, 0, 1, .1)['precision'])
        for args in [(1, 0, 36001, 1, .1), (1, 0, True, 1, .1), (1, 0, 1, 0, .1), (1, 0, 1, 1, 0)]:
            with self.subTest(args=args), self.assertRaises(decide.DecideError):
                decide.precision_at_base_rate(*args)

    def test_vote(self):
        self.assertEqual(events.k_of_n([True, True, True, False, False, False], 3, 5),
                         [False, False, True, True, True, False])
        self.assertEqual(events.k_of_n([], 1, 1), [])
        for flags, k, n in [([True], 2, 1), ([True], 1, 0), ([1], 1, 1)]:
            with self.subTest(flags=flags), self.assertRaises(events.EventError):
                events.k_of_n(flags, k, n)

    def test_hysteresis(self):
        self.assertEqual(events.hysteresis([.59, .6, .5, .4, .39, .6], .6, .4),
                         [False, True, True, True, False, True])
        with self.assertRaises(events.EventError):
            events.hysteresis([.5], .4, .6)

    def test_refractory(self):
        flags = [True, False, True, False, True, False, True]
        self.assertEqual(events.refractory(flags, 1, 3), [0, 4])
        self.assertEqual(events.refractory(flags, 1, 2), [0, 2, 4, 6])
        self.assertEqual(events.refractory(flags, 1, 0), [0, 2, 4, 6])

    def test_match(self):
        result = events.match_events([.5, 1.5, 4], [1, 1, 6], .5)
        self.assertEqual(result['matches'], [
            {'truth_s': 1., 'predicted_s': .5, 'delta_s': -.5},
            {'truth_s': 1., 'predicted_s': 1.5, 'delta_s': .5}])
        self.assertEqual((result['hits'], result['misses'], result['false_alarms']), (2, 1, 1))
        self.assertEqual(events.match_events([], [1], 0)['misses'], 1)
        for args in [([2, 1], [], 1), ([-1], [], 1), ([], [], -1)]:
            with self.subTest(args=args), self.assertRaises(events.EventError):
                events.match_events(*args)

    def test_reference_reports(self):
        for i in range(3):
            actual = json.loads((ROOT / f'output{i}_session.json').read_text(encoding='utf-8'))
            expected = json.loads((ROOT / f'sample_output{i}_session.json').read_text(encoding='utf-8'))
            for doc in (actual, expected):
                doc.pop('generated', None)
                doc['audio'].pop('source', None)
            self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
