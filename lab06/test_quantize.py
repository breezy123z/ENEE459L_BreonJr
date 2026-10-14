"""Independent arithmetic and experiment checks for Lab 6."""
import json
import math
from pathlib import Path
import runpy
import unittest

from tensors import Tensor, TensorError, Sample, load_model, materialise_samples
from quantize import (quantize_tensor, dequantize, quantization_error, bytes_stored,
                      effective_bits, precision_row, pareto_front, to_fp16,
                      select_calibration, as_tensor, Quantized)
from quantize_helper import _round_half_up, _range_of

ROOT = Path(__file__).parent


class QuantizationTests(unittest.TestCase):
    def test_complete_reference(self):
        runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')
        self.assertEqual(json.loads((ROOT / 'quantization_test_results.json').read_text()),
                         json.loads((ROOT / 'sample_quantization_results.json').read_text()))

    def test_symmetric_hand_example(self):
        t = as_tensor('x', [-2, -1, 0, 1, 2])
        q = quantize_tensor(t, bits=3)
        self.assertEqual(q.codes, (-3, -1, 0, 2, 3))
        self.assertEqual(q.zeros, (0,))
        self.assertAlmostEqual(q.scales[0], 2 / 3)
        self.assertEqual(q.clipped, 0)
        self.assertEqual(dequantize(q).data, (-2., -.66666667, 0., 1.33333333, 2.))
        self.assertEqual(t.data, (-2., -1., 0., 1., 2.))

    def test_asymmetric_positive_range(self):
        q = quantize_tensor(as_tensor('x', [0, 1, 2, 3]), bits=2, symmetric=False)
        self.assertEqual(q.codes, (-2, -1, 0, 1))
        self.assertEqual(q.scales, (1.,))
        self.assertEqual(q.zeros, (-2,))
        self.assertEqual(dequantize(q).data, (0., 1., 2., 3.))

    def test_clipping_given_range(self):
        q = quantize_tensor(as_tensor('x', [-2, -1, 0, 1, 2]), bits=3, given_range=(-1, 1))
        self.assertEqual(q.codes, (-3, -3, 0, 3, 3))
        self.assertEqual(q.clipped, 2)
        self.assertEqual(q.clipped_fraction, .4)

    def test_per_channel_scales(self):
        t = Tensor('x', (2, 2), (-1., 1., -100., 100.))
        pc = quantize_tensor(t, bits=3, granularity='per_channel')
        pt = quantize_tensor(t, bits=3)
        self.assertEqual(pc.codes, (-3, 3, -3, 3))
        self.assertEqual(dequantize(pc).data, t.data)
        self.assertGreater(quantization_error([t], [dequantize(pt)])['relative'], 0)

    def test_zero_and_constant_ranges(self):
        for data in ([0, 0, 0], [2, 2, 2], [-2, -2, -2]):
            for symmetric in (True, False):
                q = quantize_tensor(as_tensor('x', data), symmetric=symmetric)
                self.assertTrue(all(s > 0 for s in q.scales))
                self.assertEqual(dequantize(q).data, tuple(float(x) for x in data))

    def test_signed_half_up(self):
        self.assertEqual([_round_half_up(x) for x in (-2.5, -.5, .5, 2.5)], [-2, 0, 1, 3])

    def test_error_by_hand(self):
        result = quantization_error([as_tensor('x', [3, 4])], [as_tensor('x', [0, 4])])
        self.assertEqual(result, {'frobenius': 3., 'relative': .6, 'max_abs': 3.})
        self.assertEqual(quantization_error([as_tensor('z', [0, 0])], [as_tensor('z', [0, 0])])['relative'], 0)

    def test_storage_metadata(self):
        t = Tensor('x', (2, 3), (1.,) * 6)
        self.assertEqual(bytes_stored([t], 'fp32'), 24)
        self.assertEqual(bytes_stored([t], 'fp16'), 12)
        self.assertEqual(bytes_stored([t], 'int8'), 14)
        self.assertEqual(bytes_stored([t], 'int8', 'per_channel'), 22)
        self.assertEqual(effective_bits([t], 'int8'), 18.6667)

    def test_fp16_conversion_and_overflow(self):
        t = as_tensor('x', [1.0001, .5])
        self.assertEqual(to_fp16(t).data, (1., .5))
        with self.assertRaises(TensorError): to_fp16(as_tensor('x', [70000]))

    def test_calibration_selection(self):
        samples = tuple(Sample('a', i % 4, (float(i),)) for i in range(12))
        self.assertEqual(select_calibration(samples, 'one_class'), (0., 4., 8.))
        self.assertEqual(select_calibration(samples, 'two_class'), (0., 1., 4., 5., 8., 9.))
        self.assertEqual(select_calibration(samples, 'first_8'), tuple(float(i) for i in range(8)))
        self.assertEqual(len(select_calibration(samples)), 12)

    def test_bad_calibration_same_weight_storage(self):
        t = as_tensor('w', [-1, 0, 1])
        config = {'label':'int8', 'precision':'int8', 'granularity':'per_channel'}
        good = precision_row('x', config, [t], [0, 1, 5], [0, 1, 5])
        bad = precision_row('x', config, [t], [0, 1], [0, 1, 5])
        self.assertEqual(good['bytes_stored'], bad['bytes_stored'])
        self.assertEqual(good['weight_error'], bad['weight_error'])
        self.assertGreater(bad['activation_error'], good['activation_error'])
        self.assertEqual(bad['clipped_fraction'], round(1/3, 8))

    def test_positive_activations_asymmetric_advantage(self):
        t = as_tensor('a', [0, 1, 2, 3])
        sym = dequantize(quantize_tensor(t, bits=2))
        asym = dequantize(quantize_tensor(t, bits=2, symmetric=False))
        self.assertLess(quantization_error([t], [asym])['relative'], quantization_error([t], [sym])['relative'])

    def test_percentile_and_mse(self):
        vals = [float(i) for i in range(100)]
        self.assertEqual(_range_of(vals, 'percentile', 80), (10., 89.))
        lo, hi = _range_of(vals, 'mse')
        self.assertTrue(0 <= lo < hi <= 99)
        q = quantize_tensor(as_tensor('x', vals), clip='percentile', percentile=80)
        self.assertGreater(q.clipped, 0)

    def test_pareto_dominance_and_ties(self):
        pts = [{'size':1, 'error':2}, {'size':2, 'error':1}, {'size':3, 'error':3}, {'size':1, 'error':2}]
        self.assertEqual(pareto_front(pts, {'size':-1, 'error':-1}), (0, 1, 3))
        self.assertEqual(pareto_front([{'score':1}, {'score':2}], {'score':1}), (1,))

    def test_activation_objective_rejects_biased_policy(self):
        ref = json.loads((ROOT/'sample_quantization_results.json').read_text())
        objectives = dict(ref['objectives'], activation_error=-1)
        self.assertEqual(pareto_front(ref['configurations'], objectives), (0, 1, 2, 3))

    def test_explicit_label_required(self):
        t = as_tensor('x', [1, 2])
        for config in ({'precision':'fp32'}, {'label':''}, {'label':'   '}, {'label':None}):
            with self.subTest(config=config):
                with self.assertRaises(TensorError):
                    precision_row('x', config, [t], [0, 1], [0, 1])

    def test_missing_quantized_metadata(self):
        q = Quantized('x', (2, 1), (1, 2), (1., 1.), (0,), 8, 'per_channel', True)
        with self.assertRaises(TensorError):
            dequantize(q)

    def test_invalid_inputs(self):
        t = as_tensor('x', [1, 2])
        for kwargs in ({'bits':1}, {'granularity':'bad'}, {'clip':'bad'}, {'clip':'percentile','percentile':0}):
            with self.assertRaises(TensorError): quantize_tensor(t, **kwargs)
        with self.assertRaises(TensorError): bytes_stored([t], 'bad')
        with self.assertRaises(TensorError): effective_bits([])
        with self.assertRaises(TensorError): quantization_error([t], [])
        with self.assertRaises(TensorError): quantization_error([t], [as_tensor('x', [1])])
        with self.assertRaises(TensorError): select_calibration([], 'all')
        with self.assertRaises(TensorError): select_calibration([], 'bad')
        with self.assertRaises(TensorError): pareto_front([], {'x':-1})
        with self.assertRaises(TensorError): pareto_front([{'x':1}], {})
        with self.assertRaises(TensorError): pareto_front([{}], {'x':-1})


if __name__ == '__main__':
    unittest.main(verbosity=2)
