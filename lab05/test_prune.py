import json
import unittest
from pathlib import Path
from main import generate_all_prune_outputs
from tensors import Tensor, TensorError
import prune as p


class PruningTests(unittest.TestCase):
    def setUp(self):
        self.t = Tensor('w', (4, 2), (1., -1., 2., -2., 3., -3., 4., -4.))

    def test_full_reference(self):
        root = Path(__file__).parent
        self.assertEqual(generate_all_prune_outputs(root / 'model.json'),
                         json.loads((root / 'sample_prune_outputs.json').read_text()))

    def test_magnitude_ties(self):
        self.assertEqual(p.magnitude_mask(self.t, .375), (0, 0, 0, 1, 1, 1, 1, 1))

    def test_ratio_endpoints(self):
        self.assertEqual(p.magnitude_mask(self.t, 0), (1,) * 8)
        self.assertEqual(p.magnitude_mask(self.t, 1), (0,) * 8)
        self.assertEqual(p.channel_keep(self.t, 1), (3,))

    def test_round_half_up(self):
        self.assertEqual(p._drop_count(10, .25), 3)

    def test_channel_norms(self):
        self.assertEqual(p.channel_keep(self.t, .5), (2, 3))
        self.assertEqual(p.channel_keep(self.t, .5, p=1), (2, 3))

    def test_mask_no_mutation(self):
        out = p.apply_mask(self.t, (0, 1) * 4)
        self.assertEqual(out.shape, self.t.shape)
        self.assertEqual(out.data, (0., -1., 0., -2., 0., -3., 0., -4.))
        self.assertEqual(self.t.data[0], 1.)
        self.assertEqual(out.parameters, 8)

    def test_drop_original_order(self):
        out = p.drop_channels(self.t, (3, 1))
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(out.data, (2., -2., 4., -4.))

    def test_invalid_requests(self):
        for ratio in (-.1, 1.1, float('nan')):
            with self.assertRaises(TensorError): p.magnitude_mask(self.t, ratio)
        for keep in ((), (0, 0), (4,), (-1,)):
            with self.assertRaises(TensorError): p.drop_channels(self.t, keep)
        for mask in ((1,), (2,) * 8):
            with self.assertRaises(TensorError): p.apply_mask(self.t, mask)
        with self.assertRaises(TensorError): p.channel_keep(self.t, .5, p=0)
        with self.assertRaises(TensorError): p.bytes_stored([self.t], 'bad')
        with self.assertRaises(TensorError): p.bytes_stored([self.t], 'masked', 'bad')
        with self.assertRaises(TensorError): p.sweep_model({'name':'x','tensors':[self.t]}, [0], ['bad'])

    def test_storage_hand_calculation(self):
        t = Tensor('q', (9,), (0.,) * 5 + (1.,) * 4, 'fp16')
        self.assertEqual(p.bytes_stored([t]), 18)
        self.assertEqual(p.bytes_stored([t], 'masked'), 36)
        self.assertEqual(p.bytes_stored([t], 'masked', 'bitmap'), 20)
        self.assertEqual(p.bytes_stored([t], 'sparse'), 24)

    def test_classification_priority(self):
        pattern = Tensor('w', (4, 2), (0., 0., 1., 2.) * 2)
        self.assertEqual(p.classify_removal([self.t], [pattern]), 'patterned')
        self.assertEqual(p.classify_removal([self.t], [pattern], 'sparse'), 'stored sparse')
        self.assertEqual(p.classify_removal([self.t], [p.drop_channels(self.t, (1,))], 'sparse'), 'structurally absent')
        self.assertEqual(p.classify_removal([self.t], [self.t]), 'dense')
        masked = p.apply_mask(self.t, (0,) + (1,) * 7)
        self.assertEqual(p.classify_removal([self.t], [masked]), 'masked')

    def test_row_zeros_vs_removed(self):
        fine = p.apply_mask(self.t, p.magnitude_mask(self.t, .5))
        row = p.sparsity_row('x', .5, 'fine', [self.t], [fine], 'masked')
        self.assertEqual(row['values_zeroed'], 4)
        self.assertEqual(row['achieved_reduction'], 0)
        row = p.sparsity_row('x', 1, 'channel', [self.t], [p.drop_channels(self.t, (3,))])
        self.assertEqual(row['values_zeroed'], 0)
        self.assertEqual(row['achieved_reduction'], .75)

    def test_sweep_order_and_fresh_baseline(self):
        rows = p.sweep_model({'name':'x','tensors':[self.t]}, [1, 0, .5])
        self.assertEqual([r['nominal_ratio'] for r in rows], [0, .5, 1] * 2)
        self.assertEqual([r['granularity'] for r in rows], ['fine'] * 3 + ['channel'] * 3)
        self.assertEqual(rows[4]['parameters_after'], 4)
        self.assertTrue(all(r['storage']=='sparse' for r in p.sweep_model({'name':'x','tensors':[self.t]}, [0], storage='sparse')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
