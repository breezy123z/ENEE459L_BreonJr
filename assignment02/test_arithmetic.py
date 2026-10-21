import unittest
from unittest.mock import patch
from graph import Graph,Layer
from arithmetic import arithmetic_intensity, _layer_macs, _layer_parameters


class IntensityTests(unittest.TestCase):
    def test_linear_hand_calculation(self):
        layer=Layer('fc','linear',(3,),(2,),bias=True)
        row=arithmetic_intensity(Graph('g',(3,),[layer]))
        # 6 MACs -> 12 FLOPs; input 12 + 8 parameters*4 + output 8 = 52.
        self.assertEqual(row['per_layer']['fc'],{'flops':12,'bytes':52.,'intensity':.2308})
        self.assertEqual(row['value'],.2308)

    def test_external_input_dtype(self):
        layer=Layer('r','relu',(4,),(4,),act_dtype='fp32')
        self.assertEqual(arithmetic_intensity(Graph('g',(4,),[layer],precision='fp16'))['total_bytes'],24.)

    def test_previous_producer_dtype(self):
        layers=[Layer('a','relu',(4,),(4,),act_dtype='int8'),
                Layer('b','relu',(4,),(4,),act_dtype='fp16')]
        self.assertEqual(arithmetic_intensity(Graph('g',(4,),layers))['per_layer']['b']['bytes'],12.)

    def test_explicit_producer_not_previous(self):
        layers=[Layer('a','relu',(4,),(4,),act_dtype='fp16'),
                Layer('b','relu',(4,),(4,),act_dtype='int8'),
                Layer('c','relu',(4,),(4,),act_dtype='fp32',reads=('a',))]
        self.assertEqual(arithmetic_intensity(Graph('g',(4,),layers))['per_layer']['c']['bytes'],24.)

    def test_dual_input_add(self):
        layers=[Layer('a','relu',(4,),(4,),act_dtype='fp16'),
                Layer('b','relu',(4,),(4,),act_dtype='fp16'),
                Layer('sum','add',(4,),(4,),act_dtype='fp32',reads=('a','b'))]
        self.assertEqual(arithmetic_intensity(Graph('g',(4,),layers))['per_layer']['sum'],
                         {'flops':0,'bytes':32.,'intensity':0.})

    def test_mixed_add_uses_handout_first_dtype(self):
        layers=[Layer('a','relu',(4,),(4,),act_dtype='fp16'),
                Layer('b','relu',(4,),(4,),act_dtype='int8'),
                Layer('sum','add',(4,),(4,),act_dtype='fp32',reads=('a','b'))]
        # Literal PDF rule: 2 * first producer bytes, even with unequal dtypes.
        self.assertEqual(arithmetic_intensity(Graph('g',(4,),layers))['per_layer']['sum']['bytes'],32.)

    def test_bn_fp32_buffers(self):
        bn=Layer('bn','bn',(2,),(2,),weight_dtype='fp16',act_dtype='fp16')
        # input 8 + gamma/beta 8 + running stats 16 + output 4 = 36.
        self.assertEqual(arithmetic_intensity(Graph('g',(2,),[bn]))['total_bytes'],36.)

    def test_relu_zero_flops_retained(self):
        row=arithmetic_intensity(Graph('g',(10,),[Layer('r','relu',(10,),(10,))]))
        self.assertEqual(row['per_layer']['r'],{'flops':0,'bytes':80.,'intensity':0.})

    def test_aggregate_is_not_mean_of_ratios(self):
        layers=[Layer('fc','linear',(3,),(2,),bias=True),Layer('r','relu',(2,),(2,))]
        result=arithmetic_intensity(Graph('g',(3,),layers))
        self.assertEqual(result['total_bytes'],68.)
        self.assertEqual(result['total_flops'],12)
        self.assertEqual(result['value'],.1765)
        self.assertNotEqual(result['value'],round((.2308+0)/2,4))

    def test_fractional_int4_storage(self):
        layer=Layer('f','linear',(3,),(1,),weight_dtype='int4')
        result=arithmetic_intensity(Graph('g',(3,),[layer]))
        self.assertEqual(result['total_bytes'],17.5)
        self.assertEqual(result['total_flops'],6)

    def test_grouped_convolution(self):
        layer=Layer('c','conv',(4,2,2),(6,2,2),kernel=(3,3),groups=2,bias=True)
        self.assertEqual(_layer_parameters(layer),114)
        self.assertEqual(_layer_macs(layer),432)
        self.assertEqual(arithmetic_intensity(Graph('g',layer.in_shape,[layer]))['total_bytes'],616.)

    def test_supplied_mac_helper_is_used(self):
        layer=Layer('x','relu',(1,),(1,))
        with patch('arithmetic._layer_macs',return_value=7):
            self.assertEqual(arithmetic_intensity(Graph('g',(1,),[layer]))['total_flops'],14)

    def test_empty_and_zero_byte_graph(self):
        empty=arithmetic_intensity(Graph('empty',(1,),[]))
        self.assertEqual(empty['value'],0.)
        self.assertEqual(empty['total_bytes'],0.)
        zero=arithmetic_intensity(Graph('zero',(0,),[Layer('r','relu',(0,),(0,))]))
        self.assertEqual(zero['per_layer']['r']['intensity'],0.)

    def test_record_contract(self):
        result=arithmetic_intensity(Graph('named',(1,),[]))
        self.assertEqual(set(result),{'type','value','message','per_layer','total_flops','total_bytes','flops_per_mac','cache_reuse_assumed','note'})
        self.assertEqual(result['type'],'computed')
        self.assertEqual(result['message'],'named: FLOPs over bytes, every byte counted once')
        self.assertEqual(result['note'],'no fusion and no cache reuse; both would lower the byte column')
        self.assertIs(result['cache_reuse_assumed'],False)
        self.assertEqual(result['flops_per_mac'],2)
        self.assertIsInstance(result['value'],float)
        self.assertIsInstance(result['total_bytes'],float)

    def test_bad_dependencies_and_duplicates(self):
        with self.assertRaises(ValueError):
            arithmetic_intensity(Graph('g',(1,),[Layer('b','relu',(1,),(1,),reads=('future',))]))
        layer=Layer('r','relu',(1,),(1,))
        with self.assertRaises(ValueError): arithmetic_intensity(Graph('g',(1,),[layer,layer]))


if __name__=='__main__': unittest.main(verbosity=2)
