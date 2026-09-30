"""Independent formula checks and liveness examples for Lab 4."""
import json
import unittest
from dataclasses import replace
from pathlib import Path

from graph import Graph, Layer, computed, measured, unknown
from complexity import count_parameters, model_size_bytes, count_activations, to_flops
from main import build_synthetic_graph, main


class ComplexityTests(unittest.TestCase):
    def test_reference_report(self):
        main()
        root = Path(__file__).parent
        self.assertEqual(json.loads((root / "complexity_results.json").read_text()),
                         json.loads((root / "sample_complexity_results.json").read_text()))

    def test_grouped_conv(self):
        layer = Layer("c", "conv", (4, 8, 8), (6, 8, 8), kernel=(3, 3), groups=2, bias=True)
        graph = Graph("grouped", layer.in_shape, [layer])
        self.assertEqual(count_parameters(graph)["value"], 6 * 2 * 3 * 3 + 6)

    def test_depthwise(self):
        layer = Layer("c", "conv", (4, 8, 8), (4, 8, 8), kernel=(3, 3), groups=4)
        self.assertEqual(count_parameters(Graph("dw", layer.in_shape, [layer]))["value"], 36)

    def test_linear_bias(self):
        layer = Layer("fc", "linear", (5,), (3,), bias=True)
        self.assertEqual(count_parameters(Graph("fc", (5,), [layer]))["value"], 18)

    def test_bn_buffers_fp32(self):
        layer = Layer("bn", "bn", (4, 8, 8), (4, 8, 8), weight_dtype="fp16")
        graph = Graph("bn", layer.in_shape, [layer])
        self.assertEqual(count_parameters(graph)["value"], 8)
        result = model_size_bytes(graph)
        self.assertEqual(result["value"], 48)
        self.assertEqual(result["per_dtype"], {"fp16": 16, "fp32": 32})
        self.assertEqual(result["buffer_bytes"], 32)

    def test_int4_not_truncated(self):
        layer = Layer("fc", "linear", (3,), (1,), weight_dtype="int4")
        self.assertEqual(model_size_bytes(Graph("q", (3,), [layer]))["value"], 1.5)

    def test_input_and_output_coexist(self):
        layer = Layer("r", "relu", (10,), (10,))
        result = count_activations(Graph("one", (10,), [layer]))
        self.assertEqual(result["value"], 80)
        self.assertEqual(result["total_elements"], 10)
        self.assertEqual(result["peak_elements"], 20)

    def test_chain_releases_dead_tensors(self):
        layers = [Layer(str(i), "relu", (10,), (10,)) for i in range(4)]
        self.assertEqual(count_activations(Graph("chain", (10,), layers))["value"], 80)

    def test_skip_kept_until_add(self):
        layers = [Layer("a", "relu", (10,), (10,)),
                  Layer("b", "relu", (10,), (10,)),
                  Layer("c", "relu", (10,), (10,)),
                  Layer("d", "add", (10,), (10,), reads=("a", "c"))]
        result = count_activations(Graph("skip", (10,), layers))
        self.assertEqual(result["value"], 120)
        self.assertEqual(result["total_elements"], 40)

    def test_mixed_activation_precision(self):
        layers = [Layer("a", "relu", (10,), (10,), act_dtype="int8"),
                  Layer("b", "relu", (10,), (10,), act_dtype="fp16")]
        result = count_activations(Graph("mixed", (10,), layers))
        self.assertEqual(result["value"], 50)
        self.assertEqual(result["total_bytes"], 30)

    def test_invalid_graphs(self):
        good = Layer("c", "conv", (4, 8, 8), (4, 8, 8), kernel=(3, 3))
        for bad in (replace(good, groups=3), replace(good, kernel=None),
                    replace(good, weight_dtype="bad"), replace(good, reads=("future",)),
                    replace(good, in_shape=(0, 8, 8))):
            graph = Graph("bad", bad.in_shape, [bad])
            for func in (count_parameters, model_size_bytes, count_activations):
                self.assertEqual(func(graph)["status"], "unknown")

    def test_duplicate_names(self):
        layer = Layer("r", "relu", (10,), (10,))
        self.assertEqual(count_activations(Graph("bad", (10,), [layer, layer]))["status"], "unknown")

    def test_empty_graph(self):
        graph = Graph("empty", (10,), [])
        self.assertEqual(count_parameters(graph)["value"], 0)
        self.assertEqual(model_size_bytes(graph)["value"], 0)
        self.assertEqual(count_activations(graph)["value"], 40)

    def test_flop_conventions(self):
        macs = measured(100, "test", per_layer={"a": 40, "b": 60})
        for convention, factor in (("mac_is_two_flops", 2), ("mac_is_one_flop", 1)):
            result = to_flops(macs, convention)
            self.assertEqual(result["value"], 100 * factor)
            self.assertEqual(result["per_layer"]["a"], 40 * factor)
            self.assertEqual(result["convention"], convention)
        self.assertEqual(macs["value"], 100)

    def test_unknown_flops(self):
        for macs in (unknown("x", "missing"), None, computed(-1, "x"),
                     computed(float("nan"), "x"), computed(1.5, "x")):
            self.assertEqual(to_flops(macs)["status"], "unknown")
        self.assertEqual(to_flops(computed(10, "x"), "bad")["status"], "unknown")


if __name__ == "__main__":
    unittest.main()

