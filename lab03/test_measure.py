"""Deterministic tests: these do not create or claim real Jetson benchmarks."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bench import Bench, CommandResult
import measure as m


class MeasureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bench = Bench(
            workload=SimpleNamespace(run=lambda: None, synchronize=lambda: None),
            telemetry=self.root,
            runner=lambda argv: CommandResult(tuple(argv), 0, "NV Power Mode: 25W\n1\n"),
        )

    def put(self, path, text):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_sync_order_and_time_units(self):
        events = []
        times = iter([10, 2_000_010, 3_000_010, 6_000_010])
        def clock():
            events.append("clock")
            return next(times)
        workload = SimpleNamespace(run=lambda: events.append("run"),
                                   synchronize=lambda: events.append("sync"))
        result = m.run_timed_iterations(Bench(workload=workload, clock=clock), 2)
        self.assertEqual(result, [2.0, 3.0])
        self.assertEqual(events, ["sync"] + ["clock", "run", "sync", "clock"] * 2)

    def test_invalid_repeat_count(self):
        for repeats in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                m.run_timed_iterations(self.bench, repeats)
        self.assertEqual(m.run_timed_iterations(self.bench, 0), [])

    def test_backwards_clock(self):
        ticks = iter([100, 50])
        with self.assertRaises(ValueError):
            m.run_timed_iterations(Bench(workload=self.bench.workload, clock=lambda: next(ticks)), 1)

    def test_adaptive_warmup(self):
        for count in (0, 10, 22, 35):
            samples = [30.0] * count + [10.0] * (100 - count)
            result = m.find_warmup_boundary(samples)
            self.assertEqual(result["value"], count)
            self.assertEqual(result["retained"], 100 - count)

    def test_warmup_strict_threshold_and_prefix(self):
        samples = [20, 15, 30] + [10] * 97
        self.assertEqual(m.find_warmup_boundary(samples)["value"], 1)

    def test_drift_flagged_separately(self):
        samples = [100 - i * 0.8 for i in range(100)]
        self.assertEqual(m.find_warmup_boundary(samples)["value"], 50)
        self.assertFalse(m.is_stationary(samples)["value"])

    def test_small_and_invalid_samples(self):
        self.assertEqual(m.find_warmup_boundary([100, 1])["status"], "unknown")
        for bad in ([float("nan")], [float("inf")], [-1]):
            self.assertEqual(m.summarize(bad)["status"], "unknown")
            self.assertEqual(m.find_warmup_boundary(bad)["status"], "unknown")
        self.assertEqual(m.is_stationary([1] * 3)["status"], "unknown")
        self.assertEqual(m.is_multimodal([1] * 3)["status"], "unknown")

    def test_summary_interpolation(self):
        result = m.summarize([1, 2, 3, 4])
        self.assertEqual(result, {"n": 4, "mean": 2.5, "std": 1.291,
                                 "min": 1, "max": 4, "p50": 2.5, "p95": 3.85, "p99": 3.97})
        self.assertEqual(m.summarize([2])["std"], 0.0)
        self.assertEqual(m.summarize([1, 2])["std"], 0.0)
        self.assertEqual(m.summarize([]), {"n": 0, **dict.fromkeys(("mean", "std", "min", "max", "p50", "p95", "p99"))})

    def test_constant_and_two_modes(self):
        self.assertEqual(m.is_multimodal([10] * 100)["status"], "unknown")
        result = m.is_multimodal([10 + i * 0.01 for i in range(50)] + [30 + i * 0.01 for i in range(50)])
        self.assertTrue(result["value"])
        self.assertEqual([mode["n"] for mode in result["modes"]], [45, 45])
        json.dumps(result, allow_nan=False)

    def test_outlier_is_not_second_mode(self):
        self.assertFalse(m.is_multimodal([10 + i * 0.01 for i in range(99)] + [100])["value"])

    def test_power_profile_and_clock_limits(self):
        self.put(m.CPUFREQ_MIN, "1000")
        self.put(m.CPUFREQ_MAX, "1000")
        result = m.probe_power_state(self.bench)
        self.assertEqual(result["value"], "25W")
        self.assertEqual(result["mode_index"], 1)
        self.assertTrue(result["jetson_clocks"])
        self.put(m.CPUFREQ_MAX, "2000")
        self.assertFalse(m.probe_power_state(self.bench)["jetson_clocks"])

    def test_unknown_clocks(self):
        self.assertIsNone(m.probe_power_state(self.bench)["jetson_clocks"])

    def test_failed_power_command(self):
        bench = Bench(workload=self.bench.workload, telemetry=self.root,
                      runner=lambda argv: CommandResult(tuple(argv), 1, "", error="denied"))
        self.assertEqual(m.probe_power_state(bench)["status"], "unknown")

    def test_temperature_peak_and_gpu_units(self):
        self.put(m.THERMAL_ZONES + "/thermal_zone0/temp", "43000")
        self.put(m.THERMAL_ZONES + "/thermal_zone0/type", "cpu")
        self.put(m.THERMAL_ZONES + "/thermal_zone1/temp", "46000")
        self.put(m.THERMAL_ZONES + "/thermal_zone1/type", "soc")
        self.put(m.THERMAL_ZONES + "/thermal_zone2/temp", "-1000")
        self.put(m.GPU_LOAD_CANDIDATES[0], "221")
        result = m.probe_telemetry(self.bench)
        self.assertEqual(result["temperature_c"]["value"], 46)
        self.assertEqual(result["temperature_c"]["zones_read"], 2)
        self.assertEqual(result["gpu_utilization_percent"]["value"], 22.1)

    def test_missing_sensors(self):
        result = m.probe_telemetry(self.bench)
        self.assertTrue(all(item["status"] == "unknown" for item in result.values()))

    def test_voltage_not_mislabeled_as_power(self):
        self.put(m.POWER_RAIL_CANDIDATES[0], "5000")
        self.assertEqual(m.probe_telemetry(self.bench)["power_mw"]["status"], "unknown")

    def test_board_voltage_times_current(self):
        base = "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon9/"
        self.put(base + "in1_label", "VDD_IN")
        self.put(base + "in1_input", "5000")
        self.put(base + "curr1_input", "1200")
        self.assertEqual(m.probe_telemetry(self.bench)["power_mw"]["value"], 6000)

    def test_legacy_power_and_gpu_fallback(self):
        original_read = m.read_text
        def read(root, path):
            if path == m.POWER_RAIL_CANDIDATES[1]:
                return "6400"
            return original_read(root, path)
        mock = patch.object(m, "read_text", side_effect=read)
        mock.start()
        self.addCleanup(mock.stop)
        self.put(m.GPU_LOAD_CANDIDATES[0], "invalid")
        self.put(m.GPU_LOAD_CANDIDATES[1], "0")
        result = m.probe_telemetry(self.bench)
        self.assertEqual(result["power_mw"]["value"], 6400)
        self.assertEqual(result["gpu_utilization_percent"]["value"], 0)

    def test_reference_schema(self):
        sample = json.loads(Path(__file__).with_name("sample_system_report.json").read_text())
        self.put(m.CPUFREQ_MIN, "1000")
        self.put(m.CPUFREQ_MAX, "2000")
        values = [10 + i / 100 for i in range(100)]
        self.assertEqual(set(m.summarize(values)), set(sample["summarize_setup"]))
        self.assertEqual(set(m.find_warmup_boundary(values)), set(sample["warmup_boundary"]))
        self.assertEqual(set(m.is_multimodal(values)), set(sample["is_multimodal"]))
        self.assertEqual(set(m.probe_power_state(self.bench)), set(sample["probe_power_state"]))

    def test_input_unchanged(self):
        values = [30, 20, 10] + [10] * 97
        before = values[:]
        m.summarize(values)
        m.find_warmup_boundary(values)
        m.is_multimodal(values)
        self.assertEqual(values, before)


if __name__ == "__main__":
    unittest.main()
