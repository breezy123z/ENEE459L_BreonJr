"""Regression checks using temporary filesystems; no hardware required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import probes as p


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def test_missing_evidence_and_command_isolation(self):
        with patch.object(p, 'run', side_effect=AssertionError('host command')):
            for name in ('module_model', 'memory_total_kb', 'root_source',
                         'nvme_present', 'pcie_link', 'thermal_zones', 'power_mode'):
                result = getattr(p, 'probe_' + name)(self.root)
                self.assertEqual(result['status'], 'unknown', name)
                self.assertIsNone(result['value'])

    def test_model_and_memory(self):
        self.put('proc/device-tree/model', 'Jetson\x00\n')
        self.put('proc/meminfo', 'MemTotal:       7789952 kB\n')
        self.assertEqual(p.probe_module_model(self.root)['value'], 'Jetson')
        self.assertEqual(p.probe_memory_total_kb(self.root)['value'], 7789952)
        self.put('proc/meminfo', 'MemTotal: invalid kB')
        self.assertEqual(p.probe_memory_total_kb(self.root)['status'], 'unknown')

    def test_fitted_does_not_mean_booted(self):
        self.put('sys/block/nvme0n1/device/model', 'Test SSD')
        for device, kind in [('/dev/mmcblk0p1', 'sd'), ('/dev/nvme0n1p1', 'nvme'),
                             ('/dev/sda1', 'other')]:
            self.put('proc/mounts', f'{device} / ext4 rw 0 0\n')
            self.assertEqual(p.probe_root_source(self.root)['kind'], kind)
            self.assertTrue(p.probe_nvme_present(self.root)['value'])

    def test_absent_drive(self):
        (self.root / 'sys/block').mkdir(parents=True)
        self.assertIs(p.probe_nvme_present(self.root)['value'], False)

    def test_pcie_selects_nvme(self):
        raw = ('0001:01:00.0 Network controller: WiFi\n'
               ' LnkCap: Speed 8GT/s, Width x1\n LnkSta: Speed 2.5GT/s, Width x1\n\n'
               '0004:01:00.0 Non-Volatile memory controller: SSD\n'
               ' LnkCap: Speed 16GT/s, Width x4\n LnkSta: Speed 8.0GT/s, Width x4')
        result = p.probe_pcie_link(self.root, raw)
        self.assertEqual(result['negotiated']['gen'], 3)
        self.assertEqual(result['capability']['gen'], 4)
        self.assertEqual(result['negotiated']['width'], 4)
        self.assertEqual(p.probe_pcie_link(self.root, 'LnkSta: Speed 8GT/s, Width x4')['status'], 'unknown')
        self.assertEqual(p.probe_pcie_link(self.root, raw.split('\n\n')[0])['status'], 'unknown')

    def test_thermal_units_and_invalid_sensor(self):
        self.put('sys/class/thermal/thermal_zone0/temp', '43000')
        self.put('sys/class/thermal/thermal_zone0/type', 'cpu-thermal')
        self.put('sys/class/thermal/thermal_zone1/temp', 'invalid')
        self.put('sys/class/thermal/thermal_zone2/temp', '44500')
        result = p.probe_thermal_zones(self.root)
        self.assertEqual(result['value'], 44.5)
        self.assertEqual(len(result['zones']), 2)
        self.assertEqual(result['zones'][0]['temp_c'], 43.0)

    def test_power_mode(self):
        result = p.probe_power_mode(self.root, 'NV Power Mode: MAXN SUPER\n2\n')
        self.assertEqual((result['value'], result['mode_id']), ('MAXN SUPER', 2))
        self.assertEqual(p.probe_power_mode(self.root, 'NV Power Mode: 25W')['status'], 'unknown')


if __name__ == '__main__':
    unittest.main()
