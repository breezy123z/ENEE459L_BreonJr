"""Test Lab 2 with fake files and libraries, without changing the Jetson."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from env import Env, ModuleNotAvailable, PythonRuntime
import probes_student as p


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def env(self, modules=None, venv=False):
        def importer(name):
            if name not in (modules or {}):
                raise ModuleNotAvailable(name)
            return modules[name]
        return Env(root=self.root, importer=importer,
                   python=PythonRuntime(prefix='/venv' if venv else '/usr', base_prefix='/usr'))

    def put(self, path, text):
        file = self.root / path.lstrip('/')
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)

    def test_missing_evidence(self):
        for probe in (p.probe_torch, p.probe_cuda, p.probe_opencv, p.probe_tensorrt, p.probe_l4t):
            with self.subTest(probe=probe.__name__):
                result = probe(self.env())
                self.assertEqual(result['status'], 'unknown')
                self.assertIsNone(result['value'])
                self.assertTrue(result['detail'])

    def test_torch_gpu_and_nvidia_tag(self):
        torch = NS(__version__='2.5.0a0+872d972e41.nv24.08', version=NS(cuda='12.6'),
                   cuda=NS(is_available=lambda: True, get_device_name=lambda index: 'Orin'))
        result = p.probe_torch(self.env({'torch': torch}))
        self.assertTrue(result['version']['nvidia_build'])
        self.assertTrue(result['cuda_available'])
        self.assertEqual(result['device_name'], 'Orin')

    def test_torch_cpu_build(self):
        torch = NS(__version__='2.5.0', version=NS(cuda=None), cuda=NS(is_available=lambda: False))
        result = p.probe_torch(self.env({'torch': torch}))
        self.assertFalse(result['cuda_available'])
        self.assertIsNone(result['device_name'])
        self.assertIn('CPU-only', result['diagnosis'])

    def test_torch_cuda_build_without_access(self):
        torch = NS(__version__='2.5.0+cu126', version=NS(cuda='12.6'), cuda=NS(is_available=lambda: False))
        result = p.probe_torch(self.env({'torch': torch}))
        self.assertIn('built with CUDA', result['diagnosis'])
        self.assertFalse(result['version']['nvidia_build'])

    def test_import_crash(self):
        def broken(name):
            raise RuntimeError('library initialization failed')
        env = Env(root=self.root, importer=broken)
        for probe in (p.probe_torch, p.probe_opencv, p.probe_tensorrt):
            self.assertIn('during import', probe(env)['detail'])

    def test_cuda_manifest(self):
        self.put('/usr/local/cuda/version.json', '{"cuda":{"version":"12.6.11"}}')
        result = p.probe_cuda(self.env())
        self.assertEqual((result['value'], result['line']), ('12.6.11', '12.6'))

    def test_bad_cuda_manifests(self):
        for raw in ('broken', '[]', '{}', '{"cuda":null}', '{"cuda":{"version":42}}'):
            self.put('/usr/local/cuda/version.json', raw)
            self.assertEqual(p.probe_cuda(self.env())['status'], 'unknown')

    def test_opencv_counts(self):
        for count in (0, 1, 2):
            cv2 = NS(__version__='4.8.0', cuda=NS(getCudaEnabledDeviceCount=lambda: count))
            result = p.probe_opencv(self.env({'cv2': cv2}))
            self.assertEqual(result['cuda_devices'], count)
            self.assertEqual(result['cuda_enabled'], count > 0)

    def test_opencv_bad_query(self):
        for count in (-1, None, '1'):
            cv2 = NS(__version__='4.8.0', cuda=NS(getCudaEnabledDeviceCount=lambda: count))
            self.assertEqual(p.probe_opencv(self.env({'cv2': cv2}))['status'], 'unknown')
        self.assertEqual(p.probe_opencv(self.env({'cv2': NS(__version__='4.8.0')}))['status'], 'unknown')

    def test_tensorrt_and_venv_hint(self):
        result = p.probe_tensorrt(self.env({'tensorrt': NS(__version__='10.3.0')}))
        self.assertEqual(result['line'], '10.3')
        self.assertIn('--system-site-packages', p.probe_tensorrt(self.env(venv=True))['detail'])
        self.assertNotIn('--system-site-packages', p.probe_tensorrt(self.env())['detail'])

    def test_l4t_release(self):
        self.put('/etc/nv_tegra_release', '# R36 (release), REVISION: 5.0, GCID: 123\n# KERNEL_VARIANT: oot')
        result = p.probe_l4t(self.env())
        self.assertEqual((result['value'], result['line']), ('36.5.0', '36.5'))

    def test_bad_l4t(self):
        for raw in ('R36 (release)', 'REVISION: 5.0', 'R36 (release)\nREVISION: 5.0', 'R36 (release), REVISION: 5..0,'):
            self.put('/etc/nv_tegra_release', raw)
            self.assertEqual(p.probe_l4t(self.env())['status'], 'unknown')

    def test_sample_schema(self):
        self.put('/usr/local/cuda/version.json', '{"cuda":{"version":"12.6.11"}}')
        self.put('/etc/nv_tegra_release', '# R36 (release), REVISION: 5.0,')
        env = self.env({
            'torch': NS(__version__='2.5.0+cu126', version=NS(cuda='12.6'), cuda=NS(is_available=lambda: True, get_device_name=lambda i: 'Orin')),
            'cv2': NS(__version__='4.8.0', cuda=NS(getCudaEnabledDeviceCount=lambda: 0)),
            'tensorrt': NS(__version__='10.3.0'),
        })
        sample = json.loads(Path(__file__).with_name('sample_system_report.json').read_text())
        def compare(expected, actual):
            self.assertIs(type(actual), type(expected))
            if isinstance(expected, dict):
                self.assertEqual(expected.keys(), actual.keys())
                for key in expected:
                    compare(expected[key], actual[key])
        for name, expected in sample.items():
            compare(expected, getattr(p, name)(env))


if __name__ == '__main__':
    unittest.main()
