"""Tests for Lab 10 arithmetic, reports, and malformed WAV handling."""
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave

import capture as c


class CaptureTests(unittest.TestCase):
    def test_supplied_report(self):
        root = Path(__file__).parent
        actual = c.run_analysis(root / 'test_voice.wav')
        expected = json.loads((root / 'test_voice_report.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected)

    def test_wav_size_stereo(self):
        self.assertEqual(c.wav_bytes(48000, 2, 24, .5), {
            'samples': 24000, 'block_align': 6, 'byte_rate': 288000,
            'data_bytes': 144000, 'header_bytes': 44, 'total_bytes': 144044})

    def test_rounding_and_custom_header(self):
        self.assertEqual(c.wav_bytes(10, 1, 16, .25, 80)['samples'], 2)
        self.assertEqual(c.wav_bytes(10, 1, 16, 0, 80)['total_bytes'], 80)

    def test_bad_geometry(self):
        for args in [(0, 1, 16, 1), (1, True, 16, 1), (1, 1, 12, 1),
                     (1, 1, 16, -1), (1, 1, 16, float('nan')),
                     (1, 1, 16, 1, -1), (1, 1, 16, True)]:
            with self.subTest(args=args), self.assertRaises(c.CaptureError):
                c.wav_bytes(*args)

    def test_partial_frame(self):
        self.assertEqual(c.duration_from_bytes(13, 2, 2, 24),
                         {'samples': 2, 'seconds': 1, 'remainder': 1, 'whole': False})
        self.assertTrue(c.duration_from_bytes(0, 44100, 1, 16)['whole'])

    def test_bad_payload(self):
        for size in [-1, True, 2.5]:
            with self.subTest(size=size), self.assertRaises(c.CaptureError):
                c.duration_from_bytes(size, 44100, 1, 16)

    def test_header_mismatch(self):
        header = dict(sample_rate=44100, channels=1, bit_depth=16, data_bytes=88200)
        audio = dict(sample_rate=16000, channels=1, bit_depth=16, seconds=2)
        result = c.header_consistency(header, audio)
        self.assertEqual(result['mismatches'], [{'field': 'sample_rate', 'header': 44100, 'audio': 16000}])
        self.assertFalse(result['agree'])
        self.assertEqual(result['duration']['seconds'], 1)
        self.assertEqual(result['stated_seconds'], 2)

    def test_missing_header_field(self):
        with self.assertRaises(c.CaptureError):
            c.header_consistency({}, {})

    def test_dbfs(self):
        self.assertIsNone(c.dbfs(0))
        self.assertEqual(c.dbfs(1), 0)
        self.assertAlmostEqual(c.dbfs(.5), -6.020599913279624)
        self.assertAlmostEqual(c.amplitude_from_dbfs(c.dbfs(.125)), .125)

    def test_bad_amplitudes(self):
        for amplitude in [-.01, True, '1', float('nan'), float('inf')]:
            with self.subTest(amplitude=amplitude), self.assertRaises(c.CaptureError):
                c.dbfs(amplitude)

    def test_bits(self):
        self.assertEqual(c.bits_in_use(None, 16), 0)
        self.assertEqual(c.bits_in_use(0, 16), 16)
        self.assertAlmostEqual(c.bits_in_use(-c.DB_PER_BIT * 4, 16), 12)
        self.assertEqual(c.bits_in_use(-200, 16), 0)

    def test_bad_bits(self):
        for level, bits in [(True, 16), (float('nan'), 16), (None, 0), (False, 16)]:
            with self.subTest(level=level), self.assertRaises(c.CaptureError):
                c.bits_in_use(level, bits)

    def test_silence_and_clipping(self):
        silent = c.level_report(0, 0, 0, 100)
        self.assertEqual(c.level_sentence(silent), 'the capture is digitally silent: all 100 samples are zero')
        self.assertFalse(silent['usable'])
        hot = c.level_report(1, .5, 2, 100)
        self.assertTrue(hot['too_hot'])
        self.assertIn('2 samples at the rail (2.0000%)', c.level_sentence(hot))
        json.dumps(silent, allow_nan=False)

    def test_crest_undefined(self):
        self.assertEqual(c.level_sentence(c.level_report(.5, 0, 0, 10)), 'peak -6.02 dBFS')

    def test_level_boundaries(self):
        self.assertFalse(c.level_report(.01, .001, 0, 1000)['too_quiet'])
        self.assertTrue(c.level_report(.001, .0001, 0, 1000)['too_quiet'])
        for args in [(.1, .2, 0, 10), (.1, .01, 11, 10)]:
            with self.assertRaises(c.CaptureError):
                c.level_report(*args)

    def test_record_empty(self):
        result = c.record_gaps({})
        self.assertEqual(len(result['missing']), 13)
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['hand_written']), 10)

    def test_blank_and_zero_metadata(self):
        result = c.record_gaps({'device_name': '  ', 'capture_gain': 0,
                                'operator_notes': '', 'filter_highpass_hz': False})
        self.assertEqual(result['blank'], ['device_name', 'operator_notes'])
        self.assertIn('capture_gain', result['present'])
        self.assertIn('filter_highpass_hz', result['present'])
        self.assertIn('2 present and empty (device_name, operator_notes)', c.record_sentence(result))

    def test_complete_metadata(self):
        result = c.record_gaps(dict.fromkeys(c.THIRTEEN_CHOICES, 'recorded'))
        self.assertTrue(result['complete'])
        self.assertIn('all 13 choices are named', c.record_sentence(result))
        self.assertIn('3 of them also in the file header and 10 of them nowhere else', c.record_sentence(result))

    def test_handout_scalar_and_record_rules(self):
        self.assertAlmostEqual(c.dbfs(2), 20 * math.log10(2))
        self.assertGreater(c.bits_in_use(6, 16), 16)
        self.assertEqual(c.record_gaps({'device_name': None})['present'], ['device_name'])
        self.assertEqual(c.record_gaps({'device_name': []})['present'], ['device_name'])
        header = dict(sample_rate=8000, channels=1, bit_depth=16, data_bytes=16)
        self.assertEqual(c.header_consistency(header, header)['stated_seconds'], 0.0)

    def make_wav(self, folder, width=2, channels=1, values=None):
        path = Path(folder) / 'audio.wav'
        values = values if values is not None else [0, 100, -100, 200]
        body = b''.join(value.to_bytes(width, 'little', signed=True) for value in values)
        with wave.open(str(path), 'wb') as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(8000)
            out.writeframes(body)
        return path

    def test_stereo_24_bit(self):
        with tempfile.TemporaryDirectory() as folder:
            result = c.run_analysis(self.make_wav(folder, 3, 2, [0, 8388607, -8388608, 0]))
            self.assertEqual(result['level_report']['samples'], 4)
            self.assertEqual(result['level_report']['clipped'], 2)
            self.assertEqual(result['wav_structure']['samples'], 2)

    def test_truncated_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_wav(folder)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(c.CaptureError):
                c.run_analysis(path)

    def test_wrong_stride(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_wav(folder)
            raw = bytearray(path.read_bytes())
            struct.pack_into('<H', raw, 32, 4)
            path.write_bytes(raw)
            with self.assertRaises(c.CaptureError):
                c.run_analysis(path)

    def test_non_pcm(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_wav(folder)
            raw = bytearray(path.read_bytes())
            struct.pack_into('<H', raw, 20, 3)
            path.write_bytes(raw)
            with self.assertRaises(c.CaptureError):
                c.run_analysis(path)

    def test_padded_unknown_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_wav(folder)
            raw = bytearray(path.read_bytes())
            raw[12:12] = b'JUNK' + struct.pack('<I', 1) + b'x\x00'
            struct.pack_into('<I', raw, 4, len(raw) - 8)
            path.write_bytes(raw)
            header, data = c._parse_wav(path)
            self.assertEqual(len(data), 8)
            self.assertEqual(header['sample_rate'], 8000)


if __name__ == '__main__':
    unittest.main()
