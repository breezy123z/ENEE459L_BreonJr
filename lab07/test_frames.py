import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import sys
from frames import (FrameError, frame_bytes, bytes_per_second, frame_stats,
                    classify, conversion_bytes, resize_bytes, normalise_bytes,
                    summarise_formats, gstreamer_pipeline, grab_frame)
from schema import BLACK_MAX, FLAT_SPREAD
from prelab import predict


class FrameTests(unittest.TestCase):
    def test_payload_by_hand(self):
        expected={'raw10':10,'gray8':8,'nv12':12,'i420':12,'rgb':24,'bgr':24,'bgrx':32}
        for fmt,n in expected.items():
            self.assertEqual(frame_bytes(4,2,fmt),n)
        self.assertEqual(frame_bytes(4,2,'RGB'),24)

    def test_fractional_byte_roundup(self):
        self.assertEqual(frame_bytes(1,1,'raw10'),2)
        self.assertEqual(frame_bytes(1,1,'nv12'),2)

    def test_bandwidth(self):
        self.assertEqual(bytes_per_second(4,2,'bgr',2.5),60.)

    def test_conversion_read_write(self):
        self.assertEqual(conversion_bytes(4,2,'nv12','bgr'),36)
        self.assertEqual(conversion_bytes(4,2,'bgr','bgr'),0)
        self.assertEqual(conversion_bytes(4,2,'BGR','bgr'),0)

    def test_resize_and_normalize(self):
        self.assertEqual(resize_bytes(4,2,2,2,'rgb'),36)
        self.assertEqual(normalise_bytes(4,2),120)
        self.assertEqual(normalise_bytes(4,2,1,2),24)

    def test_reference_arithmetic(self):
        root=Path(__file__).parent
        ref=json.loads((root/'reference_output/frame_report.json').read_text())['memory_pricing']
        calculated=predict()
        for key in ('format_comparison','bandwidth_bytes_per_sec','pipeline_stage_costs'):
            self.assertEqual(calculated[key],ref[key])

    def test_stats_generator(self):
        stats=frame_stats(v for v in [0.,.25,.5,.75,1.])
        self.assertEqual(stats['samples'],5.)
        self.assertEqual(stats['min'],0.)
        self.assertEqual(stats['max'],1.)
        self.assertEqual(stats['spread'],1.)
        self.assertAlmostEqual(stats['mean'],.5)
        self.assertAlmostEqual(stats['sd'],math.sqrt(.125))

    def test_single_and_constant(self):
        for pixels in ([.5],[.5]*100):
            stats=frame_stats(pixels)
            self.assertEqual(stats['sd'],0.)
            self.assertEqual(stats['spread'],0.)

    def test_black_priority(self):
        self.assertEqual(classify(frame_stats([0.,BLACK_MAX]))[0],'black')

    def test_flat_not_black(self):
        self.assertEqual(classify(frame_stats([.5]*10))[0],'flat')

    def test_dark_structured(self):
        self.assertEqual(classify(frame_stats([0.,.15]))[0],'dark')

    def test_scene(self):
        label,reason=classify(frame_stats([0.,.2,.8]))
        self.assertEqual(label,'scene')
        self.assertIn('Exposed and structured',reason)

    def test_thresholds(self):
        stats={'max':.5,'mean':.1,'spread':FLAT_SPREAD}
        self.assertEqual(classify(stats)[0],'scene')

    def test_classification_missing_metrics(self):
        for key in ('max', 'spread', 'mean'):
            stats={'max':1., 'spread':1., 'mean':.5}
            del stats[key]
            with self.subTest(key=key):
                with self.assertRaises(FrameError): classify(stats)

    def test_bad_inputs(self):
        for dims in ((0,2),(-1,2),(2.5,2),(True,2)):
            with self.assertRaises(FrameError): frame_bytes(*dims,'rgb')
        with self.assertRaises(FrameError): frame_bytes(2,2,'bad')
        for rate in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(FrameError): bytes_per_second(2,2,'rgb',rate)
        for pixels in ([],[float('nan')],[-.1],[1.1]):
            with self.assertRaises(FrameError): frame_stats(pixels)
        with self.assertRaises(FrameError): normalise_bytes(2,2,channels=0)

    def test_pipeline_headless(self):
        text=gstreamer_pipeline(display_width=224,display_height=224)
        for token in ('nvarguscamerasrc','NV12','nvvidconv','BGRx','BGR','appsink'):
            self.assertIn(token,text)
        self.assertNotIn('imshow',text)
        self.assertIn('width=(int)224',text)

    def test_capture_releases_on_success(self):
        camera=MagicMock(); camera.isOpened.return_value=True
        image=object(); camera.read.return_value=(True,image)
        fake=MagicMock(); fake.VideoCapture.return_value=camera
        with patch.dict(sys.modules,{'cv2':fake}):
            self.assertIs(grab_frame('pipeline',warmup_frames=2),image)
        self.assertEqual(camera.read.call_count,3)
        camera.release.assert_called_once()

    def test_unopened_camera_released(self):
        camera=MagicMock(); camera.isOpened.return_value=False
        fake=MagicMock(); fake.VideoCapture.return_value=camera
        with patch.dict(sys.modules,{'cv2':fake}):
            with self.assertRaises(RuntimeError): grab_frame('pipeline')
        camera.release.assert_called_once()

    def test_capture_releases_on_failure(self):
        camera=MagicMock(); camera.isOpened.return_value=True
        camera.read.return_value=(False,None)
        fake=MagicMock(); fake.VideoCapture.return_value=camera
        with patch.dict(sys.modules,{'cv2':fake}):
            with self.assertRaises(RuntimeError): grab_frame('pipeline',warmup_frames=0)
        camera.release.assert_called_once()


if __name__=='__main__':
    unittest.main(verbosity=2)
