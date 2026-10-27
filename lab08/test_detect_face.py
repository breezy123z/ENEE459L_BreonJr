import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from PIL import Image
from detect_face import hard_nms, non_maximum_suppression, hysteresis_thresholding, canny_scipy


class VisionTests(unittest.TestCase):
    def test_nms_duplicate_and_disjoint(self):
        boxes=np.array([[0,0,10,10,.8],[0,0,10,10,.9],[20,20,30,30,.7]],float)
        original=boxes.copy()
        out=hard_nms(boxes,.3)
        np.testing.assert_array_equal(out,boxes[[1,2]])
        np.testing.assert_array_equal(boxes,original)

    def test_nms_exact_iou_boundary(self):
        boxes=np.array([[0,0,2,2,.9],[0,0,1,2,.8]],float)
        self.assertEqual(len(hard_nms(boxes,.5)),2)
        self.assertEqual(len(hard_nms(boxes,.49)),1)

    def test_nms_topk_and_candidates(self):
        boxes=np.array([[i*10,0,i*10+1,1,i/10] for i in range(5)],float)
        self.assertEqual(len(hard_nms(boxes,.3,top_k=1)),1)
        out=hard_nms(boxes,.3,candidate_size=2)
        np.testing.assert_array_equal(out[:,4],[.4,.3])

    def test_nms_empty_and_degenerate(self):
        self.assertEqual(hard_nms(np.empty((0,5)),.3).shape,(0,5))
        out=hard_nms(np.array([[0,0,0,0,.9],[0,0,0,0,.8]]),.3)
        self.assertEqual(len(out),2)

    def test_nms_bad_inputs(self):
        for data,threshold in [(np.zeros((2,4)),.3),(np.zeros((2,5)),-1)]:
            with self.assertRaises(ValueError): hard_nms(data,threshold)

    def test_directional_sectors(self):
        for degrees,pair in [(0,((2,1),(2,3))),(45,((3,1),(1,3))),
                             (90,((1,2),(3,2))),(135,((1,1),(3,3)))]:
            with self.subTest(degrees=degrees):
                magnitude=np.zeros((5,5)); magnitude[2,2]=5
                angle=np.full((5,5),np.deg2rad(degrees))
                self.assertEqual(non_maximum_suppression(magnitude,angle)[2,2],5)
                magnitude[pair[0]]=6
                self.assertEqual(non_maximum_suppression(magnitude,angle)[2,2],0)

    def test_angle_wrap_and_ties(self):
        m=np.zeros((5,5)); m[2,1:4]=5
        for deg in (-180,0,180,360):
            self.assertEqual(non_maximum_suppression(m,np.full(m.shape,np.deg2rad(deg)))[2,2],5)

    def test_edges_border_zero(self):
        out=non_maximum_suppression(np.ones((4,4)),np.zeros((4,4)))
        self.assertTrue(np.all(out[[0,-1],:]==0)); self.assertTrue(np.all(out[:,[0,-1]]==0))

    def test_hysteresis_transitive_reverse_chain(self):
        img=np.zeros((5,7)); img[2,1:5]=40; img[2,5]=100
        out=hysteresis_thresholding(img,20,60)
        np.testing.assert_array_equal(out[2,1:6],np.full(5,255))
        self.assertEqual(set(np.unique(out)),{0,255})

    def test_hysteresis_diagonal_and_isolated(self):
        img=np.zeros((6,6)); img[0,0]=100; img[1,1]=40; img[2,2]=40; img[5,5]=40
        out=hysteresis_thresholding(img,20,60)
        self.assertEqual(out[2,2],255); self.assertEqual(out[5,5],0)

    def test_threshold_inclusive(self):
        out=hysteresis_thresholding(np.array([[60,20,19]],float),20,60)
        np.testing.assert_array_equal(out,[[255,255,0]])

    def test_blank_canny_no_division_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'blank.png'; dest=Path(tmp)/'edges.png'
            Image.new('L',(16,16),128).save(src)
            out=canny_scipy(src,dest)
            self.assertEqual(out.dtype,np.uint8); self.assertEqual(np.count_nonzero(out),0)
            self.assertTrue(dest.exists())

    def test_canny_step_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=np.zeros((32,32),dtype=np.uint8); data[:,16:]=255
            src=Path(tmp)/'step.png'; Image.fromarray(data).save(src)
            edges=canny_scipy(src,Path(tmp)/'edges.png')
            self.assertGreater(np.count_nonzero(edges),0)
            self.assertEqual(np.count_nonzero(edges[:,:10]),0)
            self.assertEqual(np.count_nonzero(edges[:,22:]),0)


if __name__=='__main__': unittest.main(verbosity=2)
