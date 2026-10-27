# Lab 8 work log

October 27, 2026. Handout dated October 28; instructor revision d450ffe.

1. Read the five-page handout and README and inspected the five stubs.
2. Created solution8 from the completed solution7 and merged only the new
   instructor update. Original starter remains in Git and work/lab08-original.tar.
3. Implemented hard_nms, detect_faces, non_maximum_suppression,
   hysteresis_thresholding, and canny_scipy without OpenCV dependencies.
4. Hard NMS uses ascending stable confidence sorting, highest-first picks,
   top_k/candidate limits, vectorized IoU, and strict > threshold suppression.
   Empty input and zero-area union are handled without NaNs.
5. Face inference uses RGB Pillow bilinear resize to 320 x 240, float32
   normalization (x-127)/128, contiguous NCHW layout, face-class confidence
   filtering, IoU 0.3, and pixel-coordinate annotations on the original image.
6. Canny uses float32 grayscale, Gaussian sigma 1.4, the exact Sobel kernels
   from the handout, 0-255 magnitude scaling, four-direction thinning, and
   dual thresholds 20/60. Flat input avoids division by zero.
7. Hysteresis uses full 8-connected propagation from strong pixels, rather
   than a single raster pass. This follows the task's connectivity goal and
   preserves weak chains regardless of traversal order. Final map is 0/255.
8. Added 13 independent tests covering NMS, all angle sectors, edge borders,
   inclusive thresholds, reverse-order weak chains, diagonal connectivity,
   blank images, and a synthetic step edge.
9. The input and sample images are instructor-provided (1920 x 1440). No
   camera capture is needed and no new personal photograph was taken.
10. Jetson already has NumPy, SciPy and Pillow. ONNX Runtime was missing;
    setup attempted in /home/student/lab08-env with system-site-packages,
    keeping the previous CUDA PyTorch installation separate.

## Run

    cd /home/student/ENEE459L/lab08
    /home/student/lab08-env/bin/python detect_face.py
    /home/student/lab08-env/bin/python -m unittest -v test_detect_face.py

Outputs: output_detected.png, edges_scipy.png, results.json. Session providers
are recorded from ONNX Runtime; a CPU run must not be described as CUDA.
Face localization does not identify people. Canny output quality depends on
thresholds, resize/smoothing conventions, and connectivity; comparisons with
the supplied samples are reported after execution, not assumed identical.
