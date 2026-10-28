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
7. Hysteresis follows the handout's single row-major pass over weak pixels,
   inspecting an 8-connected neighborhood, promoting if a strong neighbor is
   present and suppressing otherwise. This exactly reproduces the sample edge
   image. The method can depend on scan order; it is not full graph tracking.
8. Added 15 independent tests covering NMS, all angle sectors, edge borders,
   inclusive thresholds, reverse-order weak chains, diagonal connectivity,
   blank images, and a synthetic step edge.
9. The input and sample images are instructor-provided (1920 x 1440). No
   camera capture is needed and no new personal photograph was taken.
10. Jetson already has NumPy, SciPy and Pillow. ONNX Runtime was missing;
    venv creation failed because ensurepip is absent. Instead installed ONNX
    Runtime and missing lightweight dependencies with --no-deps into
    /home/student/lab08-deps, leaving system NumPy/SciPy/PyTorch unchanged.
    run_lab08.sh adds this isolated folder to PYTHONPATH.

## Run

    cd /home/student/ENEE459L/lab08
    sh run_lab08.sh
    sh run_lab08.sh -m unittest -v test_detect_face.py

Outputs: output_detected.png, edges_scipy.png, results.json. Session providers
are recorded from ONNX Runtime; a CPU run must not be described as CUDA.
Face localization does not identify people. Canny output quality depends on
thresholds, resize/smoothing conventions, and connectivity; comparisons with
the supplied samples are reported after execution, not assumed identical.

## Hysteresis comparison resolved

Initially tested full 8-connected propagation, which produced 24,010 edge
pixels. It retained all 16,508 reference edges plus 7,502 additional weak-chain
pixels. Tested the handout's literal single-pass procedure: it produced 16,508
edge pixels and matched sample_edges_scipy.png pixel-for-pixel. The final default
therefore uses that procedure. The order-dependent weak-chain test documents
its limitation instead of claiming complete transitive connectivity.

## Final Jetson verification - October 28, 2026

All 15 tests passed on the Jetson in 0.054 seconds. Executed the complete
pipeline with the supplied face.jpg and ONNX model. Output image dimensions
remain 1920 x 1440. One face was detected at confidence 0.9998503923416138,
box [829, 224, 1140, 752]. Session provider: CPUExecutionProvider.
The optional ONNX package is the CPU wheel; no CUDA acceleration is claimed.

Final Canny output has 16,508 edge pixels and is pixel-for-pixel identical to
sample_edges_scipy.png (edge IoU/F1 and pixel agreement all 1.0).
The predicted face box has IoU approximately 0.993 against the green outline
inferred from the instructor sample. Annotation colors/text differ. The sample
is an example, not labeled ground truth, so neither comparison is a dataset
accuracy claim. Both generated images were inspected visually.

Dependency setup: system NumPy remains 1.21.5; the isolated lab08-deps contains
NumPy 1.24.4 to satisfy ONNX Runtime while remaining compatible with system
SciPy 1.8.0. Pillow 9.0.1 is supported through a resize-enum fallback. ONNX
Runtime 1.23.2 and its missing lightweight dependencies are separate from
system site-packages. Existing CUDA-enabled PyTorch was not changed.

The first venv attempt failed due to missing ensurepip; no venv is used.
The first inference tests exposed the old Pillow API and were repaired before
final execution. The earlier full-connectivity edge experiment is not the
submitted algorithm. requirements-jetson-extra.txt reproduces the successful
extra dependency folder. All final images, reports, and test logs are included.
