# Lab 7: requirements, pre-lab work, and remaining hardware checks

October 21, 2026. Instructor revision: 61d62aa (posting lab 7).

## What needs to be done

- [x] Implement frame_bytes: pixels times format bytes/pixel, rounded up.
- [x] Implement bytes_per_second: frame payload times assumed FPS.
- [x] Implement frame_stats: min, max, mean, spread, population SD, count.
- [x] Implement classify: black first, then flat, then dark, otherwise scene.
- [x] Implement conversion_bytes: source read plus destination write.
- [x] Implement resize_bytes: full source read plus resized output write.
- [x] Implement normalise_bytes: uint8 input read plus float output write.
- [x] Implement summarise_formats: all seven formats against RGB baseline.
- [x] Test independent small examples and instructor sample arithmetic.
- [x] Generate explicitly computed pre-lab predictions.
- [x] Verify the implementation and predictions on the Jetson.
- [ ] Capture a new image from the student's physical camera.
- [ ] Inspect image and measured luma statistics; resolve black/flat/dark issues.
- [ ] Save this run's output/captured_frame.jpg and output/frame_report.json.
- [ ] Review any separate ELMS handout for additional in-class requirements.

## Steps taken

1. Fetched instructor main and read all new lab files and the sample report.
2. Created solution7 from the completed solution6. The instructor's intervening
   Lab 6 change conflicted with our rewrite: retained our completed lab06/quantize.py
   and merged the new Lab 7 files. Previous solutions remain preserved.
3. Saved the original Lab 7 in Git and local work/lab07-original.tar.
4. Wrote the eight stubs with simple comments and input validation. Statistics
   use Welford's streaming population variance, accepting generators without
   another full-frame list. Classification uses normalized [0,1] luma.
5. Made OpenCV/NumPy imports local to hardware capture so the pure functions
   and prelab.py run using only standard Python. Kept the provided GStreamer
   pipeline and warmup procedure. Added camera release on open failure and an
   explicit error if image writing fails. Capture output stays in lab07/output.
6. Moved the instructor's image/report to reference_output, so their sample
   cannot be mistaken for a new student capture. New hardware captures go to output.
7. Added 18 tests: formats, rounding, FPS, conversion, resizing, normalization,
   reference arithmetic, statistics, diagnostic thresholds, invalid data, and
   mocked camera cleanup. Local checks and deployment results are recorded below.
8. Connected to the Jetson: OpenCV 4.8.0 reports GStreamer YES (1.20.3), but the
   initial device check found no /dev/video* nodes. Camera capture is pending.

## Can this be done before class?

All math, diagnostics, tests, and predicted reports can be completed now. The
physical image, scene validation, and camera troubleshooting need a connected
sensor. The existing main assumes 30 FPS; it does not measure delivered FPS,
dropped frames, jitter, throughput, or actual memory traffic.

The README/schema mention comparing pipelines and verdicts, but this starter
contains only a one-frame main and a simpler sample report. schema.validate
expects a different, larger report and is not called by this main. No claim is
made that the generated report meets that unrelated schema. Additional timing
or pipeline-comparison tasks would need confirmation from the ELMS handout.

## Predictions at 1920 x 1080, assumed 30 FPS

| Quantity | Bytes or bytes/second |
|---|---:|
| RAW10 frame (packed wire estimate) | 2,592,000 |
| GRAY8 frame | 2,073,600 |
| NV12 or I420 frame | 3,110,400 |
| RGB or BGR frame | 6,220,800 |
| BGRx frame | 8,294,400 |
| BGR payload/second | 186,624,000 |
| NV12 payload/second | 93,312,000 |
| NV12 to BGR read + write | 9,331,200 |
| RGB resize 1080p to 224 x 224 read + write | 6,371,328 |
| Full 1080p uint8 to FP32 read + write | 31,104,000 |
| After-resize 224 x 224 normalization read + write | 752,640 |

At 30 FPS BGR payload is about 177.98 MiB/s, versus 88.99 MiB/s NV12.
These are packed payload/one-read-one-write estimates, not measured physical
bus traffic; row pitch, headers, allocator overhead, caches, conversion workspaces,
extra copies, and RAW10 line-packing constraints are not modeled. Fractional
byte payloads round up; real YUV420/RAW10 format geometry constraints may require
larger aligned dimensions. Per-stage numbers are independent examples: the
starter normalizes full resolution even though it also prices a resize.

BLACK_MAX is inclusive, FLAT_SPREAD is exclusive, and mean < 0.10 means dark.
Black/flat diagnoses are heuristics, not proof of sensor failure. A real dark
or uniform scene can trigger them. Nonfinite/out-of-range normalized luma is
rejected. SD is population SD, not sample SD.

## Run

On any Python 3.10+ computer, from lab07:

    python3 prelab.py
    python3 -m unittest -v test_frames.py

With OpenCV/GStreamer, NumPy, and the CSI camera on the Jetson:

    python3 frames.py

Read the image visually and output/frame_report.json. Do not expect real scene
statistics or absolute image paths to equal the instructor sample. Only the
same-dimension arithmetic should match. prelab_predictions.json is calculated
and deliberately contains no measured scene or invented hardware statistics.

## Jetson verification outcome

Updated the clean existing checkout from solution6 to solution7 on October 21.
Ran prelab.py and all 18 tests successfully on the board (0.014 seconds).
The reference arithmetic comparison passed; prelab predictions are identical.
Attempted one real capture with a 35-second external timeout safeguard. Argus
reported "No cameras available" and the read failed with "Camera opened, but
failed to capture a valid frame." Device enumeration returned []. No new
image or measured scene report was produced. Did not restart services or change
camera/system configuration. Raw session logs remain on the Jetson at
/tmp/lab07-tests.txt and /tmp/lab07-capture.txt (temporary, may vanish at reboot).

In class, check the sensor connection with staff; camera ribbons should be
handled only with board power off. Once the camera is available, run:

    cd /home/student/ENEE459L/lab07
    python3 frames.py

Inspect output/captured_frame.jpg and output/frame_report.json. Confirm that
the scene is real, neither black nor suspiciously flat, and its width/height
match expectations. If a valid scene is dark, improve lighting and retake.
Use git status, then add only the new output artifacts before committing them.
A different scene naturally produces different stats from the sample.
