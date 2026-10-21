# CSI camera fix - October 21, 2026

## Diagnosis

The board is an NVIDIA Jetson Orin Nano Super developer kit, with L4T R36.5.
Before configuration: no /dev/video nodes, no IMX219 nodes in the running device
tree, and no sensor-specific driver registered. Camera software and IMX219/477
modules and overlays were already installed. Direct GStreamer also reported
no cameras available; the synthetic pipeline worked.
The student read the camera board marking as J8IM228-V3.0. Identification as
IMX219 was initially tentative; successful hardware probing after configuration
and real frames now establish a working IMX219-compatible configuration.

## Changes and recovery

Backed up /boot/extlinux/extlinux.conf to:

    /home/student/ENEE459L-backups/camera-config-20261021/extlinux.conf.before

Validated the IMX219 Dual overlay against a temporary copy of the active device
tree. At the user's request, used NVIDIA Jetson-IO's command-line interface:

    sudo python3 /opt/nvidia/jetson-io/config-by-hardware.py -n '2=Camera IMX219 Dual'

This configures header 2 (Jetson 22pin CSI Connector), adds a JetsonIO boot entry,
and sets it as the default. It references the installed
/boot/tegra234-p3767-camera-p3768-imx219-dual.dtbo overlay. The original primary
boot entry remains. Rebooted the Jetson once to activate the profile.
The backup can restore the previous configuration if a future rollback is
needed; this is not required for normal operation.

## Verification

- /dev/video0 appeared after reboot.
- IMX219 entries appeared in the running device tree.
- Direct GStreamer received stream data; its teardown printed Argus CANCELLED
  messages and returned 1, so that first transport attempt was not counted as
  a clean final check.
- Then used take_selfie.gstreamer_pipeline() with cv2.VideoCapture and read
  20 consecutive valid frames. Camera opened successfully; every read succeeded.
- Last frame shape: (540, 960, 3). Mean pixel value: 84.8697299382716 on 0-255.
- OpenCV cleaned up successfully. No image was saved in these feed tests.

The raw session log is /tmp/lab07-feed-check.txt on the Jetson (temporary).
A live stream check does not change the provenance of the submitted chessboard
image, which remains TA-approved supplied data. Any new photo should be prompted
and explicitly arranged before capture, as requested by the student.

## Capture when ready

    cd /home/student/ENEE459L/lab07
    python3 take_selfie.py

This saves test_photo.jpg at 960 x 540. To capture at the full-resolution pipeline
and generate a fresh analysis report, use frames.py, taking care to choose a
separate output directory so the approved submission is not overwritten.

NVIDIA documentation:
https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/HR/ConfiguringTheJetsonExpansionHeaders.html
