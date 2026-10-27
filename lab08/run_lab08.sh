#!/bin/sh
# Keep optional ONNX Runtime packages separate from system/PyTorch packages.
set -eu
cd "$(dirname "$0")"
export PYTHONPATH="$HOME/lab08-deps${PYTHONPATH:+:$PYTHONPATH}"
if [ "$#" -eq 0 ]; then
    exec python3 detect_face.py
else
    exec python3 "$@"
fi
