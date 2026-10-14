"""Run the instructor's quantization experiment from any directory."""
from pathlib import Path
import os
import runpy

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    previous = Path.cwd()
    try:
        os.chdir(root)
        runpy.run_path(str(root / "quantize.py"), run_name="__main__")
    finally:
        os.chdir(previous)
