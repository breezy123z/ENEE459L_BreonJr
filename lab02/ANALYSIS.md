# Lab 2 findings

Measured on the Jetson on September 16, 2026 using `/usr/bin/python3`.
The evidence is in `system_report.json`.

1. **PyTorch:** This interpreter cannot import `torch` because the module is
   missing from its import path. The probe reports `unknown`; it cannot
   determine a PyTorch version or GPU availability. This does not prove that
   PyTorch is absent from every possible Python environment on the board.

2. **CUDA:** The toolkit manifest reports **12.6.11**, release line **12.6**.
   Reading this manifest establishes the toolkit version; it does not by
   itself establish that PyTorch can use CUDA.

3. **OpenCV:** Version **4.8.0** imports successfully and reports **0 CUDA
   devices**. That count does not alone distinguish a CPU-only OpenCV build
   from an unavailable GPU.

4. **TensorRT:** Version **10.3.0**, release line **10.3**, imports successfully
   in the system Python interpreter.

5. **L4T:** The release file contains R36 and revision 5.0. The combined version
   is **36.5.0**, release line **36.5**.

## Run the assignment

```bash
cd ~/ENEE459L/lab02
python3 probes.py
cat system_report.json
```

The five implementations are in `probes_student.py`. `probes.py` is a small
launcher matching the command in the instructor's README. Use the Python
interpreter from the course environment if the instructor supplies one.

## Tests

```bash
python3 -m unittest -v test_probes.py
```

All 13 tests pass. They check valid and missing evidence, invalid file data,
CPU-only PyTorch, CUDA builds without GPU access, NVIDIA build tags, OpenCV
device counts, TensorRT environment hints, and L4T parsing. Simulated successful
results match the sample's keys and value types. The live PyTorch result uses
the assignment's explicit unknown format instead of inventing sample values.

The README references ELMS slides, which were not available in the repository.
The sample JSON is a format example, not an authoritative course version lock.
Confirm required versions and the intended Python environment with the instructor.

## Original backup

Untouched upstream files are stored in:
`/home/student/ENEE459L-backups/lab02-original-8383895.tar`

They also remain available in upstream Git commit `8383895`.
