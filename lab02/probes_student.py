from __future__ import annotations

import json
import re
from typing import Any

from env import (
    Env,
    ModuleNotAvailable,
    getattr_path,
    major_minor,
    read_text,
    unknown,
)


# These patterns find the NVIDIA build tag and the L4T release numbers.
_NV_LOCAL_TAG = re.compile(r"(?:^|\.)nv\d", re.IGNORECASE)
_L4T_RELEASE = re.compile(r"R(\d+)\s*\(release\)", re.IGNORECASE)
_L4T_REVISION = re.compile(r"REVISION:\s*(\d+(?:\.\d+)*)(?=\s|,|$)")


def _split_local_version(raw: str) -> dict[str, Any]:
    """Separate the public version from the build tag after '+'."""
    if not raw:
        return {"raw": raw, "public": None, "local": None, "nvidia_build": False}

    public, sep, local = raw.partition("+")
    local = local if sep else None

    return {
        "raw": raw,
        "public": public or None,
        "local": local,
        "nvidia_build": bool(local and _NV_LOCAL_TAG.search(local)),
    }


def probe_torch(env: Env) -> dict[str, Any]:
    """Check the PyTorch version and whether it can see the GPU."""

    # 1. Import PyTorch using the supplied environment.
    source = "import torch"

    try:
        torch = env.importer("torch")
    except ModuleNotAvailable as error:
        return unknown(source, f"PyTorch cannot be imported: {error}")
    except Exception as error:
        return unknown(source, f"PyTorch failed during import: {error}")


    # 2. Read the version and separate its build tag.
    raw = getattr_path(torch, "__version__")

    if not isinstance(raw, str) or not raw.strip():
        return unknown(source, "PyTorch did not provide a version")

    version = _split_local_version(raw)
    cuda_version = getattr_path(torch, "version.cuda")


    # 3. Ask PyTorch whether CUDA works and read the device name.
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception as error:
        return unknown(source, f"PyTorch CUDA query failed: {error}")


    # 4. Explain the result without treating a build tag as proof.
    if cuda_available:
        diagnosis = "torch is installed and sees the GPU"
    elif cuda_version is None:
        diagnosis = (
            "torch cannot see the GPU and reports no CUDA build version; "
            "this is consistent with a CPU-only wheel"
        )
    else:
        diagnosis = (
            "torch was built with CUDA but cannot see the GPU; "
            "check the driver, device access, and Python environment"
        )


    # 5. Return the measurements in the sample report format.
    return {
        "value": raw,
        "source": source,
        "status": "ok",
        "version": version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_name": device_name,
        "diagnosis": diagnosis,
    }


def probe_cuda(env: Env) -> dict[str, Any]:
    """Read the CUDA toolkit version from its installed manifest."""

    # 1. Read the manifest under the supplied filesystem root.
    source = "/usr/local/cuda/version.json"
    raw = read_text(env.root, source)

    if raw is None:
        return unknown(source, "CUDA version manifest is missing or unreadable")


    # 2. Parse the JSON and read the CUDA SDK version.
    try:
        manifest = json.loads(raw)
    except (ValueError, TypeError) as error:
        return unknown(source, f"CUDA version manifest is invalid JSON: {error}")

    cuda = manifest.get("cuda") if isinstance(manifest, dict) else None
    version = cuda.get("version") if isinstance(cuda, dict) else None

    if not isinstance(version, str) or major_minor(version) is None:
        return unknown(source, "CUDA manifest has no valid cuda.version string")


    # 3. Keep both the full version and its major-minor release line.
    return {
        "value": version,
        "source": source,
        "status": "ok",
        "line": major_minor(version),
    }


def probe_opencv(env: Env) -> dict[str, Any]:
    """Read the OpenCV version and count its available CUDA devices."""

    # 1. Import OpenCV and read its version.
    source = "import cv2"

    try:
        cv2 = env.importer("cv2")
    except ModuleNotAvailable as error:
        return unknown(source, f"OpenCV cannot be imported: {error}")
    except Exception as error:
        return unknown(source, f"OpenCV failed during import: {error}")

    version = getattr_path(cv2, "__version__")

    if not isinstance(version, str) or not version.strip():
        return unknown(source, "OpenCV did not provide a version")


    # 2. Find and call the CUDA device-count function.
    get_count = getattr_path(cv2, "cuda.getCudaEnabledDeviceCount")

    if not callable(get_count):
        return unknown(source, "OpenCV CUDA device-count function is unavailable")

    try:
        count = get_count()
    except Exception as error:
        return unknown(source, f"OpenCV CUDA query failed: {error}")

    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return unknown(source, f"OpenCV returned an invalid CUDA device count: {count!r}")


    # 3. Describe GPU availability. Zero devices alone does not prove why.
    if count > 0:
        detail = "OpenCV can see CUDA devices"
    else:
        detail = (
            "OpenCV reports no CUDA devices; it may be a CPU-only build "
            "or the GPU may be unavailable"
        )


    # 4. Return the version and the CUDA measurements.
    return {
        "value": version,
        "source": source,
        "status": "ok",
        "cuda_devices": count,
        "cuda_enabled": count > 0,
        "detail": detail,
    }


def probe_tensorrt(env: Env) -> dict[str, Any]:
    """Check whether this Python interpreter can import TensorRT."""

    # 1. Import TensorRT and explain a possible virtual-environment issue.
    source = "import tensorrt"

    try:
        tensorrt = env.importer("tensorrt")
    except ModuleNotAvailable as error:
        detail = f"TensorRT cannot be imported: {error}"

        if env.python.prefix != env.python.base_prefix:
            detail += (
                "; this interpreter is in a virtual environment. "
                "Check whether it was created with --system-site-packages"
            )

        return unknown(source, detail)
    except Exception as error:
        return unknown(source, f"TensorRT failed during import: {error}")


    # 2. Read the full version and check its major-minor release line.
    version = getattr_path(tensorrt, "__version__")

    if not isinstance(version, str) or major_minor(version) is None:
        return unknown(source, "TensorRT did not provide a valid version")


    # 3. Return the installed version.
    return {
        "value": version,
        "source": source,
        "status": "ok",
        "line": major_minor(version),
    }


def probe_l4t(env: Env) -> dict[str, Any]:
    """Read the Linux for Tegra release and revision."""

    # 1. Read the release file under the supplied filesystem root.
    source = "/etc/nv_tegra_release"
    raw = read_text(env.root, source)

    if not raw:
        return unknown(source, "L4T release file is missing or unreadable")


    # 2. Find the release and revision on the same line.
    for line in raw.splitlines():
        release = _L4T_RELEASE.search(line)
        revision = _L4T_REVISION.search(line)

        if release and revision:
            break
    else:
        return unknown(source, "L4T release or revision could not be parsed")


    # 3. Join R36 and revision 5.0 into the version 36.5.0.
    version = f"{release.group(1)}.{revision.group(1)}"


    # 4. Preserve the original text as evidence.
    return {
        "value": version,
        "source": source,
        "status": "ok",
        "line": major_minor(version),
        "raw": raw,
    }


def main() -> None:
    """Run all five probes and save their results."""

    # 1. Use the current interpreter and the real Jetson filesystem.
    env = Env.real()


    # 2. Collect each result separately.
    report = {
        "probe_torch": probe_torch(env),

        "probe_cuda": probe_cuda(env),

        "probe_opencv": probe_opencv(env),

        "probe_tensorrt": probe_tensorrt(env),

        "probe_l4t": probe_l4t(env),
    }


    # 3. Write the JSON report in the current directory.
    with open("system_report.json", "w", encoding="utf-8") as output:
        json.dump(report, output, indent=4)
        output.write("\n")


if __name__ == "__main__":
    main()
