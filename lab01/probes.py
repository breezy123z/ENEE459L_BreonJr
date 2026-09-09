"""Read Jetson platform evidence, with injectable filesystems and CLI output."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def read_text(root: Path, rel: str) -> str | None:
    """Read a rooted file; missing or unreadable data is a normal outcome."""
    try:
        return (Path(root) / rel.lstrip("/")).read_text(
            errors="replace"
        ).strip("\x00 \t\r\n")
    # Some Jetson sysfs sensors return no data to the buffered reader.
    except (OSError, UnicodeDecodeError, TypeError):
        return None


def run(cmd: list[str]) -> str | None:
    """Return successful command output, with a bounded wait."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def unknown(source: str, why: str) -> dict[str, Any]:
    """Keep the provenance and reason when evidence is unavailable."""
    return {"value": None, "source": source, "status": "unknown", "detail": why}


_SPEED_RE = re.compile(r"Speed\s+([\d.]+)GT/s")
_WIDTH_RE = re.compile(r"Width\s+x(\d+)")
_GEN_BY_GTS = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}


def _parse_link_line(line: str) -> dict[str, Any]:
    speed = _SPEED_RE.search(line)
    width = _WIDTH_RE.search(line)
    try:
        gts = float(speed.group(1)) if speed else None
    except ValueError:
        gts = None
    return {
        "raw": line.strip(),
        "gts": gts,
        "width": int(width.group(1)) if width else None,
        "gen": _GEN_BY_GTS.get(gts),
    }


def generate_interpretation_string(negotiated, capability):
    """Describe measured speed and width without assuming a carrier limit."""
    def label(link):
        speed = (f"Gen{link['gen']}" if link["gen"] is not None
                 else f"{link['gts']:g} GT/s")
        return f"{speed} x{link['width']}"

    if (negotiated["gts"], negotiated["width"]) == (
        capability["gts"], capability["width"]
    ):
        return f"link running at its full capability, {label(negotiated)}"
    return (
        f"drive capable of {label(capability)}, link running at "
        f"{label(negotiated)}; speed or width differs from device capability"
    )


def probe_module_model(root: Path = Path("/")) -> dict[str, Any]:
    """Read the bootloader-provided board identity."""
    src = "/proc/device-tree/model"
    raw = read_text(root, src)
    if not raw:
        return unknown(src, "device tree model absent or unreadable")
    return {"value": raw, "source": src, "status": "ok"}


def probe_memory_total_kb(root: Path = Path("/")) -> dict[str, Any]:
    """Report kernel-visible RAM in kB without rounding to the board label."""
    src = "/proc/meminfo"
    raw = read_text(root, src)
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB\s*$", raw or "", re.M)
    if match is None or int(match.group(1)) <= 0:
        return unknown(src, "MemTotal missing, unreadable, or invalid")
    return {"value": int(match.group(1)), "source": src, "status": "ok"}


def probe_root_source(root: Path = Path("/")) -> dict[str, Any]:
    """Identify the mounted root device independently of fitted storage."""
    src = "/proc/mounts"
    raw = read_text(root, src)
    if raw is None:
        return unknown(src, "mount table missing or unreadable")
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "/":
            device = re.sub(
                r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[0]
            )
            kind = ("nvme" if re.fullmatch(r"/dev/nvme\d+n\d+(p\d+)?", device)
                    else "sd" if re.fullmatch(r"/dev/mmcblk\d+(p\d+)?", device)
                    else "other")
            return {"value": device, "kind": kind, "source": src, "status": "ok"}
    return unknown(src, "no root mount entry found in mount table")


def probe_nvme_present(root: Path = Path("/")) -> dict[str, Any]:
    """Distinguish an absent drive from an unreadable storage inventory."""
    src = "/sys/block/nvme0n1"
    try:
        devices = list((Path(root) / "sys/block").iterdir())
    except OSError:
        return unknown(src, "block device inventory missing or unreadable")
    present = any(device.name == "nvme0n1" for device in devices)
    if not present:
        return {"value": False, "model": None, "source": src, "status": "ok"}
    model = read_text(root, src + "/device/model")
    if not model:
        result = unknown(src, "NVMe is fitted, but its model is unreadable")
        result["model"] = None
        result["present"] = True
        return result
    return {"value": True, "model": model, "source": src, "status": "ok"}


def probe_pcie_link(
    root: Path = Path("/"), lspci_output: str | None = None
) -> dict[str, Any]:
    """Read capability and negotiated status from the same NVMe endpoint."""
    src = "lspci -vv"
    if lspci_output is None:
        if Path(root).resolve() != Path("/"):
            return unknown(src, "inject lspci_output when using a fake filesystem")
        lspci_output = run(["lspci", "-vv"])
    if not lspci_output:
        return unknown(src, "lspci unavailable, failed, or returned no output")

    # Full lspci output includes unrelated devices; never combine their links.
    blocks = re.split(r"\n\s*\n", lspci_output.strip())
    nvme = [block for block in blocks
            if re.search(r"Non-Volatile memory|NVM Express|NVMe", block, re.I)]
    if nvme:
        block = nvme[0]
    elif re.search(r"(?m)^\S*\d{2}:\d{2}\.\d\s", lspci_output):
        return unknown(src, "no NVMe endpoint found in PCI inventory")
    else:
        block = lspci_output  # Tests may supply only the two link lines.

    links = {}
    for key, marker in (("capability", "LnkCap:"), ("negotiated", "LnkSta:")):
        line = next((line for line in block.splitlines()
                     if line.strip().startswith(marker)), None)
        if line is None:
            return unknown(src, f"{marker} missing; elevated access may be needed")
        links[key] = _parse_link_line(line)
        if links[key]["gts"] is None or not links[key]["width"]:
            return unknown(src, f"{marker} speed or width unreadable")
    negotiated, capability = links["negotiated"], links["capability"]
    return {
        "value": negotiated["raw"],
        "negotiated": negotiated,
        "capability": capability,
        "interpretation": generate_interpretation_string(negotiated, capability),
        "source": src,
        "status": "ok",
    }


def probe_thermal_zones(root: Path = Path("/")) -> dict[str, Any]:
    """Convert readable thermal zones from millidegrees C and report the peak."""
    src = "/sys/class/thermal/thermal_zone*/temp"
    zones = []
    for path in sorted((Path(root) / "sys/class/thermal").glob("thermal_zone*")):
        rel = "/sys/class/thermal/" + path.name
        raw = read_text(root, rel + "/temp")
        try:
            temp_c = int(raw) / 1000
        except (TypeError, ValueError):
            continue
        zones.append({
            "zone": path.name,
            "type": read_text(root, rel + "/type"),
            "temp_c": temp_c,
        })
    if not zones:
        return unknown(src, "no readable thermal zone temperatures")
    return {
        "value": max(zone["temp_c"] for zone in zones),
        "zones": zones, "source": src, "status": "ok",
    }


def probe_power_mode(
    root: Path = Path("/"), nvpmodel_output: str | None = None
) -> dict[str, Any]:
    """Query the current mode without changing the board's power configuration."""
    src = "nvpmodel -q"
    if nvpmodel_output is None:
        if Path(root).resolve() != Path("/"):
            return unknown(src, "inject nvpmodel_output with a fake filesystem")
        nvpmodel_output = run(["nvpmodel", "-q"])
    match = re.search(
        r"^NV Power Mode:\s*([^\r\n]+)\r?\n\s*(\d+)\s*$",
        nvpmodel_output or "", re.M,
    )
    if match is None:
        return unknown(src, "power mode name or ID missing or unreadable")
    return {
        "value": match.group(1).strip(), "mode_id": int(match.group(2)),
        "source": src, "status": "ok",
    }


if __name__ == "__main__":
    report = {
        "module_model": probe_module_model(),
        "memory_total_kb": probe_memory_total_kb(),
        "root_source": probe_root_source(),
        "nvme_present": probe_nvme_present(),
        "pcie_link": probe_pcie_link(),
        "thermal_zones": probe_thermal_zones(),
        "power_mode": probe_power_mode(),
    }
    with open("system_report.json", "w", encoding="utf-8") as output:
        json.dump(report, output, indent=4)
        output.write("\n")
