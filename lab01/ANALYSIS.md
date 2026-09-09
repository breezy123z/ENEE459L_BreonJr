# Lab 01 analysis

Evidence: `system_report.json`, collected on this Jetson with
`sudo python3 probes.py` on September 9, 2026. Values are observations,
not constants copied from the sample report.

1. **the_module_exists:** Yes. The device tree identifies an NVIDIA Jetson
   Orin Nano Engineering Reference Developer Kit Super.
2. **the_memory_matches_the_label:** Yes under the lab's >= 6 GB check.
   Linux reports 7,789,952 kB (about 7.43 GiB). Kernel-visible memory is
   smaller than the nominal capacity because hardware reserves memory.
3. **boots_from_nvme:** Yes. `/proc/mounts` places `/` on `/dev/nvme0n1p1`.
   This is direct root-mount evidence, independent of drive presence.
4. **nvme_is_fitted:** Yes. `/sys/block/nvme0n1` exists and its model is
   `K-RICARD NVME 256GB`.
5. **the_link_is_negotiated:** Yes. The NVMe endpoint reports both capability
   and negotiated status: 8 GT/s per lane, width x4.
6. **the_link_is_gen3:** Yes. 8 GT/s corresponds to Gen3. This SSD advertises
   Gen3 x4 and negotiates Gen3 x4, so there is no measured capability gap.
   The code selects the NVMe endpoint rather than the Wi-Fi controller.
7. **the_thermal_zones_are_readable:** Yes. Six zones returned temperatures;
   the maximum in this snapshot is 50.968 C. Sysfs millidegrees are divided
   by 1000. Unreadable sensors are omitted rather than assigned temperatures.
8. **the_power_mode_is_set:** Readable: 25W, mode ID 1. This is the current
   configuration, not MAXN SUPER. Whether 25W is the intended classroom
   mode must be compared with the instructor's requested configuration.

## Validation

- `python3 -m unittest -v test_probes.py`: seven tests pass, covering missing
  evidence, simulated filesystem isolation, RAM parsing, SD versus NVMe boot,
  absent storage, PCI endpoint selection, thermal units, and power-mode parsing.
- The live report has the same keys and value types as
  `sample_system_report.json`; all seven probes report `status: ok`.
- The ELMS Module 1 slides are not in this repository. Implementation follows
  the available README, starter interfaces, and sample report.
