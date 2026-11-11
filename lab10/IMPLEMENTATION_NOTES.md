# Lab 10 implementation notes

## Objective and scope

Check that a PCM microphone recording has a coherent file format, a usable signal level, and enough recorded capture choices to reproduce it. Eight function stubs in capture.py were completed. The supplied test_voice.wav is instructor audio, not a new microphone recording. No voice identification or AI inference is performed.

## Steps taken

1. Fetched instructor main at 716bcff (adding lab 10 questions).
2. Created solution10 from solution9 and merged the lab without making another Jetson checkout or replacing earlier solutions.
3. Saved the untouched starter locally as ../lab10-original.tar. Git history also preserves it.
4. Read README.md, capture.py, schema.py, and the reference report. The ELMS Module 1 slides are not included in the repository and have not been checked.
5. Implemented the eight functions below with numbered sections and shared numeric validation.
6. Added 23 tests for geometry, boundaries, metadata, malformed WAVs, 24-bit stereo, and exact reference-report comparison.
7. Fixed two supplied-code issues: the console always printed 'of 8 bits', and the WAV reader trusted truncated payloads/contradictory header values. Added an optional --session argument so handwritten choices can actually be audited from a JSON file.
8. Verified locally: all 23 tests pass, and every field matches the supplied report exactly.
9. Connected to the Jetson and inspected arecord -l. Only NVIDIA APE internal interfaces appeared; no USB microphone was listed. No new audio was captured.

## Functions written

| Function | What it does | Why it matters |
|---|---|---|
| wav_bytes | Rounds sample_rate * seconds to frames; multiplies by channels * bit_depth/8 and adds a configurable header size. | Predicts PCM payload and canonical WAV file size. |
| duration_from_bytes | Uses divmod(payload, frame stride), then complete frames / sample rate. | Reports duration and exposes partial-frame bytes. |
| header_consistency | Compares sample_rate, channels, bit_depth; recomputes duration from header bytes. | Detects disagreement between stated format and file metadata. |
| dbfs | Returns 20*log10(amplitude), or None at zero; validates finite [0,1]. | Converts levels without serializing infinity for silence. |
| bits_in_use | Returns max(0, bit_depth + peak_dbfs/DB_PER_BIT), or zero for silence. | Estimates how much available amplitude resolution is reached. |
| level_sentence | Formats peak/RMS/crest; names silence; adds clipped count and percentage if nonzero. | Makes a numeric level report readable. |
| record_gaps | Separates present, missing, and blank entries among the 13 capture choices. | Exposes information the WAV header cannot store. |
| record_sentence | Lists exact missing/blank choices or reports all 13 complete. | Gives an actionable documentation checklist. |
| _finite_number / _nonnegative_int | Shared validation, rejecting booleans, non-finite numbers, and invalid counts. | Keeps malformed inputs from producing believable results. |

## Supplied functions and targeted changes

CaptureError, _positive_int, block_align, amplitude_from_dbfs, full_scale_range_db, and level_report were already implemented. Their bodies are unchanged. run_analysis was already provided and is unchanged. _parse_wav was supplied but strengthened to reject truncated RIFF/chunks, missing data, non-PCM formats, partial frames, and inconsistent byte_rate/block_align; unknown padded chunks remain supported. main was supplied; its bit-depth display was corrected and --session added. schema.py, README.md, the sample WAV, and reference JSON remain unchanged.

## Important starter inconsistencies and assumptions

- capture.py defines capture-environment choices (device_name, microphone distance, etc.) and CLIP_RATE_MAX=0.001. schema.py instead defines signal-processing choices (frame_length, n_mels, etc.) and CLIP_RATE_MAX=0.0001. The reference JSON agrees with capture.py, so its definitions were retained. These two lists should not be treated as interchangeable.
- The README's `python capture.py` command omits the required WAV path. Use the commands in SUBMISSION.md.
- Duration helper output `samples` counts complete PCM frames; level_report.samples counts individual channel samples. For stereo those counts differ by two.
- Frame rounding uses Python round (ties to even). Durations and payloads may be zero for arithmetic helpers; actual analysis requires nonempty audio.
- Header mismatches are reported as field names. agree compares the three format fields; alignment and recomputed duration are separate returned evidence. stated_seconds is preserved, not silently substituted for measured duration.
- wav_bytes predicts a canonical 44-byte header unless given another size. Extra RIFF chunks can make a real file larger than that estimate. The parser accepts those chunks, but the supplied report layout does not expose their total overhead.
- Missing/None/whitespace and empty collections count as unrecorded. Numeric zero and False remain valid named choices, for example no high-pass filter. This checks completeness, not whether a description is truthful.
- bits_in_use is a peak-based resolution estimate, not measured converter ENOB or signal-to-noise ratio.
- The supplied report's header_consistency.agrees compares header values with defaults read from that same header unless --session is supplied. It does not independently prove the recording device ran at the advertised rate.

## Supplied audio findings

| Finding | Result | Interpretation |
|---|---:|---|
| Rate / channels / depth | 44,100 Hz / mono / 16 bits | Instructor's provided PCM format |
| Duration | 10.0 seconds | 441,000 complete frames |
| Payload / canonical total | 882,000 / 882,044 bytes | 2 bytes per mono frame |
| Peak | -25.31 dBFS | Within supplied usable window of -40 to -1 dBFS |
| RMS | -51.01 dBFS | Average signal energy is much lower than peak |
| Crest factor | 25.70 dB | Peak-to-RMS ratio |
| Clipped samples | 0 | No samples reached the clipping rail |
| Bits in use / unused headroom | 11.795 / 4.205 | Peak-based estimate from a 16-bit file |
| Capture choices named | 3 of 13 | Header supplies rate/depth/channels; ten handwritten choices are unknown |

Do not fill those ten choices using guesses about the instructor's recording. For a new capture, fill session_template.json with actual observed settings, save it under another name, then pass it with --session.

## Jetson verification

All 23 tests passed on the Jetson in 0.199 seconds. The generated report matches the supplied reference exactly. Saved evidence: test-results.txt, run-results.txt, generated_test_voice_report.json, and audio-devices.txt. The recording remains the instructor sample; no new voice was captured.
