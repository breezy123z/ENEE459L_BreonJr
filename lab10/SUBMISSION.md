# Lab 10 work and submission

Eight required functions are implemented in capture.py. See IMPLEMENTATION_NOTES.md for steps taken, the function table, source-code inconsistencies, and results.

## Run on the Jetson

```bash
cd ~/ENEE459L/lab10
python3 capture.py test_voice.wav -o generated_test_voice_report.json
python3 -m unittest -v test_capture.py
```

The original test_voice_report.json remains untouched for comparison. Python's standard library is sufficient; no packages need installing.

## Optional future microphone recording

```bash
arecord -l
# Substitute the actual card and device found above; do not assume card 2.
arecord -D plughw:CARD,DEVICE -f S16_LE -r 44100 -c 1 -d 10 my_voice.wav
python3 capture.py my_voice.wav -o my_voice_report.json
```

Make a copy of session_template.json and fill all 13 values with actual observations, including the three format fields, before running:

```bash
python3 capture.py my_voice.wav --session my_session.json -o my_voice_report.json
```

Do not submit the empty-string template as completed metadata. Do not overwrite the instructor's test_voice.wav with your recording. Review any personal recording before adding it to a public repository.

The supplied sample is usable by the provided level thresholds, with no clipping and about 11.8 of 16 bits reached. Its ten handwritten capture choices are unknown. At the user's direction, this submission uses the instructor-provided test because microphones are unavailable. A fresh recording is not part of this submission. Missing handwritten metadata is reported honestly rather than guessed.

Branch: https://github.com/breezy123z/ENEE459L_BreonJr/tree/solution10/lab10
