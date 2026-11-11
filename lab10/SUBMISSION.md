# Lab 10 work and submission

Eight required functions are implemented in capture.py. See IMPLEMENTATION_NOTES.md for steps taken, the function table, source-code inconsistencies, and results.

## Run on the Jetson

```bash
cd ~/ENEE459L/lab10
python3 capture.py test_voice.wav -o generated_test_voice_report.json
python3 -m unittest -v test_capture.py
```

The original test_voice_report.json remains untouched for comparison. Python's standard library is sufficient; no packages need installing.

## Record your own audio when a microphone is connected

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

Do not submit the null template as completed metadata. Do not overwrite the instructor's test_voice.wav with your recording. Review any personal recording before adding it to a public repository.

The supplied sample is usable by the provided level thresholds, with no clipping and about 11.8 of 16 bits reached. Its ten handwritten capture choices are unknown. A fresh Jetson microphone recording is still pending: the initial device enumeration showed only internal NVIDIA APE interfaces, not a USB microphone.

Branch: https://github.com/breezy123z/ENEE459L_BreonJr/tree/solution10/lab10
