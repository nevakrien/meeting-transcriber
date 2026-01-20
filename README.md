# meeting-transcriber

Local-first meeting transcription tooling for sensitive recordings. The current CLI records audio from a microphone, splits it into segments based on silence or a maximum length, and prepares the audio for downstream transcription and diarization workflows.

## Requirements

- Python 3.10+
- FFmpeg on PATH (used by pydub)
- PortAudio system library (Linux only)

### Linux setup

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev ffmpeg
```

### Windows setup

- Install FFmpeg and add it to PATH
- The `sounddevice` wheels bundle PortAudio on Windows, so no extra system install is typically needed

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install sounddevice soundfile pydub
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

## Usage

Record from the microphone and stop with Ctrl+C:

```bash
.venv/bin/python meeting_transcriber.py record --split
```

Record for a fixed duration (seconds) and split:

```bash
.venv/bin/python meeting_transcriber.py record --duration 120 --split
```

Split an existing WAV file:

```bash
.venv/bin/python meeting_transcriber.py split recordings/latest.wav
```

## Defaults

- Sample rate: 16 kHz, mono
- Silence threshold: -40 dBFS
- Min silence: 700 ms
- Keep silence: 300 ms
- Max segment length: 120 seconds

Override with `--min-silence`, `--silence-thresh`, `--keep-silence`, or `--max-segment`.

## Project status

- Audio capture and segmentation is working.
- Diarization + transcription are next; OpenVINO-backed diarization is planned.
