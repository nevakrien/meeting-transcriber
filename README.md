# meeting-transcriber

Local-first meeting transcription tooling for sensitive recordings. The current CLI records audio from a microphone, splits it into segments based on silence or a maximum length, and prepares audio for downstream transcription and diarization workflows.

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

in the right venv

```bash
pip install -r requirments.txt
```

On Windows, replace `python` with `.venv\Scripts\python.exe`.

## Usage

Record from microphone and stop with Ctrl+C:

```bash
python meeting_transcriber.py record --split
```

Record for a fixed duration (seconds) and split:

```bash
python meeting_transcriber.py record --duration 120 --split
```

Split an existing WAV file:

```bash
python meeting_transcriber.py split recordings/latest.wav
```

Run speaker diarization:

```bash
python meeting_transcriber.py diarize recordings/latest.wav
```

## Defaults

- Sample rate: 16 kHz, mono
- Silence threshold: -40 dBFS
- Min silence: 700 ms
- Keep silence: 300 ms
- Max segment length: 120 seconds

Override with `--min-silence`, `--silence-thresh`, `--keep-silence`, or `--max-segment`.

## Project status

- ✅ Audio capture and segmentation working
- ✅ OpenVINO-optimized speaker diarization 
- 🔄 Transcription integration next
- 📋 Test data included (see `test_data/`)

## Test Data

Small sample of Yoruba speech data included in `test_data/yoruba_samples/` for testing diarization and transcription algorithms. See `test_data/README.md` for details.