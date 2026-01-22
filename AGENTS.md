# AGENTS

Reference notes for working in this repo.

## Environment
- run `micromamba activate meeting` this should get you in the right venv. make sure commands ran only in that venv
## System packages

- Linux: `sudo apt-get install -y portaudio19-dev ffmpeg`
- Windows: install FFmpeg and add it to PATH

## Run the CLI

```bash
python meeting_transcriber.py record --split
python meeting_transcriber.py split recordings/latest.wav
```

## Tests

- No tests yet. Add pytest-based tests when diarization alignment lands.

## TODO

- Add OpenVINO-only diarization pipeline (OpenVINO runtime + model conversion).
- Avoid Hugging Face runtime dependencies; if HF model download is needed, cache weights locally and run inference via OpenVINO only.
- Add pytest + a synthetic alignment test for transcript + diarization merge logic.
- Document diarization setup steps in README (model download/conversion, OpenVINO deps).

- NEVER use importlib instead everything should import globally and if it doesnt exist in the venv thats a bug

## Notes

- Keep output audio in `recordings/` and `segments/`.
- Avoid adding large binary samples to git history.
- We wana avoid HF as a hard requirment as much as possible. this is because HF tends to run into enviorment bugs and weird cashes. openvino is the prefered tech sinc it does not suffer these issues when used by itself.