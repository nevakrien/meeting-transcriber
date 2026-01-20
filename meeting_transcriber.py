import argparse
import json
import os
import pathlib
import queue
import sys
from typing import Any, List, Optional

import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_MIN_SILENCE_MS = 700
DEFAULT_SILENCE_THRESH_DBFS = -40
DEFAULT_KEEP_SILENCE_MS = 300
DEFAULT_MAX_SEGMENT_MS = 120_000
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization"
DEFAULT_MODEL_CACHE_DIR = pathlib.Path("models/pyannote")
DEFAULT_DIARIZATION_DEVICE = "AUTO"


class RecordingInterrupted(Exception):
    pass


def _ensure_directory(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def record_microphone(
    output_path: pathlib.Path,
    duration_seconds: Optional[float],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> None:
    _ensure_directory(output_path.parent)

    audio_queue: "queue.Queue[List[List[float]]]" = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"Input status: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    print("Recording... Press Ctrl+C to stop.")
    with sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        subtype="PCM_16",
    ) as file_handle:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            callback=callback,
        ):
            try:
                if duration_seconds is None:
                    while True:
                        file_handle.write(audio_queue.get())
                else:
                    total_frames = int(duration_seconds * sample_rate)
                    written_frames = 0
                    while written_frames < total_frames:
                        data = audio_queue.get()
                        file_handle.write(data)
                        written_frames += len(data)
            except KeyboardInterrupt as exc:
                if duration_seconds is None:
                    print("Recording stopped.")
                    return
                raise RecordingInterrupted("Recording interrupted") from exc


def _split_large_chunk(chunk: AudioSegment, max_ms: int) -> List[AudioSegment]:
    if len(chunk) <= max_ms:
        return [chunk]

    segments: List[AudioSegment] = []
    start = 0
    while start < len(chunk):
        end = min(start + max_ms, len(chunk))
        segments.append(chunk[start:end])
        start = end
    return segments


def split_audio(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    silence_thresh_dbfs: int = DEFAULT_SILENCE_THRESH_DBFS,
    keep_silence_ms: int = DEFAULT_KEEP_SILENCE_MS,
    max_segment_ms: int = DEFAULT_MAX_SEGMENT_MS,
) -> List[pathlib.Path]:
    _ensure_directory(output_dir)

    audio = AudioSegment.from_file(str(input_path))
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_dbfs,
        keep_silence=keep_silence_ms,
    )

    if not chunks:
        chunks = [audio]

    output_paths: List[pathlib.Path] = []
    index = 1
    for chunk in chunks:
        for sub_chunk in _split_large_chunk(chunk, max_segment_ms):
            filename = f"{input_path.stem}_chunk_{index:03d}.wav"
            output_path = output_dir / filename
            sub_chunk.export(output_path, format="wav")
            output_paths.append(output_path)
            index += 1

    return output_paths


def _sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _configure_hf_cache(cache_dir: pathlib.Path) -> pathlib.Path:
    hf_cache_dir = cache_dir / "huggingface"
    _ensure_directory(hf_cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache_dir))
    return hf_cache_dir


def _prepare_openvino_segmentation(
    pipeline, model_id: str, cache_dir: pathlib.Path, device: str
) -> None:
    import importlib

    np = importlib.import_module("numpy")
    ov = importlib.import_module("openvino")
    torch = importlib.import_module("torch")

    model_slug = _sanitize_model_id(model_id)
    ov_model_path = cache_dir / f"{model_slug}_segmentation.xml"
    onnx_path = ov_model_path.with_suffix(".onnx")

    core = ov.Core()

    if not ov_model_path.exists():
        torch.onnx.export(
            pipeline._segmentation.model,
            torch.zeros((1, 1, 80000)),
            onnx_path,
            input_names=["chunks"],
            output_names=["outputs"],
            dynamic_axes={"chunks": {0: "batch_size", 2: "wave_len"}},
        )
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, str(ov_model_path))
    else:
        ov_model = core.read_model(ov_model_path)

    compiled_model = core.compile_model(ov_model, device)
    output_port = compiled_model.output(0)
    input_port = compiled_model.input(0)

    def infer_segmentation(chunks) -> Any:
        if hasattr(chunks, "detach"):
            data = chunks.detach().cpu().numpy()
        else:
            data = np.asarray(chunks)
        result = compiled_model({input_port: data})
        return result[output_port]

    pipeline._segmentation.infer = infer_segmentation


def diarize_audio(
    input_path: pathlib.Path,
    model_id: str = DEFAULT_DIARIZATION_MODEL,
    cache_dir: pathlib.Path = DEFAULT_MODEL_CACHE_DIR,
    device: str = DEFAULT_DIARIZATION_DEVICE,
) -> List[dict]:
    _ensure_directory(cache_dir)
    _configure_hf_cache(cache_dir)

    import importlib

    pyannote_audio = importlib.import_module("pyannote.audio")
    pipeline = pyannote_audio.Pipeline.from_pretrained(model_id)
    _prepare_openvino_segmentation(pipeline, model_id, cache_dir, device)

    diarization = pipeline({"uri": input_path.stem, "audio": str(input_path)})
    turns: List[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    return turns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record audio and split into chunks on silence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Record from microphone")
    record_parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("recordings/latest.wav"),
        help="Output WAV path.",
    )
    record_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (omit for Ctrl+C).",
    )
    record_parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
    )
    record_parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
    )
    record_parser.add_argument(
        "--split",
        action="store_true",
        help="Split immediately after recording.",
    )
    record_parser.add_argument(
        "--segments-dir",
        type=pathlib.Path,
        default=pathlib.Path("segments"),
    )

    split_parser = subparsers.add_parser("split", help="Split existing file")
    split_parser.add_argument("input", type=pathlib.Path)
    split_parser.add_argument(
        "--segments-dir",
        type=pathlib.Path,
        default=pathlib.Path("segments"),
    )

    diarize_parser = subparsers.add_parser("diarize", help="Run speaker diarization")
    diarize_parser.add_argument("input", type=pathlib.Path)
    diarize_parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output JSON path.",
    )
    diarize_parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("segments"),
        help="Directory for diarization output when --output is omitted.",
    )
    diarize_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_DIARIZATION_MODEL,
        help="Hugging Face model id for pyannote pipeline.",
    )
    diarize_parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=DEFAULT_MODEL_CACHE_DIR,
        help="Directory to cache pipeline and OpenVINO models.",
    )
    diarize_parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DIARIZATION_DEVICE,
        help="OpenVINO device (e.g. CPU, AUTO).",
    )

    for subparser in (record_parser, split_parser):
        subparser.add_argument(
            "--min-silence",
            type=int,
            default=DEFAULT_MIN_SILENCE_MS,
        )
        subparser.add_argument(
            "--silence-thresh",
            type=int,
            default=DEFAULT_SILENCE_THRESH_DBFS,
            help="Silence threshold in dBFS.",
        )
        subparser.add_argument(
            "--keep-silence",
            type=int,
            default=DEFAULT_KEEP_SILENCE_MS,
        )
        subparser.add_argument(
            "--max-segment",
            type=int,
            default=DEFAULT_MAX_SEGMENT_MS,
        )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "record":
        try:
            record_microphone(
                args.output,
                args.duration,
                sample_rate=args.sample_rate,
                channels=args.channels,
            )
        except RecordingInterrupted:
            print("Recording interrupted before duration finished.")
            return

        if args.split:
            output_paths = split_audio(
                args.output,
                args.segments_dir,
                min_silence_ms=args.min_silence,
                silence_thresh_dbfs=args.silence_thresh,
                keep_silence_ms=args.keep_silence,
                max_segment_ms=args.max_segment,
            )
            print(f"Wrote {len(output_paths)} segments to {args.segments_dir}")
        return

    if args.command == "split":
        output_paths = split_audio(
            args.input,
            args.segments_dir,
            min_silence_ms=args.min_silence,
            silence_thresh_dbfs=args.silence_thresh,
            keep_silence_ms=args.keep_silence,
            max_segment_ms=args.max_segment,
        )
        print(f"Wrote {len(output_paths)} segments to {args.segments_dir}")
        return

    if args.command == "diarize":
        output_path = args.output
        if output_path is None:
            output_path = args.output_dir / f"{args.input.stem}_speakers.json"
        _ensure_directory(output_path.parent)

        turns = diarize_audio(
            args.input,
            model_id=args.model,
            cache_dir=args.cache_dir,
            device=args.device,
        )
        output_payload = {"audio": str(args.input), "segments": turns}
        output_path.write_text(json.dumps(output_payload, indent=2))
        print(f"Wrote diarization to {output_path}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
