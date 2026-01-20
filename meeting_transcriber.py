import argparse
import pathlib
import queue
import sys
from typing import Iterable, List, Optional

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


def _split_large_chunk(chunk: AudioSegment, max_ms: int) -> Iterable[AudioSegment]:
    if len(chunk) <= max_ms:
        yield chunk
        return

    start = 0
    while start < len(chunk):
        end = min(start + max_ms, len(chunk))
        yield chunk[start:end]
        start = end


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

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
