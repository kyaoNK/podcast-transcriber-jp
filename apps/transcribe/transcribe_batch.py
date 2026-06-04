from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.common.sleep_guard import prevent_sleep
from core.review.transcribe_audio import (
    transcribe_with_speaker_tracks,
    transcribe_with_diarization,
    transcribe_with_faster_whisper,
    transcribe_with_whisperx,
)
from apps.transcribe.transcribe import ProcessingLogger, write_transcript_text_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="複数音声ファイルを順番に文字起こしする。")
    p.add_argument("--audio", nargs="*", default=[], help="処理する音声ファイル。複数指定可")
    p.add_argument("--input-dir", default=None, help="フォルダ内の音声を処理する場合の入力フォルダ")
    p.add_argument("--glob", default="*.wav", help="--input-dir 用のglob。既定: *.wav")
    p.add_argument("--recursive", action="store_true", help="--input-dir を再帰検索する")
    p.add_argument("--output-dir", default=None, help="出力先フォルダ。省略時は各音声と同じフォルダ")
    p.add_argument("--output-suffix", default="_diarized.json", help="出力JSON suffix")
    p.add_argument("--text-suffix", default="_diarized.txt", help="出力TXT suffix")
    p.add_argument("--no-text", action="store_true", help="TXTを出力しない")
    p.add_argument("--skip-existing", action="store_true", help="出力JSONが既にあればスキップ")
    p.add_argument("--stop-on-error", action="store_true", help="1件失敗したら停止する")
    p.add_argument("--log-output", default=None, help="一括処理ログtxt")
    p.add_argument("--readable-log-output", default=None, help="一括処理Markdownログ")

    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--transcriber", choices=["whisperx", "faster-whisper"], default="faster-whisper")
    p.add_argument("--language", default="ja")
    p.add_argument("--initial-prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--hotwords", default=None)
    p.add_argument("--hotwords-file", default=None)
    p.add_argument("--diarize", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--speaker-track", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--speaker-track-active-db", default="auto")
    p.add_argument("--speaker-track-overlap-db", default="auto")
    p.add_argument("--speaker-track-margin", default="auto")
    return p.parse_args()


def resolve_batch_log_paths(args: argparse.Namespace, audio_files: list[Path]) -> tuple[Path, Path]:
    base = Path(args.output_dir) if args.output_dir else audio_files[0].parent
    txt = Path(args.log_output) if args.log_output else base / "transcribe_batch.log.txt"
    md = Path(args.readable_log_output) if args.readable_log_output else base / "transcribe_batch.log.md"
    return txt, md


def read_text_arg(value: str | None, file_value: str | None) -> str | None:
    if value:
        return value
    if file_value:
        return Path(file_value).read_text(encoding="utf-8").strip()
    return None


def collect_audio_files(args: argparse.Namespace) -> list[Path]:
    files = [Path(p) for p in args.audio]
    if args.input_dir:
        base = Path(args.input_dir)
        pattern = f"**/{args.glob}" if args.recursive else args.glob
        files.extend(base.glob(pattern))
    # 順序を安定させ、重複を除く
    seen: set[Path] = set()
    result: list[Path] = []
    for path in sorted((p.resolve() for p in files), key=lambda p: str(p).lower()):
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        result.append(path)
    return result


def parse_speaker_tracks(values: list[str]) -> list[tuple[str, Path]]:
    tracks: list[tuple[str, Path]] = []
    for i, value in enumerate(values, start=1):
        if "=" in value:
            label, path = value.split("=", 1)
            label = label.strip() or f"TRACK_{i}"
        else:
            label, path = f"TRACK_{i}", value
        tracks.append((label, Path(path.strip().strip('"'))))
    return tracks


def output_paths(audio: Path, args: argparse.Namespace) -> tuple[Path, Path | None]:
    out_dir = Path(args.output_dir) if args.output_dir else audio.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{audio.stem}{args.output_suffix}"
    text_path = None if args.no_text else out_dir / f"{audio.stem}{args.text_suffix}"
    return json_path, text_path


def transcribe_one(
    audio: Path,
    json_path: Path,
    text_path: Path | None,
    args: argparse.Namespace,
    initial_prompt: str | None,
    hotwords: str | None,
    speaker_tracks: list[tuple[str, Path]],
) -> int:
    if args.skip_existing and json_path.exists():
        print(f"[batch] skip existing: {json_path}", flush=True)
        return 0

    print(f"[batch] start: {audio}", flush=True)
    if speaker_tracks:
        segments = transcribe_with_speaker_tracks(
            audio,
            json_path,
            speaker_tracks=speaker_tracks,
            model_size=args.whisper_model,
            language=args.language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            active_db=args.speaker_track_active_db,
            overlap_db=args.speaker_track_overlap_db,
            margin=args.speaker_track_margin,
        )
    elif args.diarize:
        segments = transcribe_with_diarization(
            audio,
            json_path,
            model_size=args.whisper_model,
            language=args.language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            hf_token=args.hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
    elif args.transcriber == "whisperx":
        if hotwords:
            print(f"[batch] warning: --hotwords は whisperx では未使用: {audio}", file=sys.stderr, flush=True)
        segments = transcribe_with_whisperx(
            audio,
            json_path,
            model_size=args.whisper_model,
            language=args.language,
            initial_prompt=initial_prompt,
        )
    else:
        segments = transcribe_with_faster_whisper(
            audio,
            json_path,
            model_size=args.whisper_model,
            language=args.language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )

    if text_path is not None:
        paths = write_transcript_text_outputs(segments, text_path)
        print(f"[batch] review saved: {paths['review']}", flush=True)
        print(f"[batch] clean saved: {paths['clean']}", flush=True)
        print(f"[batch] simple saved: {paths['simple']}", flush=True)
        print(f"[batch] quality summary saved: {paths['quality_md']}", flush=True)

    print(f"[batch] saved: {json_path} ({len(segments)} segments)", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    audio_files = collect_audio_files(args)
    if not audio_files:
        print("[batch] audio file not found", file=sys.stderr)
        return 1

    log_txt, log_md = resolve_batch_log_paths(args, audio_files)
    logger = ProcessingLogger(log_txt, log_md)
    logger.event(
        "一括処理開始",
        {
            "files": len(audio_files),
            "output_dir": args.output_dir,
            "skip_existing": args.skip_existing,
            "log_txt": str(log_txt),
            "log_md": str(log_md),
        },
    )

    initial_prompt = read_text_arg(args.initial_prompt, args.prompt_file)
    hotwords = read_text_arg(args.hotwords, args.hotwords_file)
    speaker_tracks = parse_speaker_tracks(args.speaker_track)
    logger.event(
        "プロンプト/用語読み込み完了",
        {
            "initial_prompt_chars": len(initial_prompt or ""),
            "hotwords_chars": len(hotwords or ""),
            "speaker_tracks": len(speaker_tracks),
        },
    )

    print(f"[batch] files: {len(audio_files)}", flush=True)
    ok = 0
    failed: list[tuple[Path, str]] = []
    for index, audio in enumerate(audio_files, start=1):
        json_path, text_path = output_paths(audio, args)
        print(f"[batch] ({index}/{len(audio_files)})", flush=True)
        logger.event(
            "ファイル処理開始",
            {"index": index, "total": len(audio_files), "audio": str(audio), "json": str(json_path)},
        )
        try:
            transcribe_one(audio, json_path, text_path, args, initial_prompt, hotwords, speaker_tracks)
            ok += 1
            logger.event(
                "ファイル処理完了",
                {"index": index, "total": len(audio_files), "audio": str(audio), "json": str(json_path), "ok": ok},
            )
        except Exception as e:
            failed.append((audio, str(e)))
            print(f"[batch] failed: {audio}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            logger.event(
                "ファイル処理失敗",
                {"index": index, "total": len(audio_files), "audio": str(audio), "error": str(e), "failed": len(failed)},
            )
            if args.stop_on_error:
                break

    print(f"[batch] done: ok={ok}, failed={len(failed)}", flush=True)
    logger.event("一括処理完了", {"ok": ok, "failed": len(failed), "total": len(audio_files)})
    if failed:
        for audio, reason in failed:
            print(f"[batch] NG: {audio} :: {reason}", file=sys.stderr, flush=True)
        logger.close("failed")
        return 2
    logger.close("done")
    return 0


if __name__ == "__main__":
    with prevent_sleep():
        exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exit_code))
