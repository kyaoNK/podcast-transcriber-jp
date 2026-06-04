"""音声→文字起こし 単独CLI。

PoCとは独立に、編集中に音声から文字起こしを取りたいときに使う。
既存の `src/review/transcribe_audio.py` を再利用する薄いラッパー。

`--diarize` を付けると pyannote.audio で word単位話者識別を行う (HF_TOKEN必須)。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.common.sleep_guard import prevent_sleep
from core.review.transcribe_audio import (
    TranscriptSegment,
    atomic_write_text,
    atomic_write_json,
    build_transcript_quality_summary,
    detect_compute_backend,
    format_compute_backend,
    load_transcript,
    make_mix_from_speaker_tracks,
    make_simple_segments,
    preprocess_speaker_tracks_for_asr,
    transcribe_with_speaker_track_fusion,
    transcribe_with_speaker_tracks,
    transcribe_with_diarization,
    transcribe_with_faster_whisper,
    transcribe_with_whisperx,
    transcript_to_review_markdown,
    transcript_quality_summary_to_markdown,
    transcript_to_simple_lines,
    transcript_to_text_lines,
    validate_audio_alignment,
)

LOG_ENCODING = "utf-8-sig"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="音声ファイルを Whisper で文字起こしする独立CLI。",
    )
    p.add_argument("--audio", required=False, help="入力 WAV/MP3/MP4 等")
    p.add_argument(
        "--output",
        default=None,
        help="出力 JSON パス。省略時は --audio と同じディレクトリの transcript.json",
    )
    p.add_argument(
        "--text-output",
        default=None,
        help="時刻付きプレーンテキスト (.txt) も出す場合のパス",
    )
    p.add_argument("--log-output", default=None, help="処理ログtxt。省略時は出力JSON横に .log.txt")
    p.add_argument("--readable-log-output", default=None, help="読みやすいMarkdownログ。省略時は出力JSON横に .log.md")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument(
        "--transcriber",
        choices=["whisperx", "faster-whisper"],
        default="faster-whisper",
        help="既定は faster-whisper (依存が軽く Windows でも動く)",
    )
    p.add_argument("--language", default="ja")
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="出力 JSON が既にあれば文字起こしをスキップして再利用する",
    )
    p.add_argument(
        "--initial-prompt",
        default=None,
        help="Whisper への initial_prompt (固有名詞・スタイル誘導)",
    )
    p.add_argument(
        "--prompt-file",
        default=None,
        help="initial_prompt をファイルから読む (--initial-prompt より優先度低)",
    )
    p.add_argument(
        "--hotwords",
        default=None,
        help="faster-whisper の hotwords。用語リストをカンマ/改行区切りで指定",
    )
    p.add_argument(
        "--hotwords-file",
        default=None,
        help="hotwords をファイルから読む (--hotwords より優先度低)",
    )
    p.add_argument(
        "--diarize",
        action="store_true",
        help="pyannote.audio で word単位話者識別を行う (HF_TOKEN 必須)",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace アクセストークン (省略時は環境変数 HF_TOKEN)",
    )
    p.add_argument("--min-speakers", type=int, default=None, help="最小話者数 (例: 2)")
    p.add_argument("--max-speakers", type=int, default=None, help="最大話者数 (例: 3)")
    p.add_argument(
        "--speaker-track",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="同期済み話者別トラック。例: --speaker-track A=a.wav --speaker-track B=b.wav",
    )
    p.add_argument(
        "--mix-from-speaker-tracks",
        action="store_true",
        help="--speaker-track からWhisper入力用のmix WAVを作って使う",
    )
    p.add_argument("--mix-output", default=None, help="mix WAVの保存先。省略時は出力JSON横")
    p.add_argument("--speaker-track-active-db", default="auto", help="このdB以上を発話中とみなす。既定: auto")
    p.add_argument("--speaker-track-overlap-db", default="auto", help="上位2トラックの差がこのdB以内ならOVERLAP。既定: auto")
    p.add_argument("--speaker-track-margin", default="auto", help="RMS判定時に前後へ足す秒数。既定: auto")
    p.add_argument(
        "--speaker-track-mode",
        choices=["fusion", "rms"],
        default="fusion",
        help="話者別トラック利用時の処理。fusion=mix+各トラックWhisper候補統合、rms=従来のRMS割当のみ",
    )
    p.add_argument(
        "--fusion-version",
        choices=["v1", "v2"],
        default="v1",
        help="fusion時の話者識別方式。v1=現状安定版、v2=ASR+RMS+mix照合の試験版",
    )
    p.add_argument("--duration-mismatch-tolerance", type=float, default=2.0, help="mix音声と話者別トラックの許容長さ差。既定: 2秒")
    p.add_argument("--allow-duration-mismatch", action="store_true", help="長さ不一致でも強制実行する。通常は非推奨")
    p.add_argument("--no-asr-preprocess", action="store_true", help="話者別トラックのASR用前処理を無効化する")
    p.add_argument(
        "--no-crosstalk-cancel",
        dest="crosstalk_cancel",
        action="store_false",
        help="話者間ブリードイン(クロストーク)の除去を無効化する。デフォルトは有効",
    )
    p.set_defaults(crosstalk_cancel=True)
    return p.parse_args()


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d} ({seconds / 3600:.3f}時間)"


class ProcessingLogger:
    def __init__(self, txt_path: Path, md_path: Path) -> None:
        self.txt_path = txt_path
        self.md_path = md_path
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self._last_markdown_write = 0.0
        self._markdown_write_interval = 15.0
        self._last_logged_percent = -5.0
        self.txt_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_text(self.txt_path, f"started: {now}\n", encoding=LOG_ENCODING)
        self._write_markdown(status="running", force=True)

    def _should_write_txt(self, message: str, data: dict[str, Any]) -> bool:
        if any(token in message for token in ("開始", "完了", "失敗", "警告", "再利用", "全処理")):
            return True
        percent = data.get("progress_percent")
        if isinstance(percent, (int, float)) and float(percent) >= self._last_logged_percent + 5.0:
            self._last_logged_percent = float(percent)
            return True
        return False

    def event(self, message: str, data: dict[str, Any] | None = None) -> None:
        elapsed = time.perf_counter() - self.started
        record = {
            "elapsed": elapsed,
            "elapsed_text": _format_elapsed(elapsed),
            "message": message,
            "data": data or {},
        }
        self.events.append(record)
        event_data = data or {}
        if self._should_write_txt(message, event_data):
            data_text = " " + json.dumps(event_data, ensure_ascii=False, default=str) if event_data else ""
            line = f"[{record['elapsed_text']}] {message}{data_text}\n"
            with self.txt_path.open("a", encoding="utf-8") as f:
                f.write(line)
        print(
            "[progress] "
            + json.dumps(
                {
                    "elapsed_text": record["elapsed_text"],
                    "message": message,
                    "data": data or {},
                },
                ensure_ascii=False,
                default=str,
            ),
            flush=True,
        )
        if elapsed - self._last_markdown_write >= self._markdown_write_interval:
            self._write_markdown(status="running", force=False)

    def close(self, status: str = "done") -> None:
        elapsed = time.perf_counter() - self.started
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.txt_path.open("a", encoding="utf-8") as f:
            f.write(f"finished: {now}\n")
            f.write(f"status: {status}\n")
            f.write(f"total_elapsed: {_format_elapsed(elapsed)}\n")
        self._write_markdown(status=status, force=True)

    def _write_markdown(self, status: str, *, force: bool = False) -> None:
        elapsed = time.perf_counter() - self.started
        if not force and elapsed - self._last_markdown_write < self._markdown_write_interval:
            return
        lines = [
            "# 処理ログ",
            "",
            f"- 状態: `{status}`",
            f"- 経過時間: `{_format_elapsed(elapsed)}`",
            f"- txtログ: `{self.txt_path}`",
            "",
            "## 進捗",
            "",
        ]
        shown_events = self.events if status != "running" else self.events[-30:]
        if status == "running" and len(self.events) > len(shown_events):
            lines.append(f"> 実行中は直近 {len(shown_events)} 件のみ表示します。完了時に全件を書き出します。")
            lines.append("")
        for index, e in enumerate(shown_events, start=1):
            lines.append(f"### {index}. {e['message']}")
            lines.append("")
            lines.append(f"- 経過時間: `{e['elapsed_text']}`")
            if e["data"]:
                lines.append("- 詳細:")
                for k, v in e["data"].items():
                    lines.append(f"  - `{k}`: {v}")
            lines.append("")
        atomic_write_text(self.md_path, "\n".join(lines) + "\n", encoding=LOG_ENCODING)
        self._last_markdown_write = elapsed


def resolve_log_paths(output: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    txt = Path(args.log_output) if args.log_output else output.with_suffix(".log.txt")
    md = Path(args.readable_log_output) if args.readable_log_output else output.with_suffix(".log.md")
    return txt, md


def resolve_transcript_text_paths(text_output: str | Path) -> tuple[Path, Path, Path]:
    """review/clean/simple の出力パスを揃える。

    `transcript.review.md` を渡した場合はその stem を基準にし、
    旧指定の `transcript.md` でも `transcript.review.md` 形式へ移行する。
    """
    path = Path(text_output)
    name = path.name
    if name.endswith(".review.md"):
        base = path.with_name(name.removesuffix(".review.md"))
    else:
        base = path.with_suffix("")
    return (
        base.with_name(base.name + ".review.md"),
        base.with_name(base.name + ".clean.md"),
        base.with_name(base.name + ".simple.txt"),
    )


def resolve_quality_summary_paths(text_output: str | Path) -> tuple[Path, Path]:
    review_path, _, _ = resolve_transcript_text_paths(text_output)
    base = review_path.with_name(review_path.name.removesuffix(".review.md"))
    return (
        base.with_name(base.name + ".quality_summary.json"),
        base.with_name(base.name + ".quality_summary.md"),
    )


def write_transcript_text_outputs(
    segments: list[TranscriptSegment],
    text_output: str | Path,
    speaker_track_segments: dict[str, list[TranscriptSegment]] | None = None,
) -> dict[str, Path]:
    review_path, clean_path, simple_path = resolve_transcript_text_paths(text_output)
    quality_json_path, quality_md_path = resolve_quality_summary_paths(text_output)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(review_path, transcript_to_review_markdown(segments), encoding="utf-8")
    atomic_write_text(clean_path, transcript_to_text_lines(segments), encoding="utf-8")
    # clean/simple は同じ採用テキストを使う。候補・詳細確認は review.md に集約する。
    simple_segments = segments
    atomic_write_text(simple_path, transcript_to_simple_lines(simple_segments), encoding="utf-8")
    quality_summary = build_transcript_quality_summary(segments)
    atomic_write_json(quality_json_path, quality_summary, encoding="utf-8")
    atomic_write_text(quality_md_path, transcript_quality_summary_to_markdown(quality_summary), encoding="utf-8")
    return {"review": review_path, "clean": clean_path, "simple": simple_path, "quality_json": quality_json_path, "quality_md": quality_md_path}


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


def resolve_output_path(audio: Path | None, output: str | None, speaker_tracks: list[tuple[str, Path]] | None = None) -> Path:
    if output:
        return Path(output)
    if audio is None:
        if speaker_tracks:
            return speaker_tracks[0][1].with_name("transcript_speaker_tracks.json")
        raise ValueError("--audio または --output が必要です")
    return audio.with_name("transcript.json")


def resolve_initial_prompt(args: argparse.Namespace) -> str | None:
    if args.initial_prompt:
        return args.initial_prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return None


def resolve_hotwords(args: argparse.Namespace) -> str | None:
    if args.hotwords:
        return args.hotwords
    if args.hotwords_file:
        return Path(args.hotwords_file).read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    args = parse_args()
    speaker_tracks = parse_speaker_tracks(args.speaker_track)
    audio = Path(args.audio) if args.audio else None
    if audio is not None and not audio.exists():
        print(f"audio not found: {audio}", file=sys.stderr)
        return 1
    if audio is None and not (args.mix_from_speaker_tracks and speaker_tracks):
        print("--audio が未指定です。話者別トラックからmixを作る場合は --mix-from-speaker-tracks を付けてください。", file=sys.stderr)
        return 1
    output = resolve_output_path(audio, args.output, speaker_tracks)
    log_txt, log_md = resolve_log_paths(output, args)
    logger = ProcessingLogger(log_txt, log_md)
    logger.event(
        "入力確認完了",
        {
            "audio": str(audio) if audio else None,
            "output": str(output),
            "speaker_tracks": len(speaker_tracks),
            "log_txt": str(log_txt),
            "log_md": str(log_md),
            "progress_percent": 1.0,
        },
    )
    backend = detect_compute_backend()
    logger.event(
        "実行環境確認",
        {
            "summary": format_compute_backend(backend),
            "faster_whisper_device": backend.get("faster_whisper_device"),
            "faster_whisper_cuda_devices": backend.get("faster_whisper_cuda_devices"),
            "faster_whisper_cuda_compute_types": backend.get("faster_whisper_cuda_compute_types"),
            "torch_cuda_available": backend.get("torch_cuda_available"),
            "torch_cuda_version": backend.get("torch_cuda_version"),
            "gpu_name": backend.get("gpu_name"),
            "progress_percent": 1.5,
        },
    )
    print(f"[transcribe] runtime: {format_compute_backend(backend)}")
    initial_prompt = resolve_initial_prompt(args)
    hotwords = resolve_hotwords(args)
    logger.event(
        "プロンプト/用語読み込み完了",
        {
            "initial_prompt_chars": len(initial_prompt or ""),
            "hotwords_chars": len(hotwords or ""),
            "progress_percent": 2.0,
        },
    )

    alignment_meta = None
    if speaker_tracks:
        try:
            alignment_meta = validate_audio_alignment(
                None if args.mix_from_speaker_tracks else audio,
                speaker_tracks,
                tolerance_sec=args.duration_mismatch_tolerance,
                allow_mismatch=args.allow_duration_mismatch,
            )
        except ValueError as e:
            logger.event("入力音声チェック失敗", {"error": str(e), "progress_percent": 2.5})
            logger.close("failed")
            print(str(e), file=sys.stderr)
            return 1
        logger.event(
            "入力音声チェック完了",
            {
                "ok": alignment_meta.get("ok"),
                "warnings": alignment_meta.get("warnings"),
                "errors_allowed": alignment_meta.get("allowed_mismatch", False),
                "audio_duration_sec": (alignment_meta.get("audio") or {}).get("duration_sec"),
                "track_duration_sec": alignment_meta.get("track_duration_sec"),
                "progress_percent": 2.5,
            },
        )

    mix_meta = None
    preprocess_meta = None
    asr_speaker_tracks = speaker_tracks
    if speaker_tracks and not args.no_asr_preprocess:
        preprocess_dir = output.parent / "preprocessed_asr"
        logger.event("ASR用音声前処理開始", {"output_dir": str(preprocess_dir), "progress_percent": 2.6})
        asr_speaker_tracks, preprocess_meta = preprocess_speaker_tracks_for_asr(
            speaker_tracks,
            preprocess_dir,
            progress=logger.event,
            progress_base=2.6,
            progress_span=0.4,
        )
        logger.event(
            "ASR用音声前処理完了",
            {
                "output_dir": str(preprocess_dir),
                "reused": preprocess_meta.get("reused", False),
                "tracks": len(asr_speaker_tracks),
                "progress_percent": 3.0,
            },
        )

    if args.mix_from_speaker_tracks:
        if not speaker_tracks:
            print("--mix-from-speaker-tracks には --speaker-track が必要です", file=sys.stderr)
            return 1
        mix_output = Path(args.mix_output) if args.mix_output else output.with_name(output.stem + "_mix.wav")
        logger.event("話者別トラックからmix作成開始", {"mix_output": str(mix_output), "progress_percent": 3.0})
        mix_meta = make_mix_from_speaker_tracks(
            asr_speaker_tracks,
            mix_output,
            progress=logger.event,
            progress_base=3.0,
            progress_span=5.0,
        )
        audio = Path(mix_meta["path"])
        logger.event(
            "話者別トラックからmix作成完了",
            {
                "mix_audio": str(audio),
                "duration_sec": round(float(mix_meta.get("duration", 0.0)), 3),
                "sample_rate": mix_meta.get("sample_rate"),
                "reused": mix_meta.get("reused", False),
                "progress_percent": 8.0,
            },
        )
        print(f"[transcribe] mix {'reused' if mix_meta.get('reused') else 'saved'}: {audio}")

    if audio is None:
        print("内部エラー: mix音声パスが解決されませんでした", file=sys.stderr)
        logger.close("failed")
        return 1

    _track_segs_for_simple: dict = {}
    if args.reuse_existing and output.exists():
        logger.event("既存JSON再利用", {"output": str(output)})
        print(f"[transcribe] reuse existing: {output}")
        segments = load_transcript(output)
        logger.event("既存JSON読み込み完了", {"segments": len(segments)})
    elif speaker_tracks:
        if args.speaker_track_mode == "fusion":
            logger.event("高精度fusion開始", {"mode": "fusion", "fusion_version": args.fusion_version, "progress_percent": 9.0})
            print(f"[transcribe] speaker-track fusion mode {args.fusion_version} (mix + speaker tracks + RMS/VAD)")
            segments, _track_segs_for_simple = transcribe_with_speaker_track_fusion(
                audio,
                output,
                speaker_tracks=asr_speaker_tracks,
                diarization_speaker_tracks=speaker_tracks,
                model_size=args.whisper_model,
                language=args.language,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                active_db=args.speaker_track_active_db,
                overlap_db=args.speaker_track_overlap_db,
                margin=args.speaker_track_margin,
                progress=logger.event,
                fusion_version=args.fusion_version,
                enable_crosstalk_cancellation=bool(args.crosstalk_cancel),
                extra_meta={
                    **({"speaker_track_mix": mix_meta} if mix_meta else {}),
                    **({"speaker_track_preprocess": preprocess_meta} if preprocess_meta else {}),
                    **({"input_alignment": alignment_meta} if alignment_meta else {}),
                },
            )
            logger.event("高精度fusion完了", {"segments": len(segments), "progress_percent": 97.0})
        else:
            logger.event("話者別RMSモード開始", {"mode": "rms"})
            print("[transcribe] speaker-track RMS mode (faster-whisper + RMS)")
            segments = transcribe_with_speaker_tracks(
                audio,
                output,
                speaker_tracks=speaker_tracks,
                model_size=args.whisper_model,
                language=args.language,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                active_db=args.speaker_track_active_db,
                overlap_db=args.speaker_track_overlap_db,
                margin=args.speaker_track_margin,
                extra_meta={
                    **({"speaker_track_mix": mix_meta} if mix_meta else {}),
                    **({"speaker_track_preprocess": preprocess_meta} if preprocess_meta else {}),
                    **({"input_alignment": alignment_meta} if alignment_meta else {}),
                },
            )
            logger.event("話者別RMSモード完了", {"segments": len(segments)})
        print(f"[transcribe] saved: {output} ({len(segments)} segments, speaker-track)")
    elif args.diarize:
        logger.event("pyannote話者識別開始")
        print(f"[transcribe] diarized mode (faster-whisper + pyannote.audio)")
        segments = transcribe_with_diarization(
            audio, output,
            model_size=args.whisper_model, language=args.language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            hf_token=args.hf_token,
            min_speakers=args.min_speakers, max_speakers=args.max_speakers,
        )
        logger.event("pyannote話者識別完了", {"segments": len(segments)})
        print(f"[transcribe] saved: {output} ({len(segments)} segments, with speaker)")
    elif args.transcriber == "whisperx":
        if hotwords:
            print("[transcribe] warning: --hotwords は whisperx では未使用です", file=sys.stderr)
        logger.event("WhisperX文字起こし開始")
        segments = transcribe_with_whisperx(
            audio, output,
            model_size=args.whisper_model, language=args.language,
            initial_prompt=initial_prompt,
        )
        logger.event("WhisperX文字起こし完了", {"segments": len(segments)})
        print(f"[transcribe] saved: {output} ({len(segments)} segments)")
    else:
        logger.event("faster-whisper文字起こし開始")
        segments = transcribe_with_faster_whisper(
            audio, output,
            model_size=args.whisper_model, language=args.language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
        logger.event("faster-whisper文字起こし完了", {"segments": len(segments)})
        print(f"[transcribe] saved: {output} ({len(segments)} segments)")

    if args.text_output:
        logger.event("文字起こしテキスト出力開始", {"text_output": str(args.text_output), "progress_percent": 98.5})
        paths = write_transcript_text_outputs(segments, args.text_output, _track_segs_for_simple or None)
        logger.event(
            "文字起こしテキスト出力完了",
            {
                "review": str(paths["review"]),
                "clean": str(paths["clean"]),
                "simple": str(paths["simple"]),
                "quality_summary": str(paths["quality_md"]),
                "progress_percent": 99.0,
            },
        )
        print(f"[transcribe] review saved: {paths['review']}")
        print(f"[transcribe] clean saved: {paths['clean']}")
        print(f"[transcribe] simple saved: {paths['simple']}")
        print(f"[transcribe] quality summary saved: {paths['quality_md']}")

    logger.event("全処理完了", {"output": str(output), "segments": len(segments), "progress_percent": 100.0})
    logger.close("done")
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    with prevent_sleep():
        exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # faster-whisper/ctranslate2(CUDA) のネイティブ終了処理が Windows で
    # 0xC0000409(3221226505) を出すことがあるため、出力保存後は明示終了する。
    os._exit(int(exit_code))
