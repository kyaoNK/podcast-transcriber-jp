"""GUIの話者別トラック・バッチを1プロセスで実行する。

モデルロードとメモリ上のASRキャッシュをジョブ間で再利用し、精度設定は単体実行と同じに保つ。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.common.sleep_guard import prevent_sleep
from apps.transcribe.transcribe import ProcessingLogger, write_transcript_text_outputs
from core.review.transcribe_audio import (
    make_mix_from_speaker_tracks,
    preprocess_speaker_tracks_for_asr,
    transcribe_with_speaker_track_fusion,
    validate_audio_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUIジョブキューを同一プロセスで順番に文字起こしする。")
    parser.add_argument("--job-queue", required=True, help="GUIが保存した jobs JSON")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _parse_speaker_tracks(values: list[object]) -> list[tuple[str, Path]]:
    tracks: list[tuple[str, Path]] = []
    for i, raw in enumerate(values, start=1):
        value = str(raw).strip().strip('"')
        if not value:
            continue
        if "=" in value:
            label, path = value.split("=", 1)
            label = label.strip() or f"TRACK_{i}"
        else:
            label, path = f"TRACK_{i}", value
        tracks.append((label, Path(path.strip().strip('"'))))
    return tracks


def _safe_output_base(job: dict[str, Any], speaker_tracks: list[tuple[str, Path]]) -> Path:
    out_dir = str(job.get("output_dir") or "").strip()
    if not out_dir:
        out_dir = str(speaker_tracks[0][1].parent)
    out_name = str(job.get("output_name") or "transcript_speaker_tracks").strip() or "transcript_speaker_tracks"
    for ch in '<>:"/\\|?*':
        out_name = out_name.replace(ch, "_")
    return Path(out_dir) / out_name


def _read_text_or_file(job: dict[str, Any], text_key: str, file_key: str) -> str | None:
    text = str(job.get(text_key) or "").strip()
    if text:
        return text
    file_path = str(job.get(file_key) or "").strip()
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return None


def _run_one(job: dict[str, Any], index: int, total: int) -> int:
    name = str(job.get("job_name") or job.get("output_name") or f"job_{index}")
    print(f"[gui] batch job {index}/{total}: {name}", flush=True)

    speaker_tracks = _parse_speaker_tracks(list(job.get("speaker_tracks") or []))
    if len(speaker_tracks) < 2:
        raise ValueError(f"{name}: 話者別トラックを2本以上指定してください")

    audio: Path | None = None

    base = _safe_output_base(job, speaker_tracks)
    output = base.with_suffix(".json")
    text_output = base.with_suffix(".md")
    log_txt = output.with_suffix(".log.txt")
    log_md = output.with_suffix(".log.md")
    logger = ProcessingLogger(log_txt, log_md)

    mix_meta: dict[str, Any] | None = None
    preprocess_meta: dict[str, Any] | None = None
    asr_speaker_tracks = speaker_tracks
    alignment_meta: dict[str, Any] | None = None
    segments: Any = None
    track_segments: Any = None
    try:
        alignment_meta = validate_audio_alignment(
            None,
            speaker_tracks,
            tolerance_sec=2.0,
            allow_mismatch=bool(job.get("allow_duration_mismatch", False)),
        )
        logger.event(
            "入力音声チェック完了",
            {
                "ok": alignment_meta.get("ok"),
                "warnings": alignment_meta.get("warnings"),
                "errors_allowed": alignment_meta.get("allowed_mismatch", False),
                "progress_percent": 2.5,
            },
        )

        if not bool(job.get("no_asr_preprocess", False)):
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

        mix_output = output.with_name(output.stem + "_mix.wav")
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
                "reused": mix_meta.get("reused", False),
                "progress_percent": 8.0,
            },
        )

        initial_prompt = _read_text_or_file(job, "initial_prompt", "prompt_file")
        hotwords = _read_text_or_file(job, "hotwords", "hotwords_file")

        fusion_version = "v2" if str(job.get("fusion_version", "v1")).lower() == "v2" else "v1"
        logger.event("高精度fusion開始", {"mode": "fusion", "fusion_version": fusion_version, "progress_percent": 9.0})
        segments, track_segments = transcribe_with_speaker_track_fusion(
            audio,
            output,
            speaker_tracks=asr_speaker_tracks,
            diarization_speaker_tracks=speaker_tracks,
            model_size=str(job.get("model") or "large-v3"),
            language=str(job.get("language") or "ja"),
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            active_db=str(job.get("speaker_track_active_db") or "auto"),
            overlap_db=str(job.get("speaker_track_overlap_db") or "auto"),
            margin=str(job.get("speaker_track_margin") or "auto"),
            progress=logger.event,
            fusion_version=fusion_version,
            enable_crosstalk_cancellation=bool(job.get("crosstalk_cancel", True)),
            extra_meta={
                **({"speaker_track_mix": mix_meta} if mix_meta else {}),
                **({"speaker_track_preprocess": preprocess_meta} if preprocess_meta else {}),
                **({"input_alignment": alignment_meta} if alignment_meta else {}),
                # ジョブJSON全文を残す。再実行や設定確認の根拠になる。
                "job": json.loads(json.dumps(job, ensure_ascii=False, default=str)),
            },
        )
        logger.event("高精度fusion完了", {"segments": len(segments), "progress_percent": 97.0})

        paths = write_transcript_text_outputs(segments, text_output, track_segments)
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

        logger.event("全処理完了", {"output": str(output), "segments": len(segments), "progress_percent": 100.0})
        logger.close("done")
        print(f"[gui] batch job done {index}/{total}: {name}", flush=True)
        return 0
    except Exception:
        logger.close("failed")
        raise
    finally:
        # ジョブ境界でメモリ/CUDAキャッシュを解放（計算結果には影響しない）
        try:
            del segments
        except Exception:
            pass
        try:
            del track_segments
        except Exception:
            pass
        try:
            del mix_meta
        except Exception:
            pass
        try:
            del preprocess_meta
        except Exception:
            pass
        try:
            del alignment_meta
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.job_queue).read_text(encoding="utf-8-sig"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(jobs, list) or not jobs:
        print("[gui] failed: job queue is empty", file=sys.stderr, flush=True)
        return 1

    failures = 0
    total = len(jobs)
    for index, raw_job in enumerate(jobs, start=1):
        if not isinstance(raw_job, dict):
            failures += 1
            print(f"[gui] batch job failed {index}/{total}: invalid job", flush=True)
            continue
        name = str(raw_job.get("job_name") or raw_job.get("output_name") or f"job_{index}")
        try:
            _run_one(raw_job, index, total)
            print(f"[gui] batch job done {index}/{total}: {name}", flush=True)
        except Exception as e:
            failures += 1
            print(f"[gui] batch job failed {index}/{total}: {name} {e}", flush=True)
            traceback.print_exc()
            if args.stop_on_error:
                break

    if failures:
        print(f"[gui] batch finished with failures: {failures}/{total}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    with prevent_sleep():
        exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Windows + CUDA/ctranslate2 の終了時クラッシュを避ける。
    os._exit(int(exit_code))
