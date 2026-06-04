"""既存の文字起こしJSONに対して、怪しい区間または指定範囲だけASR retryする。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.common.sleep_guard import prevent_sleep
from apps.transcribe.transcribe import ProcessingLogger, write_transcript_text_outputs
from core.review.transcribe_audio import retry_transcript_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="既存JSONの一部だけASR retryする。")
    parser.add_argument("--json", required=True, help="既存の文字起こしJSON")
    parser.add_argument("--audio", required=True, help="retryに使うASR対象音声/mix音声")
    parser.add_argument("--output", help="出力JSON。省略時は *.manual_retry.json")
    parser.add_argument("--text-output", help="review/clean/simple等の出力ベース")
    parser.add_argument("--start", type=float, help="retry範囲の開始秒。省略時は自動検出")
    parser.add_argument("--end", type=float, help="retry範囲の終了秒。省略時は自動検出")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--force-replace", action="store_true", help="候補に重大な異常がなければ差し替えやすくする")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = Path(args.json)
    output = Path(args.output) if args.output else json_path.with_name(json_path.stem + ".manual_retry.json")
    log_txt = output.with_suffix(".log.txt")
    log_md = output.with_suffix(".log.md")
    logger = ProcessingLogger(log_txt, log_md)
    try:
        logger.event(
            "手動retry準備完了",
            {
                "json": str(json_path),
                "audio": str(args.audio),
                "output": str(output),
                "start": args.start,
                "end": args.end,
                "progress_percent": 1.0,
            },
        )
        segments, meta = retry_transcript_segments(
            args.audio,
            json_path,
            output,
            start_sec=args.start,
            end_sec=args.end,
            model_size=args.whisper_model,
            language=args.language,
            force_replace=bool(args.force_replace),
            progress=logger.event,
        )
        text_output = Path(args.text_output) if args.text_output else output.with_suffix(".md")
        paths = write_transcript_text_outputs(segments, text_output, None)
        logger.event(
            "手動retryテキスト出力完了",
            {
                "review": str(paths["review"]),
                "clean": str(paths["clean"]),
                "simple": str(paths["simple"]),
                "quality_summary": str(paths["quality_md"]),
                "attempted": meta.get("attempted"),
                "replaced": meta.get("replaced"),
                "progress_percent": 99.0,
            },
        )
        logger.event("全処理完了", {"output": str(output), "segments": len(segments), "progress_percent": 100.0})
        logger.close("done")
        print(f"[retry] saved: {output}", flush=True)
        return 0
    except Exception as e:
        logger.event("手動retry失敗", {"error": str(e), "progress_percent": 100.0})
        logger.close("failed")
        print(f"[retry] failed: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    with prevent_sleep():
        code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code))
