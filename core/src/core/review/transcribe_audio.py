from __future__ import annotations

import bisect
import collections
import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.common.timecode import format_hhmmss_ms


_FASTER_WHISPER_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_PYANNOTE_PIPELINE_CACHE: dict[str, Any] = {}
_SUDACHI_TOKENIZER_CACHE: Any | None = None
_SUDACHI_SPLIT_MODE: Any | None = None
_SUDACHI_AVAILABLE: bool | None = None
ProgressCallback = Callable[[str, dict[str, Any] | None], None]
SPEAKER_TRACK_ALLOWED_SUFFIXES = {".wav"}
ASR_CHUNK_SECONDS = 8 * 60.0
ASR_CHUNK_OVERLAP_SECONDS = 2.0
ASR_CHUNK_MIN_DURATION_SECONDS = 12 * 60.0
ASR_RETRY_CONTEXT_SECONDS = 0.75
ASR_RETRY_MAX_SEGMENTS = 80
ASR_RETRY_MIN_WINDOW_SECONDS = 1.5
ASR_RETRY_MAX_WINDOW_SECONDS = 18.0
# faster-whisper の内部Silero VAD パラメータ。
# 過去にmin_silence_duration_ms=500で試したが、Whisperが文の途中で切ってしまい
# 文頭欠落・重複セグメント・外国語誤認識が頻発したため、デフォルト相当に戻している。
# Whisperは1〜2秒程度のコンテキストがあったほうが日本語認識の精度が上がる。
ASR_VAD_PARAMETERS = {
    "min_silence_duration_ms": 2000,
    "speech_pad_ms": 400,
    "min_speech_duration_ms": 250,
    "threshold": 0.5,
}
# retry時のtemperature sweep（複数温度で試行し最良logprobを採用）
ASR_RETRY_TEMPERATURES = (0.0, 0.2, 0.4)


def _progress(progress: ProgressCallback | None, message: str, **data: Any) -> None:
    if progress is not None:
        progress(message, data or None)


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """同一ディレクトリの一時ファイルに書いてから置換する。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(target)


def atomic_write_json(path: str | Path, payload: Any, encoding: str = "utf-8") -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), encoding=encoding)


@dataclass
class TranscriptSegment:
    segment_id: str
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)  # {start, end, word, speaker?}
    primary_speaker: str | None = None  # word の多数決で決まる主たる話者
    speaker_scores: dict[str, float] = field(default_factory=dict)  # 話者別RMS dBなど
    overlap_speakers: list[str] = field(default_factory=list)
    is_overlap: bool = False
    confidence: str | None = None
    confidence_reasons: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    asr: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


@dataclass
class DiarizationResult:
    intervals: list[tuple[float, float, str]]
    segment_speakers: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def transcript_to_text_lines(segments: list[TranscriptSegment]) -> str:
    """話者ターンごとにまとめた会話形式テキストを返す。編集者の閲覧用。

    同じ話者が続く segment はひとつのターンにまとめ、話者が変わったときだけ
    ヘッダ行を出力する。

    例:
        [00:00:00.000 - 00:00:05.234] SPEAKER_00
          パスワードの時代は終わりつつあります。
          生体認証が主流になってきています。

        [00:00:05.500 - 00:00:08.100] SPEAKER_01
          確かにパスキーが普及してきていますね。
    """
    segments = [seg for seg in segments if seg.text.strip() and not seg.asr.get("suppress_in_text_outputs")]
    if not segments:
        return ""

    lines: list[str] = []
    turn_start: float = segments[0].start
    turn_end: float = segments[0].end
    turn_speaker: str | None = segments[0].primary_speaker
    turn_texts: list[str] = [segments[0].text.strip()]

    def flush_turn() -> None:
        sp_label = turn_speaker or "不明"
        header = f"[{format_hhmmss_ms(turn_start)} - {format_hhmmss_ms(turn_end)}] {sp_label}"
        body = "  " + "\n  ".join(t for t in turn_texts if t)
        lines.append(header)
        lines.append(body)
        lines.append("")

    for seg in segments[1:]:
        if seg.primary_speaker == turn_speaker:
            turn_end = seg.end
            turn_texts.append(seg.text.strip())
        else:
            flush_turn()
            turn_start = seg.start
            turn_end = seg.end
            turn_speaker = seg.primary_speaker
            turn_texts = [seg.text.strip()]

    flush_turn()
    return "\n".join(lines)


def _format_mmss(seconds: float) -> str:
    """秒を `H:MM:SS` 形式に変換（LLM入力用・秒精度）。"""
    total = max(0, int(seconds))
    ss = total % 60
    total_minutes = total // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    if hh:
        return f"{hh}:{mm:02d}:{ss:02d}"
    return f"{mm}:{ss:02d}"


def transcript_to_simple_lines(segments: list[TranscriptSegment]) -> str:
    """1セグメント1行の簡易フォーマットを返す。LLM入力向け。

    形式: [M:SS] 話者名 | テキスト
    """
    lines: list[str] = []
    for seg in segments:
        if seg.asr.get("suppress_in_text_outputs"):
            continue
        text = seg.text.strip()
        if not text:
            continue
        if seg.primary_speaker == "OVERLAP" and seg.overlap_speakers:
            speaker = "+".join(seg.overlap_speakers)
        else:
            speaker = seg.primary_speaker or "不明"
        lines.append(f"[{_format_mmss(seg.start)}] {speaker} | {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def make_simple_segments(
    fused: list[TranscriptSegment],
    speaker_track_segments: dict[str, list[TranscriptSegment]],
) -> list[TranscriptSegment]:
    """simple.txt 用に話者トラックのセグメント境界を使った出力リストを生成する。

    fused（mix境界）ではなく、各話者トラックのWhisperセグメントを出力単位とする。
    RMS/VAD判定（fused）で話者が確定しているセグメントに紐付け、
    OVERLAP / 不明は fused の mix 境界をそのまま使う。
    """
    output: list[TranscriptSegment] = []
    used: set[tuple[str, float, float]] = set()

    # OVERLAP / 不明 を fused からそのまま追加
    for seg in fused:
        if (seg.primary_speaker == "OVERLAP" or seg.primary_speaker is None) and not seg.asr.get("suppress_in_text_outputs"):
            output.append(seg)

    # bisect 用に fused の開始時刻リストを事前構築（O(m)）
    fused_starts = [seg.start for seg in fused]

    # 各話者トラックセグメントを話者ラベルで判定して追加
    for speaker, track_segs in speaker_track_segments.items():
        for ts in track_segs:
            text = ts.text.strip()
            if not text:
                continue
            # bisect で重複する fused セグメントのみ走査: O(log m + k)
            lo = max(0, bisect.bisect_right(fused_starts, ts.start) - 1)
            durations: dict[str, float] = {}
            for i in range(lo, len(fused)):
                seg = fused[i]
                if seg.start >= ts.end:
                    break
                ov = _overlap_seconds(seg.start, seg.end, ts.start, ts.end)
                if ov <= 0:
                    continue
                sp = seg.primary_speaker or "不明"
                durations[sp] = durations.get(sp, 0.0) + ov
            if not durations:
                continue
            dom = max(durations, key=lambda k: durations[k])
            if dom != speaker:
                continue
            key = (speaker, round(ts.start, 3), round(ts.end, 3))
            if key in used:
                continue
            used.add(key)
            new_seg = TranscriptSegment(
                segment_id=f"simple_{speaker}_{ts.start:.3f}",
                start=ts.start,
                end=ts.end,
                text=text,
                primary_speaker=speaker,
                source="speaker_track",
            )
            output.append(new_seg)

    output.sort(key=lambda s: s.start)
    return output


def transcript_to_review_markdown(segments: list[TranscriptSegment]) -> str:
    """候補・低信頼理由を含む、人間確認用Markdownを返す。"""
    lines: list[str] = [
        "# 文字起こしレビュー",
        "",
        "- `OK`: そのまま使えそうな区間",
        "- `要確認`: overlap、候補不一致、小声などで人が見るべき区間",
        "",
    ]
    if not segments:
        return "\n".join(lines)

    def needs_review(seg: TranscriptSegment) -> bool:
        return bool(seg.is_overlap or seg.confidence in {"low", "medium"} or seg.confidence_reasons)

    def can_merge(prev: TranscriptSegment, cur: TranscriptSegment) -> bool:
        if (prev.primary_speaker or "不明") != (cur.primary_speaker or "不明"):
            return False
        if needs_review(prev) or needs_review(cur):
            return False
        if cur.start - prev.end > 1.2:
            return False
        return True

    chunks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    for seg in segments:
        if not current or can_merge(current[-1], seg):
            current.append(seg)
        else:
            chunks.append(current)
            current = [seg]
    if current:
        chunks.append(current)

    for index, chunk in enumerate(chunks, start=1):
        start = min(seg.start for seg in chunk)
        end = max(seg.end for seg in chunk)
        speaker = chunk[0].primary_speaker or "不明"
        review = any(needs_review(seg) for seg in chunk)
        status = "要確認" if review else "OK"
        confidence_values = [seg.confidence for seg in chunk if seg.confidence]
        confidence = min(confidence_values, key=lambda v: {"low": 0, "medium": 1, "high": 2}.get(v, 1), default="unknown")

        lines.append(f"## {index}. {format_hhmmss_ms(start)} - {format_hhmmss_ms(end)}｜{speaker}｜{status}")
        lines.append("")
        if review:
            reasons: list[str] = []
            for seg in chunk:
                reasons.extend(seg.confidence_reasons)
                if seg.is_overlap and "overlap" not in reasons:
                    reasons.append("overlap")
            reason_text = ", ".join(dict.fromkeys(reasons)) if reasons else "不明"
            lines.append(f"> **要確認**: 信頼度 {confidence} / 理由: {reason_text}")
            lines.append("")

        for seg in chunk:
            text = seg.text.strip() or "(textなし)"
            lines.append(f"> {text}")
        lines.append("")

        review_segments = [seg for seg in chunk if needs_review(seg)]
        if review_segments:
            for seg in review_segments:
                if seg.is_overlap and seg.overlap_speakers:
                    lines.append(f"- 重なり候補: {', '.join(seg.overlap_speakers)}")
                if seg.speaker_scores:
                    scores = ", ".join(f"{k}: {v:.1f}dB" for k, v in seg.speaker_scores.items())
                    lines.append(f"- RMS: {scores}")
                if seg.candidates:
                    sorted_candidates = sorted(
                        seg.candidates,
                        key=lambda c: float(c.get("score", -999.0)) if isinstance(c.get("score"), (int, float)) else -999.0,
                        reverse=True,
                    )
                    adopted_norm = _normalize_candidate_text(seg.text)
                    grouped: dict[str, dict[str, Any]] = {}
                    for cand in sorted_candidates:
                        text = str(cand.get("text", "")).strip()
                        if not text:
                            continue
                        key = _normalize_candidate_text(text)
                        group = grouped.setdefault(
                            key,
                            {"text": text, "sources": [], "best_score": None, "start": cand.get("start"), "end": cand.get("end")},
                        )
                        label = cand.get("source", "candidate")
                        cand_speaker = cand.get("speaker")
                        source_label = f"{label}/{cand_speaker}" if cand_speaker else str(label)
                        if source_label not in group["sources"]:
                            group["sources"].append(source_label)
                        score = cand.get("score")
                        if isinstance(score, (int, float)) and (
                            group["best_score"] is None or float(score) > float(group["best_score"])
                        ):
                            group["best_score"] = float(score)
                            group["start"] = cand.get("start")
                            group["end"] = cand.get("end")

                    adopted_group = grouped.get(adopted_norm)
                    if adopted_group:
                        score = adopted_group.get("best_score")
                        score_text = f" / score {score:.2f}" if isinstance(score, (int, float)) else ""
                        lines.append(f"- 採用根拠: {', '.join(adopted_group['sources'])}{score_text}")

                    alternatives = [g for key, g in grouped.items() if key != adopted_norm]
                    alternatives.sort(
                        key=lambda g: float(g.get("best_score", -999.0)) if isinstance(g.get("best_score"), (int, float)) else -999.0,
                        reverse=True,
                    )
                    if alternatives:
                        lines.append("- 別候補:")
                        for group in alternatives:
                            score = group.get("best_score")
                            score_text = f" / score {score:.2f}" if isinstance(score, (int, float)) else ""
                            time = ""
                            if group.get("start") is not None and group.get("end") is not None:
                                time = f"（{format_hhmmss_ms(float(group['start']))} - {format_hhmmss_ms(float(group['end']))}）"
                            lines.append(
                                f"  - **{', '.join(group['sources'])}**{time}{score_text}: {group['text']}"
                            )
            lines.append("")
    return "\n".join(lines)


def load_transcript(path: str | Path) -> list[TranscriptSegment]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("segments", [])
    segments: list[TranscriptSegment] = []
    for i, item in enumerate(data):
        segments.append(TranscriptSegment(
            segment_id=str(item.get("segment_id", item.get("id", i))),
            start=float(item["start"]),
            end=float(item["end"]),
            text=str(item.get("text", "")).strip(),
            words=list(item.get("words") or []),
            primary_speaker=item.get("primary_speaker"),
            speaker_scores={str(k): float(v) for k, v in (item.get("speaker_scores") or {}).items()},
            overlap_speakers=[str(v) for v in (item.get("overlap_speakers") or [])],
            is_overlap=bool(item.get("is_overlap", False)),
            confidence=item.get("confidence"),
            confidence_reasons=[str(v) for v in (item.get("confidence_reasons") or [])],
            candidates=list(item.get("candidates") or []),
            asr=dict(item.get("asr") or {}),
            source=item.get("source"),
        ))
    return segments


def save_transcript(segments: list[TranscriptSegment], path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"segments": [asdict(s) for s in segments]}
    if meta:
        payload["meta"] = meta
    atomic_write_json(path, payload, encoding="utf-8")


def _load_transcript_with_meta(path: str | Path) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {}
    if isinstance(payload, dict):
        meta = dict(payload.get("meta") or {})
    return load_transcript(path), meta


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    st = path.stat()
    return {
        "path": str(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_dir(cache_dir: str | Path | None, kind: str) -> Path | None:
    if cache_dir is None:
        return None
    path = Path(cache_dir) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _segments_fingerprint(segments: list[TranscriptSegment]) -> str:
    return _stable_hash(
        [
            {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg.text,
                "words": [
                    {
                        "start": round(float(w.get("start", 0.0)), 3),
                        "end": round(float(w.get("end", 0.0)), 3),
                        "word": str(w.get("word", "")),
                    }
                    for w in seg.words
                ],
            }
            for seg in segments
        ]
    )


def _device_auto() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def detect_compute_backend() -> dict[str, Any]:
    """文字起こし/話者識別で使える実行環境を軽量に調べる。

    faster-whisper は CTranslate2、pyannote は PyTorch を使うため、
    CUDA判定を分けて表示する。
    """
    info: dict[str, Any] = {
        "faster_whisper_device": "cpu",
        "faster_whisper_compute_type": "auto",
        "faster_whisper_cuda_devices": 0,
        "faster_whisper_cuda_compute_types": [],
        "pyannote_device": "cpu",
        "torch_cuda_available": False,
        "torch_cuda_version": None,
        "gpu_name": None,
    }

    try:
        import ctranslate2  # type: ignore

        cuda_devices = int(ctranslate2.get_cuda_device_count())
        info["faster_whisper_cuda_devices"] = cuda_devices
        if cuda_devices > 0:
            info["faster_whisper_device"] = "cuda"
            try:
                info["faster_whisper_cuda_compute_types"] = sorted(
                    str(x) for x in ctranslate2.get_supported_compute_types("cuda")
                )
            except Exception as e:
                info["faster_whisper_cuda_compute_types_error"] = str(e)
        else:
            try:
                info["faster_whisper_cpu_compute_types"] = sorted(
                    str(x) for x in ctranslate2.get_supported_compute_types("cpu")
                )
            except Exception as e:
                info["faster_whisper_cpu_compute_types_error"] = str(e)
    except Exception as e:
        info["faster_whisper_backend_error"] = str(e)

    try:
        import torch  # type: ignore

        torch_cuda = bool(torch.cuda.is_available())
        info["torch_cuda_available"] = torch_cuda
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch_cuda:
            info["pyannote_device"] = "cuda"
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception as e:
        info["torch_backend_error"] = str(e)

    return info


def format_compute_backend(info: dict[str, Any] | None = None) -> str:
    info = info or detect_compute_backend()
    whisper = "GPU(cuda)" if info.get("faster_whisper_device") == "cuda" else "CPU"
    parts = [f"Whisper: {whisper}"]
    gpu_name = info.get("gpu_name")
    if gpu_name:
        parts.append(str(gpu_name))
    return " / ".join(parts)


def _get_faster_whisper_model(model_size: str) -> Any:
    """同一プロセスの連続処理で faster-whisper モデルを再利用する。"""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper が未インストールです。対応: pip install faster-whisper"
        ) from e

    key = (model_size, "auto", "auto")
    model = _FASTER_WHISPER_MODEL_CACHE.get(key)
    if model is None:
        model = WhisperModel(model_size, device="auto", compute_type="auto")
        _FASTER_WHISPER_MODEL_CACHE[key] = model
    return model


def _audio_duration_seconds(audio_path: Path) -> float:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    except Exception:
        return 0.0


def _write_audio_chunk(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    block_frames: int = 48000 * 30,
) -> dict[str, Any]:
    import soundfile as sf  # type: ignore

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(input_path), mode="r") as src:
        sample_rate = int(src.samplerate)
        channels = int(src.channels)
        start_frame = max(0, int(round(start_sec * sample_rate)))
        end_frame = min(int(src.frames), int(round(end_sec * sample_rate)))
        frames = max(0, end_frame - start_frame)
        src.seek(start_frame)
        with sf.SoundFile(str(output_path), mode="w", samplerate=sample_rate, channels=channels, subtype="PCM_16") as out:
            remaining = frames
            while remaining > 0:
                n = min(int(block_frames), remaining)
                data = src.read(n, dtype="float32", always_2d=True)
                if data.size == 0:
                    break
                out.write(data)
                remaining -= len(data)
    return {
        "path": str(output_path),
        "start_sec": round(float(start_sec), 3),
        "end_sec": round(float(end_sec), 3),
        "frames": frames,
        "sample_rate": sample_rate,
        "channels": channels,
    }


def _faster_whisper_transcribe_only(
    audio_path: str | Path,
    model_size: str = "large-v3",
    language: str = "ja",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    progress: ProgressCallback | None = None,
    progress_label: str = "Whisper",
    progress_base: float | None = None,
    progress_span: float | None = None,
    cache_dir: str | Path | None = None,
    chunk_seconds: float = ASR_CHUNK_SECONDS,
    chunk_overlap_seconds: float = ASR_CHUNK_OVERLAP_SECONDS,
    enable_retry: bool = True,
) -> list[TranscriptSegment]:
    """faster-whisper を呼び出して segments を返すだけ (保存はしない)。

    長尺音声は固定長chunkへ分けて独立ASRし、時刻を絶対時刻へ戻す。
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    def emit_progress(ratio: float, *, current_sec: float | None = None, duration_sec: float | None = None) -> None:
        ratio = max(0.0, min(1.0, ratio))
        data: dict[str, Any] = {
            "stage": progress_label,
            "stage_percent": round(ratio * 100.0, 1),
        }
        if progress_base is not None and progress_span is not None:
            data["progress_percent"] = round(progress_base + progress_span * ratio, 1)
        if current_sec is not None:
            data["current_sec"] = round(float(current_sec), 2)
        if duration_sec:
            data["duration_sec"] = round(float(duration_sec), 2)
        _progress(progress, f"{progress_label} 文字起こし進捗", **data)

    cache_root = _cache_dir(cache_dir, "asr")
    audio_duration = _audio_duration_seconds(audio_path)
    use_chunked_asr = bool(cache_root is not None and audio_duration >= ASR_CHUNK_MIN_DURATION_SECONDS and chunk_seconds > 0.0)
    cache_path: Path | None = None
    cache_key = ""
    if cache_root is not None:
        cache_key = _stable_hash(
            {
                "kind": "faster_whisper_segments_v5",
                "audio": _file_fingerprint(audio_path),
                "model_size": model_size,
                "language": language,
                "initial_prompt": initial_prompt or "",
                "hotwords": hotwords or "",
                "vad_filter": True,
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "chunked_asr": use_chunked_asr,
                "chunk_seconds": chunk_seconds if use_chunked_asr else None,
                "chunk_overlap_seconds": chunk_overlap_seconds if use_chunked_asr else None,
                "retry": bool(enable_retry),
                "retry_context_seconds": ASR_RETRY_CONTEXT_SECONDS if enable_retry else None,
                "retry_max_segments": ASR_RETRY_MAX_SEGMENTS if enable_retry else None,
            }
        )
        cache_path = cache_root / f"{cache_key}.json"
        if cache_path.exists():
            try:
                segments, meta = _load_transcript_with_meta(cache_path)
                if meta.get("cache_key") == cache_key:
                    emit_progress(1.0)
                    _progress(
                        progress,
                        f"{progress_label} 文字起こしキャッシュ利用",
                        cache=str(cache_path),
                        segments=len(segments),
                        progress_percent=round((progress_base or 0.0) + (progress_span or 0.0), 1)
                        if progress_base is not None and progress_span is not None
                        else None,
                    )
                    return segments
            except Exception as e:
                _progress(progress, f"{progress_label} 文字起こしキャッシュ読込失敗", cache=str(cache_path), error=str(e))

    model = _get_faster_whisper_model(model_size)
    duration_f = audio_duration
    last_reported = -0.05
    emit_progress(0.0, current_sec=0.0, duration_sec=duration_f or None)

    def collect_segments(
        transcribe_path: Path,
        *,
        offset_sec: float = 0.0,
        core_start: float | None = None,
        core_end: float | None = None,
        segment_id_prefix: str = "",
        initial_prompt_override: str | None = initial_prompt,
        hotwords_override: str | None = hotwords,
        vad_filter: bool = True,
        retry_decode: bool = False,
        temperature_override: float | None = None,
    ) -> list[TranscriptSegment]:
        decode_kwargs: dict[str, Any] = {}
        if retry_decode:
            decode_kwargs.update(
                {
                    "temperature": 0.0,
                    "repetition_penalty": 1.05,
                    "no_repeat_ngram_size": 3,
                }
            )
        if temperature_override is not None:
            decode_kwargs["temperature"] = float(temperature_override)
        vad_kwargs: dict[str, Any] = {}
        if vad_filter:
            vad_kwargs["vad_parameters"] = dict(ASR_VAD_PARAMETERS)
        raw_segments, _info = model.transcribe(
            str(transcribe_path),
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt_override,
            hotwords=hotwords_override,
            # 無音区間後の幻覚検出時に無音をスキップして再処理
            hallucination_silence_threshold=2.0,
            # 無音判定を積極化（既定0.6→0.4）。Whisperが無音を「喋っている」と誤判定するのを防ぐ
            no_speech_threshold=0.4,
            **vad_kwargs,
            **decode_kwargs,
        )
        out: list[TranscriptSegment] = []
        for i, s in enumerate(raw_segments):
            seg_start = float(s.start) + offset_sec
            seg_end = float(s.end) + offset_sec
            center = (seg_start + seg_end) / 2.0
            if core_start is not None and center < core_start:
                continue
            if core_end is not None and center >= core_end:
                continue
            words = [
                {"start": float(w.start) + offset_sec, "end": float(w.end) + offset_sec, "word": w.word.strip()}
                for w in (s.words or [])
                if w.start is not None and w.end is not None
            ]
            asr = {
                "avg_logprob": getattr(s, "avg_logprob", None),
                "no_speech_prob": getattr(s, "no_speech_prob", None),
                "compression_ratio": getattr(s, "compression_ratio", None),
            }
            out.append(TranscriptSegment(
                segment_id=f"{segment_id_prefix}{i}",
                start=seg_start,
                end=seg_end,
                text=s.text.strip(),
                words=words,
                asr={k: float(v) for k, v in asr.items() if v is not None},
            ))
        return out

    def asr_quality_score(text: str, asr: dict[str, Any], flags: list[str]) -> float:
        try:
            avg_logprob = float(asr.get("avg_logprob", -3.0))
        except (TypeError, ValueError):
            avg_logprob = -3.0
        try:
            no_speech_prob = float(asr.get("no_speech_prob", 0.0))
        except (TypeError, ValueError):
            no_speech_prob = 0.0
        try:
            compression_ratio = float(asr.get("compression_ratio", 1.0))
        except (TypeError, ValueError):
            compression_ratio = 1.0
        serious_penalty = 1.25 * len(
            set(flags)
            & {
                "mojibake_or_replacement_char",
                "unexpected_foreign_script",
                "long_latin_noise",
                "local_repetition",
                "high_compression_ratio",
                "short_fragment_noise",
            }
        )
        length_bonus = min(len(_normalize_repetition_text(text)) / 40.0, 0.2)
        return avg_logprob - no_speech_prob - max(0.0, compression_ratio - 2.4) * 0.25 - serious_penalty + length_bonus

    def merged_candidate_from_retry_segments(
        retry_segments: list[TranscriptSegment],
        *,
        retry_start: float,
        retry_end: float,
        target_start: float | None = None,
        target_end: float | None = None,
    ) -> TranscriptSegment | None:
        window_start = retry_start if target_start is None else target_start
        window_end = retry_end if target_end is None else target_end
        usable = [
            s
            for s in retry_segments
            if s.text.strip() and _overlap_seconds(s.start, s.end, window_start, window_end) > 0.0
        ]
        if not usable:
            return None
        text_parts: list[str] = []
        for s in usable:
            clipped = _segment_text_in_window(s, window_start, window_end).strip()
            if clipped:
                text_parts.append(clipped)
            elif (s.end - s.start) <= (window_end - window_start) + 0.4:
                text_parts.append(s.text.strip())
        text = " ".join(text_parts).strip()
        if not text:
            return None
        words: list[dict[str, Any]] = []
        for s in usable:
            words.extend(s.words)
        avg_logprobs = [float(s.asr["avg_logprob"]) for s in usable if "avg_logprob" in s.asr]
        no_speech_probs = [float(s.asr["no_speech_prob"]) for s in usable if "no_speech_prob" in s.asr]
        compression_ratios = [float(s.asr["compression_ratio"]) for s in usable if "compression_ratio" in s.asr]
        asr: dict[str, Any] = {}
        if avg_logprobs:
            asr["avg_logprob"] = sum(avg_logprobs) / len(avg_logprobs)
        if no_speech_probs:
            asr["no_speech_prob"] = sum(no_speech_probs) / len(no_speech_probs)
        if compression_ratios:
            asr["compression_ratio"] = max(compression_ratios)
        return TranscriptSegment(
            segment_id="retry_candidate",
            start=max(retry_start, min(s.start for s in usable)),
            end=min(retry_end, max(s.end for s in usable)),
            text=text,
            words=words,
            asr=asr,
        )

    def retry_suspicious_segments(segments_in: list[TranscriptSegment]) -> dict[str, Any]:
        if not enable_retry or not segments_in or cache_root is None:
            return {"enabled": bool(enable_retry), "attempted": 0, "replaced": 0}

        thresholds = _estimate_quality_thresholds(segments_in)
        retry_root = cache_root.parent / "asr_retry_audio" / cache_key
        attempted = 0
        replaced = 0
        skipped = 0
        errors = 0
        serious_flags = {
            "mojibake_or_replacement_char",
                "unexpected_foreign_script",
                "long_latin_noise",
                "katakana_phonetic_noise",
                "local_repetition",
                "high_compression_ratio",
            }
        low_quality_combo = {"low_avg_logprob", "high_compression_ratio"}
        candidates: list[tuple[int, TranscriptSegment, list[str]]] = []
        for idx, seg in enumerate(segments_in):
            flags = _quality_flags_for_text(seg.text, seg.asr, max(0.0, seg.end - seg.start), thresholds)
            if seg.asr.get("suspected_repetition_hallucination") and "local_repetition" not in flags:
                flags.append("suspected_repetition_hallucination")
            retry_reason = bool(set(flags) & serious_flags) or bool(low_quality_combo <= set(flags))
            if retry_reason:
                candidates.append((idx, seg, flags))
        if not candidates:
            return {"enabled": True, "attempted": 0, "replaced": 0, "auto_thresholds": thresholds}
        if len(candidates) > ASR_RETRY_MAX_SEGMENTS:
            skipped = len(candidates) - ASR_RETRY_MAX_SEGMENTS
            candidates = candidates[:ASR_RETRY_MAX_SEGMENTS]

        _progress(
            progress,
            f"{progress_label} 怪しい区間のretry開始",
            candidates=len(candidates),
            skipped=skipped,
        )

        duration_limit = duration_f or _audio_duration_seconds(audio_path)
        for retry_no, (idx, seg, flags) in enumerate(candidates, start=1):
            attempted += 1
            seg.asr["retry_attempted"] = True
            seg.asr["retry_flags"] = flags
            retry_start = max(0.0, float(seg.start) - ASR_RETRY_CONTEXT_SECONDS)
            retry_end = min(duration_limit or float(seg.end), float(seg.end) + ASR_RETRY_CONTEXT_SECONDS)
            window = retry_end - retry_start
            if window < ASR_RETRY_MIN_WINDOW_SECONDS:
                pad = (ASR_RETRY_MIN_WINDOW_SECONDS - window) / 2.0
                retry_start = max(0.0, retry_start - pad)
                retry_end = min(duration_limit or retry_end + pad, retry_end + pad)
            if retry_end - retry_start > ASR_RETRY_MAX_WINDOW_SECONDS:
                center = (float(seg.start) + float(seg.end)) / 2.0
                half = ASR_RETRY_MAX_WINDOW_SECONDS / 2.0
                retry_start = max(0.0, center - half)
                retry_end = min(duration_limit or center + half, center + half)
            retry_path = retry_root / f"retry_{idx:05d}_{int(round(retry_start * 1000))}_{int(round(retry_end * 1000))}.wav"
            try:
                if not retry_path.exists():
                    _write_audio_chunk(audio_path, retry_path, retry_start, retry_end)
                best_candidate: TranscriptSegment | None = None
                best_flags: list[str] = []
                best_score = -999.0
                variants: list[dict[str, Any]] = []
                for temp in ASR_RETRY_TEMPERATURES:
                    variants.append({"name": f"vad_on_t{temp}", "vad_filter": True, "temperature": temp})
                    variants.append({"name": f"vad_off_t{temp}", "vad_filter": False, "temperature": temp})
                for variant in variants:
                    retry_segments = collect_segments(
                        retry_path,
                        offset_sec=retry_start,
                        segment_id_prefix=f"r{retry_no:04d}_",
                        initial_prompt_override=None,
                        hotwords_override=None,
                        vad_filter=bool(variant["vad_filter"]),
                        retry_decode=True,
                        temperature_override=float(variant["temperature"]),
                    )
                    candidate = merged_candidate_from_retry_segments(
                        retry_segments,
                        retry_start=retry_start,
                        retry_end=retry_end,
                        target_start=float(seg.start),
                        target_end=float(seg.end),
                    )
                    if candidate is None:
                        continue
                    candidate_flags = _quality_flags_for_text(
                        candidate.text,
                        candidate.asr,
                        max(0.0, candidate.end - candidate.start),
                        thresholds,
                    )
                    score = asr_quality_score(candidate.text, candidate.asr, candidate_flags)
                    if score > best_score:
                        best_candidate = candidate
                        best_flags = candidate_flags
                        best_score = score
                        best_candidate.asr["retry_variant"] = str(variant["name"])

                original_score = asr_quality_score(seg.text, seg.asr, flags)
                normalized_original = _normalize_repetition_text(seg.text)
                normalized_candidate = _normalize_repetition_text(best_candidate.text) if best_candidate else ""
                candidate_serious = set(best_flags) & serious_flags
                can_replace = bool(
                    best_candidate
                    and normalized_candidate
                    and normalized_candidate != normalized_original
                    and not candidate_serious
                    and (
                        (set(flags) & serious_flags and best_score >= original_score + 0.15)
                        or best_score >= original_score + 0.45
                    )
                )
                if can_replace and best_candidate is not None:
                    original = {
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                        "asr": dict(seg.asr),
                        "flags": flags,
                        "score": round(original_score, 4),
                    }
                    seg.text = best_candidate.text
                    seg.words = best_candidate.words
                    seg.asr.update(best_candidate.asr)
                    seg.asr["retry_replaced"] = True
                    seg.asr["retry_original"] = original
                    seg.asr["retry_score"] = round(best_score, 4)
                    seg.asr["retry_candidate_flags"] = best_flags
                    seg.asr["retry_audio"] = str(retry_path)
                    replaced += 1
                else:
                    seg.asr["retry_replaced"] = False
                    seg.asr["retry_best_score"] = round(best_score, 4)
                    seg.asr["retry_original_score"] = round(original_score, 4)
                    seg.asr["retry_candidate_flags"] = best_flags
                    seg.asr["retry_audio"] = str(retry_path)
            except Exception as e:
                errors += 1
                seg.asr["retry_error"] = str(e)

        _progress(
            progress,
            f"{progress_label} 怪しい区間のretry完了",
            attempted=attempted,
            replaced=replaced,
            skipped=skipped,
            errors=errors,
        )
        return {
            "enabled": True,
            "attempted": attempted,
            "replaced": replaced,
            "skipped": skipped,
            "errors": errors,
            "auto_thresholds": thresholds,
        }

    segments: list[TranscriptSegment] = []
    chunk_meta: list[dict[str, Any]] = []
    if use_chunked_asr and cache_root is not None:
        chunks_root = cache_root.parent / "asr_audio_chunks" / cache_key
        core_start = 0.0
        chunk_index = 0
        total_chunks = max(1, int((duration_f + chunk_seconds - 0.001) // chunk_seconds)) if chunk_seconds else 1
        _progress(
            progress,
            f"{progress_label} 長尺chunk文字起こし開始",
            duration_sec=round(duration_f, 2),
            chunk_seconds=chunk_seconds,
            overlap_seconds=chunk_overlap_seconds,
            chunks=total_chunks,
        )
        while core_start < duration_f - 0.001:
            core_end = min(duration_f, core_start + chunk_seconds)
            chunk_start = max(0.0, core_start - chunk_overlap_seconds)
            chunk_end = min(duration_f, core_end + chunk_overlap_seconds)
            chunk_path = chunks_root / f"chunk_{chunk_index:04d}_{int(round(core_start * 1000))}_{int(round(core_end * 1000))}.wav"
            chunk_no = chunk_index + 1
            _progress(
                progress,
                f"{progress_label} chunk {chunk_no}/{total_chunks} 開始",
                chunk=chunk_no,
                chunks=total_chunks,
                core_start_sec=round(core_start, 2),
                core_end_sec=round(core_end, 2),
                chunk_start_sec=round(chunk_start, 2),
                chunk_end_sec=round(chunk_end, 2),
                chunk_audio=str(chunk_path),
                progress_percent=round(progress_base + progress_span * (core_start / duration_f), 1)
                if progress_base is not None and progress_span is not None and duration_f
                else None,
            )
            if not chunk_path.exists():
                chunk_info = _write_audio_chunk(audio_path, chunk_path, chunk_start, chunk_end)
            else:
                chunk_info = {"path": str(chunk_path), "start_sec": round(chunk_start, 3), "end_sec": round(chunk_end, 3), "reused": True}
            chunk_segments = collect_segments(
                chunk_path,
                offset_sec=chunk_start,
                core_start=core_start,
                core_end=core_end,
                segment_id_prefix=f"c{chunk_index:04d}_",
            )
            segments.extend(chunk_segments)
            chunk_info.update({"core_start_sec": round(core_start, 3), "core_end_sec": round(core_end, 3), "segments": len(chunk_segments)})
            chunk_meta.append(chunk_info)
            _progress(
                progress,
                f"{progress_label} chunk {chunk_no}/{total_chunks} 完了",
                chunk=chunk_no,
                chunks=total_chunks,
                segments=len(chunk_segments),
                reused=bool(chunk_info.get("reused", False)),
                core_start_sec=round(core_start, 2),
                core_end_sec=round(core_end, 2),
            )
            core_start = core_end
            chunk_index += 1
            ratio = max(0.0, min(1.0, core_end / duration_f)) if duration_f else 1.0
            if ratio >= last_reported + 0.05 or ratio >= 0.995:
                emit_progress(ratio, current_sec=core_end, duration_sec=duration_f)
                last_reported = ratio
        segments.sort(key=lambda seg: (seg.start, seg.end))
        for i, seg in enumerate(segments):
            seg.segment_id = str(i)
    else:
        segments = collect_segments(audio_path)
        if duration_f <= 0.0 and segments:
            duration_f = max(seg.end for seg in segments)
        for i, seg in enumerate(segments):
            seg.segment_id = str(i)
            if duration_f > 0:
                ratio = max(0.0, min(1.0, seg.end / duration_f))
                if ratio >= last_reported + 0.05 or ratio >= 0.995:
                    emit_progress(ratio, current_sec=seg.end, duration_sec=duration_f)
                    last_reported = ratio

    retry_meta = retry_suspicious_segments(segments)

    if cache_path is not None:
        try:
            save_transcript(
                segments,
                cache_path,
                meta={
                    "cache_key": cache_key,
                    "cache_kind": "faster_whisper_segments_v5",
                    "audio": str(audio_path),
                    "model": model_size,
                    "language": language,
                    "condition_on_previous_text": False,
                    "chunked_asr": use_chunked_asr,
                    "chunk_seconds": chunk_seconds if use_chunked_asr else None,
                    "chunk_overlap_seconds": chunk_overlap_seconds if use_chunked_asr else None,
                    "chunks": chunk_meta,
                    "retry": retry_meta,
                },
            )
            _progress(progress, f"{progress_label} 文字起こしキャッシュ保存", cache=str(cache_path), segments=len(segments))
        except Exception as e:
            _progress(progress, f"{progress_label} 文字起こしキャッシュ保存失敗", cache=str(cache_path), error=str(e))
    return segments


def retry_transcript_segments(
    audio_path: str | Path,
    transcript_path: str | Path,
    output_path: str | Path | None = None,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    model_size: str = "large-v3",
    language: str = "ja",
    force_replace: bool = False,
    progress: ProgressCallback | None = None,
) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    """既存JSONを読み、指定範囲または怪しい区間だけASR retryしたJSONを作る。

    通常は「元より明確に良い」場合だけ差し替える。force_replace=True の場合は
    重大な品質フラグがない候補なら差し替える。
    """
    audio_path = Path(audio_path)
    transcript_path = Path(transcript_path)
    output_path = Path(output_path) if output_path is not None else transcript_path.with_name(transcript_path.stem + ".manual_retry.json")
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"transcript json not found: {transcript_path}")

    segments, meta = _load_transcript_with_meta(transcript_path)
    thresholds = _estimate_quality_thresholds(segments)
    duration = _audio_duration_seconds(audio_path)
    model = _get_faster_whisper_model(model_size)
    cache_key = _stable_hash(
        {
            "kind": "manual_asr_retry_v1",
            "audio": _file_fingerprint(audio_path),
            "transcript": _file_fingerprint(transcript_path),
            "model": model_size,
            "language": language,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "force_replace": force_replace,
        }
    )
    retry_root = output_path.parent / ".transcribe_cache" / "manual_retry_audio" / cache_key

    serious_flags = {
        "mojibake_or_replacement_char",
        "unexpected_foreign_script",
        "long_latin_noise",
        "katakana_phonetic_noise",
        "local_repetition",
        "high_compression_ratio",
    }

    def seg_flags(seg: TranscriptSegment) -> list[str]:
        flags = _quality_flags_for_text(seg.text, seg.asr, max(0.0, seg.end - seg.start), thresholds)
        if seg.asr.get("suspected_repetition_hallucination") and "suspected_repetition_hallucination" not in flags:
            flags.append("suspected_repetition_hallucination")
        return flags

    def score_text(text: str, asr: dict[str, Any], flags: list[str]) -> float:
        try:
            avg_logprob = float(asr.get("avg_logprob", -3.0))
        except (TypeError, ValueError):
            avg_logprob = -3.0
        try:
            no_speech_prob = float(asr.get("no_speech_prob", 0.0))
        except (TypeError, ValueError):
            no_speech_prob = 0.0
        try:
            compression_ratio = float(asr.get("compression_ratio", 1.0))
        except (TypeError, ValueError):
            compression_ratio = 1.0
        penalty = 1.25 * len(set(flags) & serious_flags)
        return avg_logprob - no_speech_prob - max(0.0, compression_ratio - 2.4) * 0.25 - penalty + min(len(_normalize_repetition_text(text)) / 40.0, 0.2)

    def collect_retry(path: Path, offset: float, vad_filter: bool) -> list[TranscriptSegment]:
        raw_segments, _info = model.transcribe(
            str(path),
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=None,
            hotwords=None,
            temperature=0.0,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )
        out: list[TranscriptSegment] = []
        for i, s in enumerate(raw_segments):
            words = [
                {"start": float(w.start) + offset, "end": float(w.end) + offset, "word": w.word.strip()}
                for w in (s.words or [])
                if w.start is not None and w.end is not None
            ]
            asr = {
                "avg_logprob": getattr(s, "avg_logprob", None),
                "no_speech_prob": getattr(s, "no_speech_prob", None),
                "compression_ratio": getattr(s, "compression_ratio", None),
            }
            out.append(
                TranscriptSegment(
                    segment_id=f"manual_retry_{i}",
                    start=float(s.start) + offset,
                    end=float(s.end) + offset,
                    text=str(s.text).strip(),
                    words=words,
                    asr={k: float(v) for k, v in asr.items() if v is not None},
                )
            )
        return out

    def merged_retry_candidate(
        retry_segments: list[TranscriptSegment],
        retry_start: float,
        retry_end: float,
        target_start: float,
        target_end: float,
    ) -> TranscriptSegment | None:
        usable = [s for s in retry_segments if s.text.strip() and _overlap_seconds(s.start, s.end, target_start, target_end) > 0.0]
        if not usable:
            return None
        parts: list[str] = []
        for s in usable:
            clipped = _segment_text_in_window(s, target_start, target_end).strip()
            if clipped:
                parts.append(clipped)
            elif (s.end - s.start) <= (target_end - target_start) + 0.4:
                parts.append(s.text.strip())
        text = " ".join(parts).strip()
        if not text:
            return None
        words: list[dict[str, Any]] = []
        for s in usable:
            words.extend(s.words)
        avg_values = [float(s.asr["avg_logprob"]) for s in usable if "avg_logprob" in s.asr]
        ns_values = [float(s.asr["no_speech_prob"]) for s in usable if "no_speech_prob" in s.asr]
        comp_values = [float(s.asr["compression_ratio"]) for s in usable if "compression_ratio" in s.asr]
        asr: dict[str, Any] = {}
        if avg_values:
            asr["avg_logprob"] = sum(avg_values) / len(avg_values)
        if ns_values:
            asr["no_speech_prob"] = sum(ns_values) / len(ns_values)
        if comp_values:
            asr["compression_ratio"] = max(comp_values)
        return TranscriptSegment(
            segment_id="manual_retry_candidate",
            start=max(retry_start, min(s.start for s in usable)),
            end=min(retry_end, max(s.end for s in usable)),
            text=text,
            words=words,
            asr=asr,
        )

    selected: list[tuple[int, TranscriptSegment, list[str]]] = []
    manual_range = start_sec is not None or end_sec is not None
    range_start = float(start_sec) if start_sec is not None else 0.0
    range_end = float(end_sec) if end_sec is not None else (duration or max((s.end for s in segments), default=0.0))
    if range_end < range_start:
        raise ValueError("end must be greater than start")
    for idx, seg in enumerate(segments):
        flags = seg_flags(seg)
        in_range = _overlap_seconds(seg.start, seg.end, range_start, range_end) > 0.0
        auto_suspicious = bool(set(flags) & serious_flags) or {"low_avg_logprob", "high_compression_ratio"} <= set(flags)
        if (manual_range and in_range) or (not manual_range and auto_suspicious):
            selected.append((idx, seg, flags))

    if len(selected) > ASR_RETRY_MAX_SEGMENTS:
        selected = selected[:ASR_RETRY_MAX_SEGMENTS]

    _progress(progress, "手動retry開始", candidates=len(selected), progress_percent=5.0)
    replaced = 0
    attempted = 0
    errors = 0
    for n, (idx, seg, original_flags) in enumerate(selected, start=1):
        attempted += 1
        retry_start = max(0.0, float(seg.start) - ASR_RETRY_CONTEXT_SECONDS)
        retry_end = min(duration or float(seg.end), float(seg.end) + ASR_RETRY_CONTEXT_SECONDS)
        if manual_range:
            retry_start = max(0.0, min(retry_start, range_start))
            retry_end = min(duration or retry_end, max(retry_end, range_end))
        if retry_end - retry_start < ASR_RETRY_MIN_WINDOW_SECONDS:
            pad = (ASR_RETRY_MIN_WINDOW_SECONDS - (retry_end - retry_start)) / 2.0
            retry_start = max(0.0, retry_start - pad)
            retry_end = min(duration or retry_end + pad, retry_end + pad)
        if retry_end - retry_start > ASR_RETRY_MAX_WINDOW_SECONDS:
            center = (seg.start + seg.end) / 2.0
            retry_start = max(0.0, center - ASR_RETRY_MAX_WINDOW_SECONDS / 2.0)
            retry_end = min(duration or center + ASR_RETRY_MAX_WINDOW_SECONDS / 2.0, center + ASR_RETRY_MAX_WINDOW_SECONDS / 2.0)
        retry_path = retry_root / f"retry_{idx:05d}_{int(round(retry_start * 1000))}_{int(round(retry_end * 1000))}.wav"
        try:
            if not retry_path.exists():
                _write_audio_chunk(audio_path, retry_path, retry_start, retry_end)
            best: TranscriptSegment | None = None
            best_flags: list[str] = []
            best_score = -999.0
            best_variant = ""
            for variant, vad in (("vad_on", True), ("vad_off", False)):
                cand = merged_retry_candidate(
                    collect_retry(retry_path, retry_start, vad),
                    retry_start,
                    retry_end,
                    float(seg.start),
                    float(seg.end),
                )
                if cand is None:
                    continue
                flags = _quality_flags_for_text(cand.text, cand.asr, max(0.0, cand.end - cand.start), thresholds)
                score = score_text(cand.text, cand.asr, flags)
                if score > best_score:
                    best = cand
                    best_flags = flags
                    best_score = score
                    best_variant = variant
            original_score = score_text(seg.text, seg.asr, original_flags)
            candidate_text = _normalize_repetition_text(best.text) if best else ""
            original_text = _normalize_repetition_text(seg.text)
            can_replace = bool(
                best
                and candidate_text
                and candidate_text != original_text
                and not (set(best_flags) & serious_flags)
                and (force_replace or best_score >= original_score + (0.15 if set(original_flags) & serious_flags else 0.45))
            )
            seg.asr["manual_retry_attempted"] = True
            seg.asr["manual_retry_audio"] = str(retry_path)
            seg.asr["manual_retry_original_score"] = round(original_score, 4)
            seg.asr["manual_retry_best_score"] = round(best_score, 4)
            seg.asr["manual_retry_candidate_flags"] = best_flags
            if can_replace and best is not None:
                original = {"text": seg.text, "start": seg.start, "end": seg.end, "asr": dict(seg.asr), "flags": original_flags}
                seg.text = best.text
                seg.words = best.words
                seg.asr.update(best.asr)
                seg.asr["manual_retry_replaced"] = True
                seg.asr["manual_retry_original"] = original
                seg.asr["manual_retry_variant"] = best_variant
                seg.asr.pop("suppress_in_text_outputs", None)
                if "manual_retry_replaced" not in seg.confidence_reasons:
                    seg.confidence_reasons.append("manual_retry_replaced")
                replaced += 1
            else:
                seg.asr["manual_retry_replaced"] = False
        except Exception as e:
            errors += 1
            seg.asr["manual_retry_error"] = str(e)
        _progress(
            progress,
            "手動retry進捗",
            current=n,
            total=len(selected),
            progress_percent=5.0 + 90.0 * (n / max(1, len(selected))),
        )

    meta = {
        **meta,
        "manual_retry": {
            "source_json": str(transcript_path),
            "audio": str(audio_path),
            "model": model_size,
            "language": language,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "force_replace": force_replace,
            "attempted": attempted,
            "replaced": replaced,
            "errors": errors,
            "auto_thresholds": thresholds,
        },
    }
    save_transcript(segments, output_path, meta=meta)
    _progress(progress, "手動retry完了", output=str(output_path), attempted=attempted, replaced=replaced, errors=errors, progress_percent=100.0)
    return segments, meta["manual_retry"]


def transcribe_with_faster_whisper(
    audio_path: str | Path,
    output_path: str | Path,
    model_size: str = "large-v3",
    language: str = "ja",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
) -> list[TranscriptSegment]:
    """軽量フォールバック。word_timestamps=Trueで単語時刻も保存する。"""
    segments = _faster_whisper_transcribe_only(
        audio_path,
        model_size=model_size,
        language=language,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )
    save_transcript(
        segments, output_path,
        meta={
            "transcriber": "faster-whisper",
            "model": model_size,
            "language": language,
            "unit": "word",
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
        },
    )
    return segments


def transcribe_with_whisperx(
    audio_path: str | Path,
    output_path: str | Path,
    model_size: str = "large-v3",
    language: str = "ja",
    batch_size: int = 8,
    initial_prompt: str | None = None,
) -> list[TranscriptSegment]:
    """WhisperX: Whisper文字起こし + forced alignmentで単語時刻を出す。

    NOTE: WhisperX は Python 3.13 以前のみサポート。Python 3.14 環境では
    `transcribe_with_diarization` (faster-whisper + pyannote.audio) を使う。
    """
    try:
        import whisperx  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "WhisperX が未インストール、または Python 3.14 で非対応です。"
            "代わりに transcribe_with_diarization (faster-whisper + pyannote) を使ってください。"
        ) from e

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    device = _device_auto()
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(model_size, device=device, compute_type=compute_type, language=language)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language, initial_prompt=initial_prompt)

    align_model, metadata = whisperx.load_align_model(language_code=result.get("language", language), device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

    segments: list[TranscriptSegment] = []
    for i, seg in enumerate(aligned.get("segments", [])):
        words = []
        for w in seg.get("words", []) or []:
            if "start" in w and "end" in w:
                words.append({"start": float(w["start"]), "end": float(w["end"]), "word": str(w.get("word", "")).strip()})
        segments.append(TranscriptSegment(
            segment_id=str(i),
            start=float(seg.get("start", words[0]["start"] if words else 0.0)),
            end=float(seg.get("end", words[-1]["end"] if words else 0.0)),
            text=str(seg.get("text", "")).strip(),
            words=words,
        ))
    save_transcript(
        segments, output_path,
        meta={
            "transcriber": "whisperx",
            "model": model_size,
            "language": language,
            "unit": "word",
            "initial_prompt": initial_prompt,
        },
    )
    return segments


# -----------------------------------------------------------------------------
# Diarization (faster-whisper + pyannote.audio)
# -----------------------------------------------------------------------------


PYANNOTE_SAMPLE_RATE = 16000
PYANNOTE_CHANNELS = 1


def _patch_torchaudio_for_pyannote() -> None:
    """pyannote.audio 3.x が参照する旧 torchaudio API を最小限スタブする。"""
    from collections import namedtuple

    try:
        import torchaudio as _ta  # type: ignore
    except Exception:
        return

    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(_ta, "AudioMetaData"):
        _ta.AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )


def _load_pyannote_audio_input(audio_path: str | Path) -> dict[str, Any]:
    """pyannote に渡す音声を必ず 16kHz/mono の waveform dict に正規化する。

    ファイルパスを直接 `Pipeline(...)` に渡すと、stereo/48kHz WAV がそのまま
    speaker embedding へ流れる環境がある。ここで soundfile + scipy で明示的に
    downmix/resample し、torchaudio/torchcodec の読み込み差異も避ける。
    """
    from math import gcd

    import soundfile as _sf  # type: ignore
    import torch  # type: ignore
    from scipy.signal import resample_poly  # type: ignore

    raw, source_sr = _sf.read(str(audio_path), always_2d=True, dtype="float32")
    source_channels = int(raw.shape[1])

    if source_channels > PYANNOTE_CHANNELS:
        raw = raw.mean(axis=1, keepdims=True)
    elif source_channels == 0:
        raise RuntimeError(f"audio has no channels: {audio_path}")

    if int(source_sr) != PYANNOTE_SAMPLE_RATE:
        g = gcd(PYANNOTE_SAMPLE_RATE, int(source_sr))
        raw = resample_poly(raw, PYANNOTE_SAMPLE_RATE // g, int(source_sr) // g, axis=0)

    waveform = torch.from_numpy(raw.T.copy()).float().contiguous()
    return {
        "waveform": waveform,
        "sample_rate": PYANNOTE_SAMPLE_RATE,
        "meta": {
            "source_sample_rate": int(source_sr),
            "source_channels": source_channels,
            "sample_rate": PYANNOTE_SAMPLE_RATE,
            "channels": PYANNOTE_CHANNELS,
            "num_samples": int(waveform.shape[1]),
        },
    }


def _pipeline_from_pretrained_for_pyannote(Pipeline: Any, hf_token: str) -> Any:
    """Python 3.14 + speechbrain LazyModule + `python -m` での inspect 失敗を避ける。"""
    import inspect

    original_getmodule = inspect.getmodule

    def safe_getmodule(object: Any, _filename: str | None = None) -> Any:
        try:
            return original_getmodule(object, _filename)
        except ImportError:
            return None

    inspect.getmodule = safe_getmodule
    try:
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    finally:
        inspect.getmodule = original_getmodule


def _load_pyannote_pipeline(hf_token: str) -> Any:
    pipeline = _PYANNOTE_PIPELINE_CACHE.get(hf_token)
    if pipeline is not None:
        return pipeline

    _patch_torchaudio_for_pyannote()

    import torch  # type: ignore

    original_torch_load = torch.load

    def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]
    try:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "pyannote.audio が未インストールです。pip install pyannote.audio"
            ) from e

        pipeline = _pipeline_from_pretrained_for_pyannote(Pipeline, hf_token)
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]

    if pipeline is None:
        raise RuntimeError(
            "pyannote パイプラインのロードに失敗。HF_TOKEN が有効か、"
            "https://huggingface.co/pyannote/speaker-diarization-3.1 で利用同意済みか確認してください。"
        )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    _PYANNOTE_PIPELINE_CACHE[hf_token] = pipeline
    return pipeline


def _run_pyannote_diarization(
    audio_path: str | Path,
    hf_token: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[tuple[float, float, str]]:
    """pyannote/speaker-diarization-3.1 標準の話者区間を返す。"""
    pipeline = _load_pyannote_pipeline(hf_token)
    audio_input = _load_pyannote_audio_input(audio_path)

    kwargs: dict[str, Any] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    diar = pipeline(audio_input, **kwargs)
    intervals: list[tuple[float, float, str]] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        intervals.append((float(turn.start), float(turn.end), str(speaker)))
    intervals.sort(key=lambda t: t[0])
    return intervals


def _cluster_count_args(
    min_speakers: int | None,
    max_speakers: int | None,
) -> tuple[int | None, int | None, int | None]:
    if min_speakers is not None and max_speakers is not None and min_speakers == max_speakers:
        return min_speakers, None, None
    return None, min_speakers, max_speakers


def _speaker_labels_from_clusters(clusters: list[int]) -> list[str]:
    """クラスタIDを出現順の SPEAKER_XX に正規化する。"""
    mapping: dict[int, str] = {}
    labels: list[str] = []
    for cluster in clusters:
        if cluster not in mapping:
            mapping[cluster] = f"SPEAKER_{len(mapping):02d}"
        labels.append(mapping[cluster])
    return labels


def _merge_segment_intervals(
    intervals: list[tuple[float, float, str]],
    max_gap: float = 0.05,
) -> list[tuple[float, float, str]]:
    """同じ話者の隣接 segment を表示用に結合する。"""
    if not intervals:
        return []
    merged: list[tuple[float, float, str]] = []
    cur_start, cur_end, cur_speaker = intervals[0]
    for start, end, speaker in intervals[1:]:
        if speaker == cur_speaker and start - cur_end <= max_gap:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end, cur_speaker))
            cur_start, cur_end, cur_speaker = start, end, speaker
    merged.append((cur_start, cur_end, cur_speaker))
    return merged


def _run_pyannote_segment_embedding_diarization(
    audio_path: str | Path,
    segments: list[TranscriptSegment],
    hf_token: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> DiarizationResult:
    """Whisper segment ごとに speaker embedding を取り、短い発話単位でクラスタリングする。

    pyannote 標準 diarization の長い区間を単語に割り当てるだけだと、
    「短い相づち/割り込み」が長い話者区間に吸収される。ここでは Whisper の
    segment 境界を話者判定単位として使い、各 segment の embedding を直接クラスタリングする。
    """
    if not segments:
        return DiarizationResult(intervals=[], meta={"diarization_method": "segment_embedding_clustering"})

    import numpy as np  # type: ignore
    import torch  # type: ignore
    from pyannote.core import SlidingWindow, SlidingWindowFeature  # type: ignore

    pipeline = _load_pyannote_pipeline(hf_token)
    audio_input = _load_pyannote_audio_input(audio_path)
    waveform = audio_input["waveform"]
    sample_rate = int(audio_input["sample_rate"])
    total_samples = int(waveform.shape[1])
    min_samples = int(getattr(pipeline._embedding, "min_num_samples", 1))

    embeddings: list[np.ndarray] = []
    valid_segments: list[TranscriptSegment] = []
    for seg in segments:
        start_sample = max(0, min(total_samples, int(float(seg.start) * sample_rate)))
        end_sample = max(start_sample, min(total_samples, int(float(seg.end) * sample_rate)))
        chunk = waveform[:, start_sample:end_sample]
        if chunk.shape[1] == 0:
            continue
        if chunk.shape[1] < min_samples:
            chunk = torch.nn.functional.pad(chunk, (0, min_samples - chunk.shape[1]))
        embedding = np.asarray(pipeline._embedding(chunk[None])).reshape(-1)
        if np.any(np.isnan(embedding)):
            continue
        embeddings.append(embedding)
        valid_segments.append(seg)

    if not valid_segments:
        return DiarizationResult(
            intervals=[],
            meta={
                "diarization_method": "segment_embedding_clustering",
                "speaker_embedding_error": "no_valid_segment_embeddings",
                "audio_normalization": audio_input.get("meta", {}),
            },
        )

    embedding_array = np.asarray(embeddings)[:, None, :]
    active_segmentations = SlidingWindowFeature(
        np.ones((len(valid_segments), 1, 1), dtype=np.float32),
        SlidingWindow(start=0.0, duration=1.0, step=1.0),
    )
    num_clusters, min_clusters, max_clusters = _cluster_count_args(min_speakers, max_speakers)
    hard_clusters, _soft_clusters, _centroids = pipeline.clustering(
        embedding_array,
        segmentations=active_segmentations,
        num_clusters=num_clusters,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )

    cluster_ids = [int(c) for c in hard_clusters.reshape(-1).tolist()]
    speaker_labels = _speaker_labels_from_clusters(cluster_ids)
    raw_intervals = [
        (float(seg.start), float(seg.end), speaker)
        for seg, speaker in zip(valid_segments, speaker_labels)
    ]
    segment_speakers = {
        str(seg.segment_id): speaker
        for seg, speaker in zip(valid_segments, speaker_labels)
    }

    return DiarizationResult(
        intervals=_merge_segment_intervals(raw_intervals),
        segment_speakers=segment_speakers,
        meta={
            "diarization_method": "segment_embedding_clustering",
            "embedding_model": str(getattr(pipeline, "embedding", "")),
            "speaker_count_args": {
                "num_speakers": num_clusters,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
            "audio_normalization": audio_input.get("meta", {}),
            "speaker_segments": [
                {
                    "segment_id": str(seg.segment_id),
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "speaker": speaker,
                }
                for seg, speaker in zip(valid_segments, speaker_labels)
            ],
        },
    )

def _speaker_at(time: float, intervals: list[tuple[float, float, str]]) -> str | None:
    """指定時刻にアクティブな話者を返す (重なりがあれば最初に見つかったもの)。"""
    for s, e, sp in intervals:
        if s <= time < e:
            return sp
    return None


def assign_speakers_to_segments(
    segments: list[TranscriptSegment],
    intervals: list[tuple[float, float, str]],
) -> list[TranscriptSegment]:
    """各 word に speaker を割り当て、segment に primary_speaker を多数決で設定。

    word の中央時刻でアクティブな話者を割り当てる。
    word に start/end が無い場合は segment の話者で代替 (多数決対象外)。
    """
    for seg in segments:
        for w in seg.words:
            try:
                ws = float(w["start"])
                we = float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            mid = (ws + we) / 2
            sp = _speaker_at(mid, intervals)
            if sp is not None:
                w["speaker"] = sp
        # primary_speaker = word単位 speaker の合計時間が最大のもの
        durations: dict[str, float] = {}
        for w in seg.words:
            sp = w.get("speaker")
            if not sp:
                continue
            try:
                durations[sp] = durations.get(sp, 0.0) + (float(w["end"]) - float(w["start"]))
            except (KeyError, TypeError, ValueError):
                continue
        if durations:
            seg.primary_speaker = max(durations.items(), key=lambda kv: kv[1])[0]
        else:
            # word で取れなければ segment 中央時刻で代替
            mid = (seg.start + seg.end) / 2
            seg.primary_speaker = _speaker_at(mid, intervals)
    return segments


def assign_segment_speakers_to_words(
    segments: list[TranscriptSegment],
    segment_speakers: dict[str, str],
) -> list[TranscriptSegment]:
    """segment embedding で決めた話者を、その segment 内の全 word に付与する。"""
    for seg in segments:
        speaker = segment_speakers.get(str(seg.segment_id))
        if not speaker:
            continue
        seg.primary_speaker = speaker
        for word in seg.words:
            word["speaker"] = speaker
    return segments


# -----------------------------------------------------------------------------
# Speaker-track RMS diarization
# -----------------------------------------------------------------------------

SpeakerTrack = tuple[str, str | Path]
AutoFloat = float | str | None


def _as_speaker_tracks(speaker_tracks: list[SpeakerTrack]) -> list[tuple[str, Path]]:
    """CLI/GUIから来た話者別トラック指定を正規化する。"""
    normalized: list[tuple[str, Path]] = []
    for i, (label, path) in enumerate(speaker_tracks, start=1):
        label = (label or f"TRACK_{i}").strip()
        if not label:
            label = f"TRACK_{i}"
        audio_path = Path(path)
        if audio_path.suffix.lower() not in SPEAKER_TRACK_ALLOWED_SUFFIXES:
            raise ValueError(f"話者別トラックはWAVのみ対応です: {audio_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"speaker track not found: {audio_path}")
        normalized.append((label, audio_path))
    if not normalized:
        raise ValueError("speaker_tracks is empty")
    return normalized


def inspect_audio_file(path: str | Path) -> dict[str, Any]:
    """音声ファイルの基本情報を読む。主に同期チェック用。"""
    import soundfile as sf  # type: ignore

    audio_path = Path(path)
    info = sf.info(str(audio_path))
    duration = float(info.duration)
    return {
        "path": str(audio_path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration": duration,
        "duration_sec": round(duration, 3),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def inspect_audio_alignment(
    audio_path: str | Path | None,
    speaker_tracks: list[SpeakerTrack],
    tolerance_sec: float = 2.0,
) -> dict[str, Any]:
    """mix/動画音声と話者別トラックの同期前提を検査する。

    - speaker_tracks 同士の長さ差が大きい場合は errors。
    - audio_path が読め、話者別トラックと長さが大きく違う場合も errors。
    - mp4等で soundfile が読めない場合は warnings にし、Whisper側には任せる。
    """
    tracks = _as_speaker_tracks(speaker_tracks)
    errors: list[str] = []
    warnings: list[str] = []
    track_infos: list[dict[str, Any]] = []

    for label, track_path in tracks:
        try:
            info = inspect_audio_file(track_path)
            info["label"] = label
            track_infos.append(info)
        except Exception as e:
            errors.append(f"話者別トラックを読めません: {label}={track_path} ({e})")

    sample_rates = {int(info["sample_rate"]) for info in track_infos if info.get("sample_rate")}
    if len(sample_rates) > 1:
        errors.append(f"話者別トラックのsample rateが一致していません: {sorted(sample_rates)}")

    durations = [float(info["duration"]) for info in track_infos if info.get("duration") is not None]
    track_duration = min(durations) if durations else None
    if durations:
        duration_span = max(durations) - min(durations)
        if duration_span > tolerance_sec:
            errors.append(
                f"話者別トラック同士の長さ差が大きすぎます: {duration_span:.2f}秒差 "
                f"(許容 {tolerance_sec:.2f}秒)"
            )

    audio_info: dict[str, Any] | None = None
    if audio_path is not None:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            errors.append(f"mix/動画音声が見つかりません: {audio_path}")
        else:
            try:
                audio_info = inspect_audio_file(audio_path)
            except Exception as e:
                warnings.append(f"mix/動画音声の長さを自動確認できません: {audio_path} ({e})")

    if audio_info is not None and track_duration is not None:
        diff = abs(float(audio_info["duration"]) - float(track_duration))
        if diff > tolerance_sec:
            errors.append(
                f"mix/動画音声と話者別トラックの長さが一致していません: {diff:.2f}秒差 "
                f"(mix {float(audio_info['duration']):.2f}秒 / tracks {track_duration:.2f}秒 / 許容 {tolerance_sec:.2f}秒)"
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "audio": audio_info,
        "speaker_tracks": track_infos,
        "track_duration_sec": round(float(track_duration), 3) if track_duration is not None else None,
        "tolerance_sec": float(tolerance_sec),
    }


def validate_audio_alignment(
    audio_path: str | Path | None,
    speaker_tracks: list[SpeakerTrack],
    tolerance_sec: float = 2.0,
    allow_mismatch: bool = False,
) -> dict[str, Any]:
    """同期チェックを行い、危険な不一致があれば例外にする。"""
    result = inspect_audio_alignment(audio_path, speaker_tracks, tolerance_sec=tolerance_sec)
    if result["errors"] and not allow_mismatch:
        raise ValueError("入力音声の同期チェックに失敗しました:\n- " + "\n- ".join(result["errors"]))
    result["allowed_mismatch"] = bool(allow_mismatch and result["errors"])
    return result


def _rms_db(samples: Any, min_db: float = -120.0) -> float:
    import math
    import numpy as np  # type: ignore

    if samples is None:
        return min_db
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return min_db
    rms = float(np.sqrt(np.mean(arr * arr, dtype=np.float64)))
    if not math.isfinite(rms) or rms <= 1e-12:
        return min_db
    return max(min_db, 20.0 * math.log10(rms))


def _is_auto_value(value: AutoFloat) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "auto", "自動"})


def _coerce_float(value: AutoFloat, fallback: float) -> float:
    if _is_auto_value(value):
        return fallback
    return float(value)  # type: ignore[arg-type]


def _estimate_speaker_track_rms_options(
    speaker_tracks: list[tuple[str, Path]],
    sample_seconds: float = 1.0,
    stride_seconds: float = 8.0,
    max_windows: int = 600,
) -> dict[str, Any]:
    """全体から軽くサンプリングしてRMS判定パラメータを推定する。"""
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    infos = [sf.info(str(path)) for _label, path in speaker_tracks]
    sample_rate = int(infos[0].samplerate)
    frames = min(int(info.frames) for info in infos)
    window_frames = max(1, int(sample_seconds * sample_rate))
    if frames <= window_frames:
        starts = np.array([0], dtype=np.int64)
    else:
        approx = max(1, int((frames / sample_rate) / max(0.1, stride_seconds)))
        n_windows = min(max_windows, approx)
        starts = np.linspace(0, frames - window_frames, num=n_windows, dtype=np.int64)

    db_rows: list[list[float]] = []
    files = [sf.SoundFile(str(path), mode="r") for _label, path in speaker_tracks]
    try:
        for start in starts:
            row: list[float] = []
            for f in files:
                f.seek(int(start))
                data = f.read(window_frames, dtype="float32", always_2d=True)
                if data.ndim == 2 and data.shape[1] > 1:
                    data = np.mean(data, axis=1)
                else:
                    data = data[:, 0]
                row.append(_rms_db(data))
            db_rows.append(row)
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass

    db = np.asarray(db_rows, dtype=np.float32)
    if db.size == 0:
        return {
            "active_db": -50.0,
            "overlap_db": 8.0,
            "margin": 0.10,
            "reason": "empty_sample",
        }

    # 各トラックの「静かな側」と「よく喋っている側」の中間を発話しきい値にする。
    active_candidates: list[float] = []
    per_track_percentiles: dict[str, dict[str, float]] = {}
    for idx, (label, _path) in enumerate(speaker_tracks):
        values = db[:, idx]
        p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
        per_track_percentiles[label] = {
            "p10": round(float(p10), 1),
            "p25": round(float(p25), 1),
            "p50": round(float(p50), 1),
            "p75": round(float(p75), 1),
            "p90": round(float(p90), 1),
        }
        active_candidates.append(float((p25 + p75) / 2.0))

    active_db = float(np.median(active_candidates))
    active_db = max(-65.0, min(-35.0, active_db))

    # 両方activeな窓の音量差から、被り判定の差分しきい値を推定する。
    sorted_db = np.sort(db, axis=1)[:, ::-1]
    top = sorted_db[:, 0]
    second = sorted_db[:, 1] if db.shape[1] >= 2 else np.full_like(top, -120.0)
    both_active = (top >= active_db) & (second >= active_db)
    diffs = top[both_active] - second[both_active]
    if diffs.size >= 10:
        overlap_db = float(np.percentile(diffs, 75))
        overlap_db = max(5.0, min(12.0, overlap_db))
    else:
        overlap_db = 8.0

    leakage_estimate: dict[str, Any] = {}
    top_idx = np.argmax(db, axis=1)
    for idx, (label, _path) in enumerate(speaker_tracks):
        others = [j for j in range(db.shape[1]) if j != idx]
        if not others:
            continue
        single_like = (top_idx == idx) & (db[:, idx] >= active_db) & ((db[:, idx] - np.max(db[:, others], axis=1)) >= 12.0)
        if np.any(single_like):
            separation = db[single_like, idx][:, None] - db[single_like][:, others]
            leakage_estimate[label] = {
                "median_separation_db": round(float(np.median(separation)), 1),
                "p10_separation_db": round(float(np.percentile(separation, 10)), 1),
                "sample_windows": int(np.sum(single_like)),
            }

    return {
        "active_db": round(active_db, 1),
        "overlap_db": round(overlap_db, 1),
        "margin": 0.10,
        "sample_seconds": sample_seconds,
        "stride_seconds": stride_seconds,
        "sample_windows": int(len(starts)),
        "per_track_db_percentiles": per_track_percentiles,
        "leakage_estimate": leakage_estimate,
    }


class SpeakerTrackRmsDiarizer:
    """同期済みの話者別音声トラックをRMSで比較して話者を推定する。

    前提:
    - 各トラックは同じ収録クロック・同じ開始時刻。
    - 少し他人の声が漏れていても、本人トラックの音量が大きいことを利用する。
    """

    def __init__(
        self,
        speaker_tracks: list[SpeakerTrack],
        active_db: AutoFloat = "auto",
        overlap_db: AutoFloat = "auto",
        margin: AutoFloat = "auto",
    ) -> None:
        import soundfile as sf  # type: ignore

        self.tracks = _as_speaker_tracks(speaker_tracks)
        self.auto_options: dict[str, Any] = {}
        if _is_auto_value(active_db) or _is_auto_value(overlap_db) or _is_auto_value(margin):
            self.auto_options = _estimate_speaker_track_rms_options(self.tracks)
        self.active_db = _coerce_float(active_db, float(self.auto_options.get("active_db", -50.0)))
        self.overlap_db = _coerce_float(overlap_db, float(self.auto_options.get("overlap_db", 8.0)))
        self.margin = _coerce_float(margin, float(self.auto_options.get("margin", 0.10)))
        self._sf = sf
        self.files: list[Any] = []

        infos = [(label, sf.info(str(path))) for label, path in self.tracks]
        self.sample_rate = int(infos[0][1].samplerate)
        for label, info in infos:
            if int(info.samplerate) != self.sample_rate:
                raise ValueError(
                    f"speaker track sample_rate mismatch: {label}={info.samplerate}, expected={self.sample_rate}"
                )
        self.frames = min(int(info.frames) for _, info in infos)
        self.duration = self.frames / self.sample_rate if self.sample_rate else 0.0
        self.info = [
            {
                "label": label,
                "path": str(path),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "frames": int(info.frames),
                "duration": float(info.duration),
                "format": str(info.format),
                "subtype": str(info.subtype),
            }
            for (label, path), (_, info) in zip(self.tracks, infos)
        ]

    def __enter__(self) -> "SpeakerTrackRmsDiarizer":
        self.files = [self._sf.SoundFile(str(path), mode="r") for _, path in self.tracks]
        return self

    def __exit__(self, *_exc: Any) -> None:
        for f in self.files:
            try:
                f.close()
            except Exception:
                pass
        self.files = []

    def scores(self, start: float, end: float, margin: float | None = None) -> dict[str, float]:
        import numpy as np  # type: ignore

        margin = self.margin if margin is None else float(margin)
        start = max(0.0, float(start) - margin)
        end = min(self.duration, float(end) + margin)
        if end <= start:
            end = min(self.duration, start + 0.02)

        start_frame = max(0, min(self.frames, int(start * self.sample_rate)))
        end_frame = max(start_frame, min(self.frames, int(end * self.sample_rate)))
        frame_count = max(0, end_frame - start_frame)

        result: dict[str, float] = {}
        for (label, _path), f in zip(self.tracks, self.files):
            f.seek(start_frame)
            data = f.read(frame_count, dtype="float32", always_2d=True)
            if data.ndim == 2 and data.shape[1] > 1:
                data = np.mean(data, axis=1)
            result[label] = round(_rms_db(data), 1)
        return result

    def assignment(self, start: float, end: float, margin: float | None = None) -> dict[str, Any]:
        scores = self.scores(start, end, margin=margin)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        active = [label for label, db in ranked if db >= self.active_db]

        dominant = ranked[0][0] if ranked else None
        is_overlap = len(active) >= 2
        ambiguous_overlap = False
        assigned = dominant
        if is_overlap:
            top = scores[active[0]]
            second = scores[active[1]]
            ambiguous_overlap = (top - second) <= self.overlap_db
            if ambiguous_overlap:
                assigned = "OVERLAP"
        elif not active:
            assigned = None

        return {
            "speaker": assigned,
            "dominant_speaker": dominant,
            "speaker_scores": scores,
            "active_speakers": active,
            "is_overlap": is_overlap,
            "ambiguous_overlap": ambiguous_overlap,
        }


def assign_speakers_from_track_rms(
    segments: list[TranscriptSegment],
    speaker_tracks: list[SpeakerTrack],
    active_db: AutoFloat = "auto",
    overlap_db: AutoFloat = "auto",
    margin: AutoFloat = "auto",
    cache_dir: str | Path | None = None,
) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    """同期済み話者別トラックの音量差から、word/segmentに話者を割り当てる。"""
    normalized_tracks = _as_speaker_tracks(speaker_tracks)
    cache_root = _cache_dir(cache_dir, "rms")
    cache_path: Path | None = None
    cache_key = ""
    if cache_root is not None:
        cache_key = _stable_hash(
            {
                "kind": "speaker_track_rms_assignment_v1",
                "segments": _segments_fingerprint(segments),
                "speaker_tracks": [
                    {"label": label, "file": _file_fingerprint(path)}
                    for label, path in normalized_tracks
                ],
                "active_db": active_db,
                "overlap_db": overlap_db,
                "margin": margin,
            }
        )
        cache_path = cache_root / f"{cache_key}.json"
        if cache_path.exists():
            try:
                cached_segments, cached_meta = _load_transcript_with_meta(cache_path)
                if cached_meta.get("cache_key") == cache_key:
                    cached_meta["cache_hit"] = True
                    cached_meta["cache_path"] = str(cache_path)
                    return cached_segments, cached_meta
            except Exception:
                pass

    with SpeakerTrackRmsDiarizer(
        normalized_tracks,
        active_db=active_db,
        overlap_db=overlap_db,
        margin=margin,
    ) as diarizer:
        for seg in segments:
            seg_assignment = diarizer.assignment(seg.start, seg.end)
            seg.speaker_scores = dict(seg_assignment["speaker_scores"])
            seg.overlap_speakers = list(seg_assignment["active_speakers"])
            seg.is_overlap = bool(seg_assignment["is_overlap"])

            durations: dict[str, float] = {}
            for word in seg.words:
                try:
                    ws = float(word["start"])
                    we = float(word["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                word_assignment = diarizer.assignment(ws, we)
                speaker = word_assignment["speaker"]
                if speaker is not None:
                    word["speaker"] = speaker
                    durations[speaker] = durations.get(speaker, 0.0) + max(0.0, we - ws)
                word["dominant_speaker"] = word_assignment["dominant_speaker"]
                word["speaker_scores"] = word_assignment["speaker_scores"]
                word["overlap_speakers"] = word_assignment["active_speakers"]
                word["overlap"] = word_assignment["is_overlap"]
                word["ambiguous_overlap"] = word_assignment["ambiguous_overlap"]

            if durations:
                seg.primary_speaker = max(durations.items(), key=lambda kv: kv[1])[0]
            else:
                seg.primary_speaker = seg_assignment["speaker"]

        meta = {
            "diarization_method": "speaker_track_rms",
            "speaker_track_active_db": float(diarizer.active_db),
            "speaker_track_overlap_db": float(diarizer.overlap_db),
            "speaker_track_margin": float(diarizer.margin),
            "speaker_track_auto_options": diarizer.auto_options,
            "speaker_tracks": diarizer.info,
            "cache_hit": False,
        }
    if cache_path is not None:
        try:
            save_transcript(
                segments,
                cache_path,
                meta={
                    **meta,
                    "cache_key": cache_key,
                    "cache_kind": "speaker_track_rms_assignment_v1",
                },
            )
            meta["cache_path"] = str(cache_path)
        except Exception as e:
            meta["cache_save_error"] = str(e)
    return segments, meta


def preprocess_speaker_tracks_for_asr(
    speaker_tracks: list[SpeakerTrack],
    output_dir: str | Path,
    block_frames: int = 48000 * 30,
    progress: ProgressCallback | None = None,
    progress_base: float | None = None,
    progress_span: float | None = None,
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    """ASR用に話者別WAVを加工して別フォルダへ保存する。

    元音声は上書きしない。本人マイクらしい区間を残し、漏れ声・無音・DC成分を抑える。
    """
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    tracks = _as_speaker_tracks(speaker_tracks)
    infos = [(label, sf.info(str(path))) for label, path in tracks]
    sample_rate = int(infos[0][1].samplerate)
    for label, info in infos:
        if int(info.samplerate) != sample_rate:
            raise ValueError(f"speaker track sample_rate mismatch: {label}={info.samplerate}, expected={sample_rate}")
    frames = min(int(info.frames) for _, info in infos)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method = "asr_speaker_track_preprocess_v1"
    gate_window_frames = max(1, int(sample_rate * 0.10))  # 100ms
    silence_db = -72.0
    overlap_keep_db = 3.0
    soft_keep_db = 9.0
    bleed_keep_db = 15.0
    target_active_rms = 10 ** (-22.0 / 20.0)
    max_gain = 4.0
    peak_limit = 0.95

    def safe_label(label: str, index: int) -> str:
        value = re.sub(r'[<>:"/\\|?*\s]+', "_", str(label).strip())[:48].strip("._")
        return value or f"TRACK_{index}"

    output_tracks = [
        (label, output_dir / f"{index:02d}_{safe_label(label, index)}.asr.wav")
        for index, (label, _) in enumerate(tracks, start=1)
    ]
    cache_key = _stable_hash(
        {
            "kind": method,
            "source_tracks": [{"label": label, "file": _file_fingerprint(path)} for label, path in tracks],
            "sample_rate": sample_rate,
            "frames": frames,
            "params": {
                "gate_window_frames": gate_window_frames,
                "silence_db": silence_db,
                "overlap_keep_db": overlap_keep_db,
                "soft_keep_db": soft_keep_db,
                "bleed_keep_db": bleed_keep_db,
                "target_active_rms": target_active_rms,
                "max_gain": max_gain,
                "peak_limit": peak_limit,
            },
        }
    )
    meta_path = output_dir / ".asr_preprocess.meta.json"
    if meta_path.exists() and all(path.exists() for _, path in output_tracks):
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached_meta.get("cache_key") == cache_key:
                cached_meta["reused"] = True
                _progress(progress, "ASR用音声前処理再利用", output_dir=str(output_dir), progress_percent=progress_base)
                return output_tracks, cached_meta
        except Exception as e:
            _progress(progress, "ASR用音声前処理キャッシュ確認失敗", cache=str(meta_path), error=str(e))

    def emit_progress(ratio: float) -> None:
        ratio = max(0.0, min(1.0, ratio))
        data: dict[str, Any] = {
            "stage": "ASR前処理",
            "stage_percent": round(ratio * 100.0, 1),
            "written_sec": round((frames * ratio) / sample_rate, 2) if sample_rate else None,
            "duration_sec": round(frames / sample_rate, 2) if sample_rate else None,
        }
        if progress_base is not None and progress_span is not None:
            data["progress_percent"] = round(progress_base + progress_span * ratio, 1)
        _progress(progress, "ASR用音声前処理進捗", **data)

    def read_mono_block(file_obj: Any, n_frames: int) -> Any:
        data = file_obj.read(n_frames, dtype="float32", always_2d=True)
        if data.ndim == 2 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        else:
            data = data[:, 0]
        # DC除去。低域カットほど強くはないが、ASRへの不要なオフセットを抑える。
        return (data - float(np.mean(data))).astype(np.float32, copy=False)

    def db_from_rms(rms: Any) -> Any:
        return 20.0 * np.log10(np.maximum(rms, 1.0e-9))

    def envelopes_for_block(stacked: Any) -> Any:
        track_count, n = stacked.shape
        centers: list[int] = []
        weight_points: list[Any] = []
        for start in range(0, n, gate_window_frames):
            end = min(n, start + gate_window_frames)
            window = stacked[:, start:end]
            rms = np.sqrt(np.mean(window * window, axis=1) + 1.0e-12)
            db = db_from_rms(rms)
            top_db = float(np.max(db))
            if top_db < silence_db:
                weights = np.zeros(track_count, dtype=np.float32)
            else:
                diff = top_db - db
                weights = np.zeros(track_count, dtype=np.float32)
                weights[diff <= overlap_keep_db] = 1.0
                soft = (diff > overlap_keep_db) & (diff <= soft_keep_db)
                weights[soft] = np.power(10.0, -diff[soft] / 20.0).astype(np.float32)
                bleed = (diff > soft_keep_db) & (diff <= bleed_keep_db)
                weights[bleed] = 0.05
                if not np.any(weights > 0):
                    weights[int(np.argmax(db))] = 1.0
            centers.append((start + end) // 2)
            weight_points.append(weights)
        x = np.arange(n, dtype=np.float32)
        center_arr = np.array(centers, dtype=np.float32)
        weights_arr = np.stack(weight_points, axis=0)
        envelopes = np.empty((track_count, n), dtype=np.float32)
        for i in range(track_count):
            envelopes[i] = np.interp(x, center_arr, weights_arr[:, i], left=weights_arr[0, i], right=weights_arr[-1, i])
        return envelopes

    track_count = len(tracks)
    active_sumsq = np.zeros(track_count, dtype=np.float64)
    active_count = np.zeros(track_count, dtype=np.float64)
    peak_abs = np.zeros(track_count, dtype=np.float64)

    files = [sf.SoundFile(str(path), mode="r") for _, path in tracks]
    try:
        scanned = 0
        while scanned < frames:
            n = min(int(block_frames), frames - scanned)
            stacked = np.stack([read_mono_block(f, n) for f in files], axis=0)
            envelopes = envelopes_for_block(stacked)
            processed = stacked * envelopes
            peak_abs = np.maximum(peak_abs, np.max(np.abs(processed), axis=1))
            active = envelopes >= 0.5
            active_sumsq += np.sum(np.where(active, processed * processed, 0.0), axis=1)
            active_count += np.sum(active, axis=1)
            scanned += n
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass

    active_rms = np.sqrt(active_sumsq / np.maximum(active_count, 1.0))
    gain = np.minimum(max_gain, target_active_rms / np.maximum(active_rms, 1.0e-6))
    gain = np.minimum(gain, peak_limit / np.maximum(peak_abs, 1.0e-6))
    gain = np.maximum(gain, 0.05)

    files = [sf.SoundFile(str(path), mode="r") for _, path in tracks]
    outs = [sf.SoundFile(str(path), mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16") for _, path in output_tracks]
    try:
        written = 0
        last_reported = -0.05
        emit_progress(0.0)
        while written < frames:
            n = min(int(block_frames), frames - written)
            stacked = np.stack([read_mono_block(f, n) for f in files], axis=0)
            processed = stacked * envelopes_for_block(stacked)
            for i, out in enumerate(outs):
                out.write(np.clip(processed[i] * float(gain[i]), -1.0, 1.0))
            written += n
            ratio = written / frames if frames else 1.0
            if ratio >= last_reported + 0.05 or ratio >= 0.995:
                emit_progress(ratio)
                last_reported = ratio
    finally:
        for f in files + outs:
            try:
                f.close()
            except Exception:
                pass

    meta = {
        "method": method,
        "cache_key": cache_key,
        "meta_path": str(meta_path),
        "output_dir": str(output_dir),
        "sample_rate": sample_rate,
        "frames": frames,
        "duration": frames / sample_rate if sample_rate else 0.0,
        "gate_window_ms": round(gate_window_frames / sample_rate * 1000.0, 1) if sample_rate else None,
        "silence_db": silence_db,
        "overlap_keep_db": overlap_keep_db,
        "soft_keep_db": soft_keep_db,
        "bleed_keep_db": bleed_keep_db,
        "target_active_rms_db": -22.0,
        "max_gain": max_gain,
        "tracks": [
            {
                "label": label,
                "source": str(src),
                "output": str(dst),
                "active_rms_before_gain": round(float(active_rms[i]), 6),
                "peak_abs_before_gain": round(float(peak_abs[i]), 6),
                "gain": round(float(gain[i]), 6),
            }
            for i, ((label, src), (_, dst)) in enumerate(zip(tracks, output_tracks))
        ],
        "reused": False,
    }
    atomic_write_json(meta_path, meta, encoding="utf-8")
    return output_tracks, meta


def make_mix_from_speaker_tracks(
    speaker_tracks: list[SpeakerTrack],
    output_path: str | Path,
    block_frames: int = 48000 * 30,
    progress: ProgressCallback | None = None,
    progress_base: float | None = None,
    progress_span: float | None = None,
) -> dict[str, Any]:
    """話者別トラックを合成してWhisper用のmono WAVを作る。

    単純加算ではなく、短い窓ごとに本人マイクらしいトラックを優先する。
    これによりマイク被り・位相ズレ・反響がmixへ重複して入る問題を抑える。
    """
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    tracks = _as_speaker_tracks(speaker_tracks)
    infos = [(label, sf.info(str(path))) for label, path in tracks]
    sample_rate = int(infos[0][1].samplerate)
    for label, info in infos:
        if int(info.samplerate) != sample_rate:
            raise ValueError(f"speaker track sample_rate mismatch: {label}={info.samplerate}, expected={sample_rate}")
    frames_list = [int(info.frames) for _, info in infos]
    frames = min(frames_list)
    max_frames = max(frames_list)
    if sample_rate and max_frames - frames > sample_rate:
        diff_sec = (max_frames - frames) / sample_rate
        _progress(
            progress,
            f"警告: 話者別トラックの長さが最大 {diff_sec:.1f}秒ずれています。最短トラックに合わせてmixをトリミングします。",
            diff_sec=round(diff_sec, 2),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mix_cache_key = _stable_hash(
        {
            "kind": "speaker_track_mix_v3",
            "source_tracks": [
                {"label": label, "file": _file_fingerprint(path)}
                for label, path in tracks
            ],
            "sample_rate": sample_rate,
            "frames": frames,
            "channels": 1,
            "subtype": "PCM_16",
            "method": "speaker_aware_soft_gate_v3",
        }
    )
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() and meta_path.exists():
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached_meta.get("cache_key") == mix_cache_key:
                data: dict[str, Any] = {
                    "stage": "mix作成",
                    "stage_percent": 100.0,
                    "written_sec": round(frames / sample_rate, 2) if sample_rate else None,
                    "duration_sec": round(frames / sample_rate, 2) if sample_rate else None,
                }
                if progress_base is not None and progress_span is not None:
                    data["progress_percent"] = round(progress_base + progress_span, 1)
                _progress(progress, "話者別トラックからmix再利用", cache=str(meta_path), output=str(output_path), **data)
                cached_meta["reused"] = True
                return cached_meta
        except Exception as e:
            _progress(progress, "mixキャッシュ確認失敗", cache=str(meta_path), error=str(e))

    def emit_mix_progress(ratio: float) -> None:
        ratio = max(0.0, min(1.0, ratio))
        data: dict[str, Any] = {
            "stage": "mix作成",
            "stage_percent": round(ratio * 100.0, 1),
            "written_sec": round((frames * ratio) / sample_rate, 2) if sample_rate else None,
            "duration_sec": round(frames / sample_rate, 2) if sample_rate else None,
        }
        if progress_base is not None and progress_span is not None:
            data["progress_percent"] = round(progress_base + progress_span * ratio, 1)
        _progress(progress, "話者別トラックからmix作成進捗", **data)

    def read_mono_block(file_obj: Any, n_frames: int) -> Any:
        data = file_obj.read(n_frames, dtype="float32", always_2d=True)
        if data.ndim == 2 and data.shape[1] > 1:
            return np.mean(data, axis=1)
        return data[:, 0]

    gate_window_frames = max(1, int(sample_rate * 0.10))  # 100ms
    silence_db = -72.0
    overlap_keep_db = 3.0
    soft_keep_db = 9.0
    bleed_keep_db = 15.0

    def db_from_rms(rms: Any) -> Any:
        return 20.0 * np.log10(np.maximum(rms, 1.0e-9))

    def mix_speaker_aware_block(chunks: list[Any]) -> Any:
        """話者別マイクの被りを抑えたASR用mixを作る。"""
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0].astype(np.float32, copy=False)

        stacked = np.stack(chunks, axis=0).astype(np.float32, copy=False)
        track_count, n = stacked.shape
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        centers: list[int] = []
        weight_points: list[Any] = []
        for start in range(0, n, gate_window_frames):
            end = min(n, start + gate_window_frames)
            window = stacked[:, start:end]
            rms = np.sqrt(np.mean(window * window, axis=1) + 1.0e-12)
            db = db_from_rms(rms)
            top_db = float(np.max(db))
            if top_db < silence_db:
                weights = np.zeros(track_count, dtype=np.float32)
            else:
                diff = top_db - db
                weights = np.zeros(track_count, dtype=np.float32)
                weights[diff <= overlap_keep_db] = 1.0
                soft = (diff > overlap_keep_db) & (diff <= soft_keep_db)
                weights[soft] = np.power(10.0, -diff[soft] / 20.0).astype(np.float32)
                bleed = (diff > soft_keep_db) & (diff <= bleed_keep_db)
                weights[bleed] = 0.08
                if not np.any(weights > 0):
                    weights[int(np.argmax(db))] = 1.0
            centers.append((start + end) // 2)
            weight_points.append(weights)

        if not centers:
            return np.zeros(n, dtype=np.float32)

        x = np.arange(n, dtype=np.float32)
        center_arr = np.array(centers, dtype=np.float32)
        weights_arr = np.stack(weight_points, axis=0)
        envelopes = np.empty((track_count, n), dtype=np.float32)
        for i in range(track_count):
            envelopes[i] = np.interp(x, center_arr, weights_arr[:, i], left=weights_arr[0, i], right=weights_arr[-1, i])
        return np.sum(stacked * envelopes, axis=0)

    # 1st pass: speaker-aware mixの全体ピークを測り、全区間で一定のscaleを使う。
    # ブロック単位正規化は音量の揺れを作るため避ける。
    files = [sf.SoundFile(str(path), mode="r") for _, path in tracks]
    peak_abs = 0.0
    try:
        scanned = 0
        while scanned < frames:
            n = min(int(block_frames), frames - scanned)
            chunks = [read_mono_block(f, n) for f in files]
            mix = mix_speaker_aware_block(chunks)
            if mix.size:
                peak_abs = max(peak_abs, float(np.max(np.abs(mix))))
            scanned += n
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass

    peak_limit = 0.98
    mix_scale = min(1.0, peak_limit / peak_abs) if peak_abs > 0.0 else 1.0

    files = [sf.SoundFile(str(path), mode="r") for _, path in tracks]
    try:
        with sf.SoundFile(str(output_path), mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16") as out:
            written = 0
            last_reported = -0.05
            emit_mix_progress(0.0)
            while written < frames:
                n = min(int(block_frames), frames - written)
                chunks = [read_mono_block(f, n) for f in files]
                mix = mix_speaker_aware_block(chunks)
                out.write(np.clip(mix * mix_scale, -1.0, 1.0))
                written += n
                ratio = written / frames if frames else 1.0
                if ratio >= last_reported + 0.05 or ratio >= 0.995:
                    emit_mix_progress(ratio)
                    last_reported = ratio
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass

    meta = {
        "path": str(output_path),
        "sample_rate": sample_rate,
        "channels": 1,
        "frames": frames,
        "duration": frames / sample_rate if sample_rate else 0.0,
        "source_tracks": [
            {"label": label, "path": str(path), "channels": int(info.channels), "frames": int(info.frames)}
            for (label, path), (_, info) in zip(tracks, infos)
        ],
        "cache_key": mix_cache_key,
        "meta_path": str(meta_path),
        "mix_method": "speaker_aware_soft_gate_v3",
        "mix_gate_window_ms": round(gate_window_frames / sample_rate * 1000.0, 1) if sample_rate else None,
        "mix_gate_silence_db": silence_db,
        "mix_gate_overlap_keep_db": overlap_keep_db,
        "mix_gate_soft_keep_db": soft_keep_db,
        "mix_gate_bleed_keep_db": bleed_keep_db,
        "mix_peak_abs_before_scale": round(float(peak_abs), 6),
        "mix_scale": round(float(mix_scale), 6),
        "reused": False,
    }
    try:
        atomic_write_json(meta_path, meta, encoding="utf-8")
    except Exception as e:
        meta["meta_save_error"] = str(e)
    return meta


def transcribe_with_speaker_tracks(
    audio_path: str | Path,
    output_path: str | Path,
    speaker_tracks: list[SpeakerTrack],
    model_size: str = "large-v3",
    language: str = "ja",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    active_db: AutoFloat = "auto",
    overlap_db: AutoFloat = "auto",
    margin: AutoFloat = "auto",
    extra_meta: dict[str, Any] | None = None,
) -> list[TranscriptSegment]:
    """Whisper文字起こし + 同期済み話者別トラックRMSで話者割当。"""
    cache_root = Path(output_path).parent / ".transcribe_cache"
    segments = _faster_whisper_transcribe_only(
        audio_path,
        model_size=model_size,
        language=language,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        cache_dir=cache_root,
    )
    segments, diarization_meta = assign_speakers_from_track_rms(
        segments,
        speaker_tracks=speaker_tracks,
        active_db=active_db,
        overlap_db=overlap_db,
        margin=margin,
        cache_dir=cache_root,
    )
    save_transcript(
        segments,
        output_path,
        meta={
            "transcriber": "faster-whisper+speaker-track-rms",
            "model": model_size,
            "language": language,
            "unit": "word+speaker",
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
            **diarization_meta,
            **(extra_meta or {}),
        },
    )
    return segments


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _asr_score(segments: list[TranscriptSegment]) -> float:
    """候補比較用の粗いASRスコア。大きいほど良い。"""
    if not segments:
        return -99.0
    scores: list[float] = []
    for seg in segments:
        avg_logprob = float(seg.asr.get("avg_logprob", -1.0))
        no_speech_prob = float(seg.asr.get("no_speech_prob", 0.0))
        scores.append(avg_logprob - no_speech_prob)
    return sum(scores) / len(scores)


def _join_segment_text(segments: list[TranscriptSegment]) -> str:
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()


def _segment_text_in_window(seg: TranscriptSegment, start: float, end: float) -> str:
    if not seg.words:
        return seg.text.strip()
    # リードイン: セグメント先頭付近の語が窓直前で始まる場合のみ適用する。
    # 長いセグメントの末尾語が次の窓に漏れ込むのを防ぐため、
    # 「窓開始から seg.start までの距離が 1.0s 以内」の語に限定する。
    lead = 0.15
    seg_start = float(seg.start)
    words: list[str] = []
    for word in seg.words:
        try:
            ws = float(word["start"])
            we = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        mid = (ws + we) / 2.0
        if mid < start:
            if (start - mid) <= lead and (start - seg_start) <= 1.0:
                pass  # セグメント冒頭の境界語のみリードイン適用
            else:
                continue
        elif mid > end:
            continue
        text = str(word.get("word", "")).strip()
        if text:
            words.append(text)
    return "".join(words).strip()


def _candidate_from_segments(
    source: str,
    speaker: str | None,
    segments: list[TranscriptSegment],
    clip_start: float | None = None,
    clip_end: float | None = None,
) -> dict[str, Any] | None:
    if clip_start is not None and clip_end is not None:
        text = " ".join(
            _segment_text_in_window(seg, clip_start, clip_end)
            for seg in segments
            if _overlap_seconds(clip_start, clip_end, seg.start, seg.end) > 0
        ).strip()
    else:
        text = _join_segment_text(segments)
    if not text:
        return None
    return {
        "source": source,
        "speaker": speaker,
        "start": clip_start if clip_start is not None else min(float(s.start) for s in segments),
        "end": clip_end if clip_end is not None else max(float(s.end) for s in segments),
        "text": text,
        "score": round(_asr_score(segments), 3),
    }


def _overlapping_segments(
    segments: list[TranscriptSegment],
    start: float,
    end: float,
    min_overlap: float = 0.08,
) -> list[TranscriptSegment]:
    result: list[TranscriptSegment] = []
    for seg in segments:
        ov = _overlap_seconds(start, end, seg.start, seg.end)
        if ov >= min_overlap:
            result.append(seg)
    return result


def _text_similarity(a: str, b: str) -> float:
    a = "".join(a.split())
    b = "".join(b.split())
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _normalize_candidate_text(text: str) -> str:
    return "".join(str(text).split())


def _normalize_repetition_text(text: str) -> str:
    """反復ハルシネーション検出用の正規化。短い相づちは過検出しない。"""
    normalized = re.sub(r"[\s、。．，,.!?！？「」『』（）()［］\[\]【】…・:：;；\"'`]+", "", str(text))
    normalized = normalized.replace("ぁ", "あ").replace("ぃ", "い").replace("ぅ", "う").replace("ぇ", "え").replace("ぉ", "お")
    return normalized


def _get_sudachi_tokenizer() -> tuple[Any | None, Any | None]:
    """SudachiPyが入っていればtokenizerを返す。未導入ならNoneで静かに無効化する。"""
    global _SUDACHI_TOKENIZER_CACHE, _SUDACHI_SPLIT_MODE, _SUDACHI_AVAILABLE
    if _SUDACHI_AVAILABLE is False:
        return None, None
    if _SUDACHI_TOKENIZER_CACHE is not None:
        return _SUDACHI_TOKENIZER_CACHE, _SUDACHI_SPLIT_MODE
    try:
        from sudachipy import dictionary, tokenizer  # type: ignore

        _SUDACHI_TOKENIZER_CACHE = dictionary.Dictionary().create()
        _SUDACHI_SPLIT_MODE = tokenizer.Tokenizer.SplitMode.C
        _SUDACHI_AVAILABLE = True
        return _SUDACHI_TOKENIZER_CACHE, _SUDACHI_SPLIT_MODE
    except Exception:
        _SUDACHI_AVAILABLE = False
        return None, None


def _sudachi_noise_flags(text: str) -> list[str]:
    """SudachiPy由来の自動ノイズ判定は現在使わない。

    未知語率・助詞/助動詞率は、番組内の専門用語・短い発話・話し言葉で誤検出しやすい。
    SudachiPyは今後、用語候補抽出や人間確認用の補助に限定する。
    """
    return []


def _katakana_to_hiragana_for_readability(text: str) -> str:
    """カタカナ音写ノイズを最低限読める形にするため、カタカナをひらがなへ寄せる。"""
    out: list[str] = []
    for ch in str(text):
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _detect_mix_repetition_hallucinations(
    mix_segments: list[TranscriptSegment],
) -> tuple[set[str], dict[str, Any]]:
    """mix ASR が同じ短文を異常反復した箇所を検出する。

    削除はしない。fusion時に話者別トラック候補へ逃がす/要確認にするための印だけ付ける。
    """
    total = len(mix_segments)
    if total < 8:
        return set(), {"suspected_segments": 0, "suspected_texts": []}

    counts: dict[str, int] = {}
    run_max: dict[str, int] = {}
    prev = ""
    run_len = 0
    for seg in mix_segments:
        text = _normalize_repetition_text(seg.text)
        if len(text) < 5:
            prev = text
            run_len = 1 if text else 0
            continue
        counts[text] = counts.get(text, 0) + 1
        if text == prev:
            run_len += 1
        else:
            prev = text
            run_len = 1
        run_max[text] = max(run_max.get(text, 0), run_len)

    min_count = max(8, int(total * 0.08))
    suspicious_texts = {
        text
        for text, count in counts.items()
        if count >= min_count or run_max.get(text, 0) >= 6
    }
    flagged: set[str] = set()
    for seg in mix_segments:
        text = _normalize_repetition_text(seg.text)
        if text in suspicious_texts:
            flagged.add(seg.segment_id)
            seg.asr["suspected_repetition_hallucination"] = True
            seg.asr["repetition_key"] = text
            seg.asr["repetition_count"] = counts.get(text, 0)
            seg.asr["repetition_max_run"] = run_max.get(text, 0)

    top = sorted(
        (
            {"text": text, "count": counts.get(text, 0), "max_run": run_max.get(text, 0)}
            for text in suspicious_texts
        ),
        key=lambda item: (-int(item["max_run"]), -int(item["count"]), str(item["text"])),
    )[:10]
    return flagged, {"suspected_segments": len(flagged), "suspected_texts": top}


def _percentile(values: list[float], percentile: float, default: float) -> float:
    if not values:
        return default
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _estimate_quality_thresholds(segments: list[TranscriptSegment]) -> dict[str, float]:
    """収録ごとのASR分布から安全側の品質閾値を推定する。"""
    avg_values: list[float] = []
    comp_values: list[float] = []
    dur_values: list[float] = []
    for seg in segments:
        try:
            avg = float(seg.asr.get("avg_logprob"))
            if -20.0 < avg < 1.0:
                avg_values.append(avg)
        except (TypeError, ValueError):
            pass
        try:
            comp = float(seg.asr.get("compression_ratio"))
            if 0.0 < comp < 100.0:
                comp_values.append(comp)
        except (TypeError, ValueError):
            pass
        dur = max(0.0, float(seg.end) - float(seg.start))
        if dur > 0.0:
            dur_values.append(dur)

    # 低logprobは収録差が出るため、下位10%よりさらに少し悪いものを強い疑いにする。
    avg_p10 = _percentile(avg_values, 10, -2.0)
    low_avg_logprob = _clamp(avg_p10 - 0.35, -4.5, -1.4)

    # compressionは話者差よりASRループに強く反応するので、上振れを見つつ上限を低めに保つ。
    comp_p95 = _percentile(comp_values, 95, 2.4)
    high_compression_ratio = _clamp(comp_p95 + 0.25, 2.4, 3.2)

    # 短断片は収録・話速差があるため下位分布を参考にする。ただし読みやすさのため範囲は制限。
    dur_p10 = _percentile(dur_values, 10, 0.5)
    short_fragment_seconds = _clamp(dur_p10, 0.25, 0.65)

    return {
        "low_avg_logprob": round(float(low_avg_logprob), 4),
        "high_compression_ratio": round(float(high_compression_ratio), 4),
        "short_fragment_seconds": round(float(short_fragment_seconds), 4),
        "latin_long_run_chars": 16.0,
        "local_repetition_char_run": 8.0,
        "katakana_phonetic_min_chars": 12.0,
        "katakana_phonetic_ratio": 0.82,
        "sudachi_enabled": 1.0 if _get_sudachi_tokenizer()[0] is not None else 0.0,
    }


def _quality_flags_for_text(
    text: str,
    asr: dict[str, Any] | None = None,
    duration: float | None = None,
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """日本語会話ASRとして怪しい特徴を返す。自動削除ではなく差し替え/要確認に使う。"""
    asr = asr or {}
    thresholds = thresholds or {}
    low_avg_threshold = float(thresholds.get("low_avg_logprob", -2.0))
    high_comp_threshold = float(thresholds.get("high_compression_ratio", 2.4))
    short_fragment_seconds = float(thresholds.get("short_fragment_seconds", 0.5))
    latin_long_run_chars = int(thresholds.get("latin_long_run_chars", 16.0))
    local_repetition_char_run = int(thresholds.get("local_repetition_char_run", 8.0))
    katakana_phonetic_min_chars = int(thresholds.get("katakana_phonetic_min_chars", 12.0))
    katakana_phonetic_ratio = float(thresholds.get("katakana_phonetic_ratio", 0.82))
    flags: list[str] = []
    raw = str(text or "").strip()
    norm = _normalize_repetition_text(raw)
    if not raw:
        return flags

    if "\ufffd" in raw or "�" in raw:
        flags.append("mojibake_or_replacement_char")
    if re.search(r"[\u0600-\u06ff\u0400-\u04ff\uac00-\ud7af]", raw):
        flags.append("unexpected_foreign_script")
    if re.search(rf"[A-Za-zÀ-ž]{{{latin_long_run_chars},}}", raw):
        flags.append("long_latin_noise")
    latin_like = len(re.findall(r"[A-Za-zÀ-ž]", raw))
    jp_like = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", raw))
    if len(raw) >= 18 and latin_like >= 10 and latin_like > jp_like * 1.5:
        flags.append("latin_heavy_noise")

    # 日本語の助詞・活用までカタカナ化された音写ノイズ。
    # 「セキュリティ」「オープンソース」のような正しい外来語だけでは反応しにくいよう、
    # カタカナ比率に加えて日本語機能語のカタカナ表記を要求する。
    jp_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", raw)
    if len(jp_chars) >= katakana_phonetic_min_chars:
        katakana_chars = re.findall(r"[ァ-ヴー]", raw)
        hira_kanji_chars = re.findall(r"[ぁ-ん\u3400-\u9fff]", raw)
        kata_ratio = len(katakana_chars) / max(1, len(jp_chars))
        phonetic_markers = re.findall(
            r"(デス|マス|デシタ|デショ|ナイ|ナル|ナッ|ソレ|コレ|アレ|コノ|ソノ|アノ|"
            r"トカ|テイウ|ッテ|ケド|カラ|ヨネ|デモ|ダカラ|ジャナイ|モンネ|ミタイ|"
            r"ワケ|コト|ヒト|モウ|ナン|ダレ|サン|キタ|ミタ|イワナイ|オモ|アル|イル)",
            raw,
        )
        if kata_ratio >= katakana_phonetic_ratio and len(hira_kanji_chars) <= 3 and len(phonetic_markers) >= 2:
            flags.append("katakana_phonetic_noise")

    if re.search(rf"([ぁ-んァ-ンーA-Za-z])\1{{{max(1, local_repetition_char_run - 1)},}}", norm):
        flags.append("local_repetition")
    if re.search(rf"(.{{1,3}})\1{{{max(1, local_repetition_char_run - 1)},}}", norm):
        flags.append("local_repetition")
    tokens = re.findall(r"[ぁ-んァ-ンーA-Za-z]{1,3}", raw)
    if len(tokens) >= 12:
        most_common_count = max(collections.Counter(tokens).values()) if tokens else 0
        if most_common_count >= 8 and most_common_count / len(tokens) >= 0.45:
            flags.append("local_repetition")
    if len(norm) >= 30:
        chars = [c for c in norm if not c.isspace()]
        if chars:
            top_ratio = collections.Counter(chars).most_common(1)[0][1] / len(chars)
            if top_ratio >= 0.45:
                flags.append("local_repetition")

    try:
        avg_logprob = float(asr.get("avg_logprob", 0.0))
        if avg_logprob < low_avg_threshold:
            flags.append("low_avg_logprob")
    except (TypeError, ValueError):
        pass
    try:
        compression_ratio = float(asr.get("compression_ratio", 0.0))
        if compression_ratio > high_comp_threshold:
            flags.append("high_compression_ratio")
    except (TypeError, ValueError):
        pass

    if duration is not None:
        text_len = len(norm)
        whitelist = {"うん", "はい", "え", "あ", "お", "ん", "はいはい"}
        if duration < short_fragment_seconds and text_len <= 2 and norm not in whitelist:
            flags.append("short_fragment_noise")

    # 既知の幻覚パターン（Whisperが学習データから生成しがちな固定文字列・記号）
    # YouTube/TVの定型句
    hallucination_youtube = {
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございました!",
        "ご視聴ありがとうございました。",
        "チャンネル登録お願いします",
        "高評価チャンネル登録お願いします",
        "次回もお楽しみに",
        "また次回お会いしましょう",
        "ありがとうございました",
    }
    if norm in hallucination_youtube or raw in hallucination_youtube:
        flags.append("hallucination_youtube_subtitle")
    # 日本語Whisperの既知の固有名詞幻覚（学習データの偏り由来）
    # 短い区間で単独で出てくる場合のみ幻覚扱い（長文の中に混じる場合は別扱い）
    hallucination_japanese_proper = {
        "深井",
        "ヤンヤン",
        "ヤンヤ",
        "ヤンヤヤン",
        "アンアン",
        "なんなん",
        "ニャンニャン",
    }
    norm_no_punct = re.sub(r"[、。!?\s\.,「」『』]", "", raw)
    if duration is not None and duration < 2.0 and norm_no_punct in hallucination_japanese_proper:
        flags.append("hallucination_japanese_proper_noun")
    # 顔文字/記号系の幻覚（"ヽ ノ" "ヽノ" "( ´ ▽ ` )" 等）
    if re.fullmatch(r"[ヽノ\s]+", raw) or re.fullmatch(r"[\s\(\)（）\[\]ノヽ・´▽\^_;:]+", raw):
        flags.append("hallucination_symbol_noise")
    # 単独カナ1〜2文字 + 0.3秒未満（"ア" "ニ" "シ" 等）
    if duration is not None and duration < 0.3 and re.fullmatch(r"[ァ-ヴー]{1,2}", raw):
        flags.append("hallucination_single_kana")
    # 数字のみ / 数字+短い英字 の幻覚（"112 ." "1000" "111 len" など）
    # 短いセグメントで数字主体のテキストはほぼ確実に幻覚
    digit_chars = len(re.findall(r"\d", raw))
    jp_chars_count = len(re.findall(r"[぀-ヿ㐀-鿿]", raw))
    if duration is not None and duration < 1.0 and digit_chars >= 2 and jp_chars_count == 0:
        if re.fullmatch(r"[\d\s\.\-,a-zA-Z]+", raw):
            flags.append("hallucination_digit_noise")

    return list(dict.fromkeys(flags))


def _collapse_intra_segment_repetition(text: str) -> tuple[str, bool]:
    """セグメントテキスト内で同じフレーズが繰り返されているパターンを1回分に圧縮する。

    例: 'あ、これはね、ライトニング あ、これはね、ライトニング' → 'あ、これはね、ライトニング'

    安全のために以下の条件を満たす場合のみ圧縮:
    - 全文がパターン (X)(セパレータX)+ で表現できる
    - 反復単位の長さが4文字以上（「うんうん」「はいはい」を誤って圧縮しない）
    """
    raw = str(text or "").strip()
    if len(raw) < 8:
        return raw, False
    # 区切り文字を含めて単位の繰り返しにマッチ
    m = re.fullmatch(r"(.+?)(?:[、。!?\s\.,]+\1)+[、。!?\s\.,]*", raw)
    if not m:
        return raw, False
    unit = m.group(1).strip()
    if len(unit) < 4:
        return raw, False
    return unit, True


def _candidate_quality_score(candidate: dict[str, Any], thresholds: dict[str, float] | None = None) -> float:
    text = str(candidate.get("text", "")).strip()
    flags = _quality_flags_for_text(text, thresholds=thresholds)
    if (
        not text
        or "local_repetition" in flags
        or "katakana_phonetic_noise" in flags
        or "unexpected_foreign_script" in flags
        or "mojibake_or_replacement_char" in flags
    ):
        return -999.0
    norm_len = len(_normalize_candidate_text(text))
    if norm_len < 2:
        return -999.0
    score = float(candidate.get("fusion_score", 0.0) or 0.0) + float(candidate.get("score", -3.0) or -3.0)
    score += min(norm_len, 80) / 80.0
    if flags:
        score -= 2.0
    return score


def _best_quality_replacement_candidate(seg: TranscriptSegment, thresholds: dict[str, float] | None = None) -> dict[str, Any] | None:
    candidates = [c for c in seg.candidates if c.get("source") != "mix"]
    candidates = sorted(candidates, key=lambda c: _candidate_quality_score(c, thresholds), reverse=True)
    if not candidates or _candidate_quality_score(candidates[0], thresholds) < -2.5:
        return None
    return candidates[0]


def apply_transcript_quality_postprocessing(segments: list[TranscriptSegment]) -> dict[str, Any]:
    """品質検査・安全な差し替え・clean/simple非表示フラグ付けを行う。"""
    thresholds = _estimate_quality_thresholds(segments)
    summary = {
        "segments": len(segments),
        "auto_thresholds": thresholds,
        "quality_flagged": 0,
        "text_replaced": 0,
        "suppressed_in_text_outputs": 0,
        "short_fragment_noise": 0,
        "katakana_phonetic_noise": 0,
        "mojibake_or_foreign": 0,
        "local_repetition": 0,
        "low_asr_quality": 0,
        "mix_track_text_mismatch": 0,
        "overlap_segments": 0,
        "hallucination_youtube": 0,
        "hallucination_symbol": 0,
        "hallucination_single_kana": 0,
        "hallucination_digit": 0,
        "hallucination_japanese_proper": 0,
        "hallucination_repeated_phrase": 0,
        "intra_segment_repetition": 0,
        "unintelligible_overlap": 0,
        "reason_counts": {},
    }

    for seg in segments:
        if seg.primary_speaker is None:
            seg.primary_speaker = "不明"
            seg.confidence_reasons.append("missing_speaker_normalized")
        if seg.primary_speaker == "OVERLAP" or seg.is_overlap:
            summary["overlap_segments"] += 1
        if "mix_track_text_mismatch" in seg.confidence_reasons:
            summary["mix_track_text_mismatch"] += 1

        # セグメント内反復の圧縮（"X X" → "X"）
        collapsed, was_collapsed = _collapse_intra_segment_repetition(seg.text)
        if was_collapsed:
            seg.asr["intra_repetition_collapsed_from"] = seg.text
            seg.text = collapsed
            if "intra_segment_repetition" not in seg.confidence_reasons:
                seg.confidence_reasons.append("intra_segment_repetition")
            summary["intra_segment_repetition"] += 1

        duration = max(0.0, float(seg.end) - float(seg.start))
        flags = _quality_flags_for_text(seg.text, seg.asr, duration, thresholds)
        if flags:
            summary["quality_flagged"] += 1
            for flag in flags:
                reason = f"quality_{flag}"
                if reason not in seg.confidence_reasons:
                    seg.confidence_reasons.append(reason)
            if seg.confidence == "high":
                seg.confidence = "medium"

        if any(f in flags for f in ("mojibake_or_replacement_char", "unexpected_foreign_script", "long_latin_noise", "latin_heavy_noise")):
            summary["mojibake_or_foreign"] += 1
        if "local_repetition" in flags:
            summary["local_repetition"] += 1
        if "short_fragment_noise" in flags:
            summary["short_fragment_noise"] += 1
        if "katakana_phonetic_noise" in flags:
            summary["katakana_phonetic_noise"] += 1
        if "low_avg_logprob" in flags or "high_compression_ratio" in flags:
            summary["low_asr_quality"] += 1
        if "hallucination_youtube_subtitle" in flags:
            summary["hallucination_youtube"] += 1
        if "hallucination_symbol_noise" in flags:
            summary["hallucination_symbol"] += 1
        if "hallucination_single_kana" in flags:
            summary["hallucination_single_kana"] += 1
        if "hallucination_digit_noise" in flags:
            summary["hallucination_digit"] += 1
        if "hallucination_japanese_proper_noun" in flags:
            summary["hallucination_japanese_proper"] += 1
        if "hallucination_repeated_phrase" in seg.confidence_reasons:
            summary["hallucination_repeated_phrase"] += 1

        hallucination_flags = {
            "hallucination_youtube_subtitle",
            "hallucination_symbol_noise",
            "hallucination_single_kana",
            "hallucination_digit_noise",
            "hallucination_japanese_proper_noun",
            "hallucination_repeated_phrase",
        }
        is_hallucination = bool(set(flags) & hallucination_flags)
        replacement_serious = bool(
            "local_repetition" in flags
            or "short_fragment_noise" in flags
            or "katakana_phonetic_noise" in flags
            or "unexpected_foreign_script" in flags
            or ("mojibake_or_replacement_char" in flags and "low_avg_logprob" in flags)
            or ("long_latin_noise" in flags and "low_avg_logprob" in flags)
            or "high_compression_ratio" in flags
            or is_hallucination
        )
        suppress_serious = bool(
            "local_repetition" in flags
            or "short_fragment_noise" in flags
            or "unexpected_foreign_script" in flags
            or ("mojibake_or_replacement_char" in flags and "low_avg_logprob" in flags)
            or ("long_latin_noise" in flags and "low_avg_logprob" in flags)
            or "high_compression_ratio" in flags
            or is_hallucination
        )
        replacement = None
        if flags and ("mix_track_text_mismatch" in seg.confidence_reasons or replacement_serious):
            replacement = _best_quality_replacement_candidate(seg, thresholds)
            if replacement is not None:
                old_text = seg.text
                seg.text = str(replacement.get("text", "")).strip()
                seg.asr["quality_replaced_from"] = old_text
                seg.asr["quality_replaced_with_source"] = replacement.get("source")
                seg.asr["quality_replaced_with_speaker"] = replacement.get("speaker")
                seg.confidence_reasons.append("quality_replaced_with_speaker_track_candidate")
                summary["text_replaced"] += 1
                # 差し替え後も怪しければ非表示へ
                flags = _quality_flags_for_text(seg.text, seg.asr, duration, thresholds)
            elif "katakana_phonetic_noise" in flags:
                old_text = seg.text
                readable_text = _katakana_to_hiragana_for_readability(old_text)
                if readable_text and readable_text != old_text:
                    seg.text = readable_text
                    seg.asr["readability_rewritten_from"] = old_text
                    seg.asr["readability_rewrite_method"] = "katakana_to_hiragana"
                    if "readability_katakana_to_hiragana" not in seg.confidence_reasons:
                        seg.confidence_reasons.append("readability_katakana_to_hiragana")
                    summary["text_replaced"] += 1
                    flags = _quality_flags_for_text(seg.text, seg.asr, duration, thresholds)

        if suppress_serious and replacement is None and _best_quality_replacement_candidate(seg, thresholds) is None:
            seg.asr["suppress_in_text_outputs"] = True
            seg.confidence = "low"
            if "suppressed_in_clean_simple" not in seg.confidence_reasons:
                seg.confidence_reasons.append("suppressed_in_clean_simple")
            summary["suppressed_in_text_outputs"] += 1

        # 長尺OVERLAPの聞き取り不能化: 5秒以上のOVERLAP区間で全候補が低品質ならテキストを置換
        if (seg.primary_speaker == "OVERLAP" or seg.is_overlap) and duration >= 5.0:
            all_candidates_bad = True
            for cand in seg.candidates or []:
                cand_score = _candidate_quality_score(cand, thresholds)
                if cand_score >= -1.5:
                    all_candidates_bad = False
                    break
            try:
                avg_logprob = float(seg.asr.get("avg_logprob", 0.0))
            except (TypeError, ValueError):
                avg_logprob = 0.0
            if all_candidates_bad and avg_logprob < -1.5:
                seg.asr["unintelligible_replaced_from"] = seg.text
                seg.text = "[聞き取り不能 — 要手動確認]"
                seg.confidence = "low"
                if "unintelligible_overlap" not in seg.confidence_reasons:
                    seg.confidence_reasons.append("unintelligible_overlap")
                summary["unintelligible_overlap"] += 1

    # クロスセグメント反復検出: 短い同じフレーズが短時間に複数回出るのは幻覚の典型
    _detect_cross_segment_repetition(segments, summary)

    reason_counts = collections.Counter(r for seg in segments for r in seg.confidence_reasons)
    summary["reason_counts"] = dict(reason_counts.most_common(30))
    return summary


def _detect_cross_segment_repetition(segments: list[TranscriptSegment], summary: dict[str, Any]) -> None:
    """同じ短い語句が短時間に繰り返し出る場合、幻覚として一括suppressする。

    ヤンヤン・深井・なんなん などWhisperが暴走して同じ語を吐き続けるパターンを検出する。
    短い (10文字以下) かつ短時間 (3秒以下) のセグメントだけを対象にする。
    """
    WINDOW_SECONDS = 30.0
    MIN_OCCURRENCES = 4
    MAX_TEXT_LEN = 10
    MAX_DURATION = 3.0

    short_segs: list[tuple[int, TranscriptSegment, str]] = []
    for idx, seg in enumerate(segments):
        text = re.sub(r"[、。!?\s\.,「」『』]", "", seg.text or "")
        if not text or len(text) > MAX_TEXT_LEN:
            continue
        duration = max(0.0, float(seg.end) - float(seg.start))
        if duration > MAX_DURATION:
            continue
        short_segs.append((idx, seg, text))

    # 各セグメントごとに、前後WINDOW_SECONDS内で同じ短いテキストが何回出るかを数える
    flagged_indices: set[int] = set()
    for i, (idx_i, seg_i, text_i) in enumerate(short_segs):
        center = (float(seg_i.start) + float(seg_i.end)) / 2.0
        window_lo = center - WINDOW_SECONDS / 2.0
        window_hi = center + WINDOW_SECONDS / 2.0
        same_indices: list[int] = []
        for idx_j, seg_j, text_j in short_segs:
            seg_center = (float(seg_j.start) + float(seg_j.end)) / 2.0
            if seg_center < window_lo or seg_center > window_hi:
                continue
            if text_j == text_i:
                same_indices.append(idx_j)
        if len(same_indices) >= MIN_OCCURRENCES:
            flagged_indices.update(same_indices)

    for idx in flagged_indices:
        seg = segments[idx]
        seg.asr["suppress_in_text_outputs"] = True
        seg.confidence = "low"
        if "hallucination_repeated_phrase" not in seg.confidence_reasons:
            seg.confidence_reasons.append("hallucination_repeated_phrase")
        if "suppressed_in_clean_simple" not in seg.confidence_reasons:
            seg.confidence_reasons.append("suppressed_in_clean_simple")
    summary["hallucination_repeated_phrase"] = len(flagged_indices)


def build_transcript_quality_summary(segments: list[TranscriptSegment]) -> dict[str, Any]:
    """保存済みsegmentsから品質サマリを作る。"""
    thresholds = _estimate_quality_thresholds(segments)
    summary = {
        "segments": len(segments),
        "auto_thresholds": thresholds,
        "low_confidence": sum(seg.confidence == "low" for seg in segments),
        "medium_confidence": sum(seg.confidence == "medium" for seg in segments),
        "overlap_segments": sum(seg.primary_speaker == "OVERLAP" or seg.is_overlap for seg in segments),
        "suppressed_in_text_outputs": sum(bool(seg.asr.get("suppress_in_text_outputs")) for seg in segments),
        "text_replaced": sum("quality_replaced_with_speaker_track_candidate" in seg.confidence_reasons for seg in segments),
        "asr_retry_attempted": sum(bool(seg.asr.get("retry_attempted")) for seg in segments),
        "asr_retry_replaced": sum(bool(seg.asr.get("retry_replaced")) for seg in segments),
        "mojibake_or_foreign": 0,
        "local_repetition": 0,
        "short_fragment_noise": 0,
        "katakana_phonetic_noise": 0,
        "low_asr_quality": 0,
        "mix_track_text_mismatch": sum("mix_track_text_mismatch" in seg.confidence_reasons for seg in segments),
        "reason_counts": {},
        "examples": [],
    }
    examples: list[dict[str, Any]] = []
    for seg in segments:
        duration = max(0.0, float(seg.end) - float(seg.start))
        flags = _quality_flags_for_text(seg.text, seg.asr, duration, thresholds)
        if any(f in flags for f in ("mojibake_or_replacement_char", "unexpected_foreign_script", "long_latin_noise", "latin_heavy_noise")):
            summary["mojibake_or_foreign"] += 1
        if "local_repetition" in flags:
            summary["local_repetition"] += 1
        if "short_fragment_noise" in flags:
            summary["short_fragment_noise"] += 1
        if "katakana_phonetic_noise" in flags:
            summary["katakana_phonetic_noise"] += 1
        if "low_avg_logprob" in flags or "high_compression_ratio" in flags:
            summary["low_asr_quality"] += 1
        if flags and len(examples) < 20:
            examples.append({
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "speaker": seg.primary_speaker or "不明",
                "flags": flags,
                "text": seg.text[:160],
            })
    reason_counts = collections.Counter(r for seg in segments for r in seg.confidence_reasons)
    summary["reason_counts"] = dict(reason_counts.most_common(30))
    summary["examples"] = examples
    return summary


def transcript_quality_summary_to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 文字起こし品質サマリ",
        "",
        f"- セグメント数: {summary.get('segments', 0)}",
        f"- low信頼: {summary.get('low_confidence', 0)}",
        f"- medium信頼: {summary.get('medium_confidence', 0)}",
        f"- OVERLAP: {summary.get('overlap_segments', 0)}",
        f"- 文字化け/外国語ノイズ疑い: {summary.get('mojibake_or_foreign', 0)}",
        f"- 局所反復疑い: {summary.get('local_repetition', 0)}",
        f"- 短断片ノイズ疑い: {summary.get('short_fragment_noise', 0)}",
        f"- カタカナ音写ノイズ疑い: {summary.get('katakana_phonetic_noise', 0)}",
        f"- 低ASR品質疑い: {summary.get('low_asr_quality', 0)}",
        f"- ASR retry実行: {summary.get('asr_retry_attempted', 0)}",
        f"- ASR retry差し替え: {summary.get('asr_retry_replaced', 0)}",
        f"- 自動差し替え: {summary.get('text_replaced', 0)}",
        f"- clean/simple非表示: {summary.get('suppressed_in_text_outputs', 0)}",
        "",
        "## 自動推定閾値",
    ]
    thresholds = summary.get("auto_thresholds") or {}
    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## 理由上位",
    ]
    reasons = summary.get("reason_counts") or {}
    if isinstance(reasons, dict):
        for reason, count in list(reasons.items())[:15]:
            lines.append(f"- {reason}: {count}")
    lines += ["", "## 例"]
    examples = summary.get("examples") or []
    if isinstance(examples, list):
        for ex in examples[:20]:
            lines.append(f"- {ex.get('start')} - {ex.get('end')}｜{ex.get('speaker')}｜{', '.join(ex.get('flags', []))}: {ex.get('text')}")
    return "\n".join(lines) + "\n"


def _choose_fused_text(
    mix_seg: TranscriptSegment,
    assigned_speaker: str | None,
    candidates: list[dict[str, Any]],
) -> tuple[str, str, list[str]]:
    """mix/話者別候補から本文と信頼度を決める。怪しい理由も返す。

    話者が確定しているセグメントでは、話者トラックのテキストを優先する。
    話者トラックは相手の声が入らないため、その話者の発話精度が高い。
    mix へのフォールバックはトラック品質が著しく低い場合のみ。
    """
    reasons: list[str] = []
    mix_text = mix_seg.text.strip()
    if mix_seg.is_overlap:
        reasons.append("overlap")

    speaker_candidates = [
        c for c in candidates
        if c.get("source") != "mix" and (assigned_speaker is None or c.get("speaker") == assigned_speaker)
    ]
    best_track = max(
        speaker_candidates,
        key=lambda c: float(c.get("score", -99.0)),
        default=None,
    )

    if assigned_speaker == "OVERLAP":
        if speaker_candidates:
            reasons.append("multiple_speaker_candidates")
        return mix_text, "low", reasons

    if best_track is None:
        reasons.append("no_speaker_track_candidate")
        return mix_text, "medium", reasons

    track_text = str(best_track.get("text", "")).strip()

    if not track_text:
        reasons.append("empty_track_text")
        return mix_text, "medium", reasons

    mix_score = _asr_score([mix_seg])
    track_score = float(best_track.get("score", -99.0))

    # 話者トラックのASR品質がmixより大幅に低い場合のみmixにフォールバック
    if track_score < mix_score - 0.5:
        reasons.append("track_low_quality")
        return mix_text, "medium", reasons

    # 話者確定時は話者トラックを優先。mixとの類似度は信頼度の判断にのみ使う
    sim = _text_similarity(mix_text, track_text)
    if sim < 0.35:
        reasons.append("mix_track_text_mismatch")
    confidence = "high" if sim >= 0.60 else "medium"
    if "overlap" in reasons and confidence == "high":
        confidence = "medium"
    return track_text, confidence, reasons


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _track_asr_support_score(
    mix_seg: TranscriptSegment,
    candidate: dict[str, Any],
    rms_scores: dict[str, float],
    active_speakers: list[str],
    dominant_speaker: str | None,
) -> tuple[float, list[str]]:
    """fusion v2用の話者候補スコア。

    文字が出ていることを主軸にし、RMSは補助点として使う。
    """
    reasons: list[str] = []
    speaker = str(candidate.get("speaker") or "")
    text = str(candidate.get("text") or "").strip()
    mix_text = mix_seg.text.strip()
    asr_score = float(candidate.get("score", -99.0))
    sim = _text_similarity(mix_text, text)

    score = 0.0
    if text:
        score += 2.0
        reasons.append("track_asr_text")
    if asr_score > -0.8:
        score += 1.0
        reasons.append("track_asr_good")
    elif asr_score > -1.3:
        score += 0.5
        reasons.append("track_asr_ok")
    else:
        score -= 0.4
        reasons.append("track_asr_weak")

    score += _clamp(sim * 1.6, 0.0, 1.6)
    if mix_text and sim < 0.25:
        score -= 0.4
        reasons.append("mix_track_low_similarity")
    elif sim >= 0.55:
        reasons.append("mix_track_similar")

    if speaker == dominant_speaker:
        score += 1.2
        reasons.append("rms_dominant")
    elif speaker in active_speakers:
        score += 0.6
        reasons.append("rms_active")
    else:
        score -= 0.6
        reasons.append("rms_inactive")

    if rms_scores and speaker in rms_scores:
        ranked = sorted(rms_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_db = ranked[0][1]
        speaker_db = rms_scores[speaker]
        if top_db - speaker_db > 12.0:
            score -= 0.5
            reasons.append("rms_far_from_top")

    return round(score, 3), reasons


def _choose_fused_text_v2(
    mix_seg: TranscriptSegment,
    candidates: list[dict[str, Any]],
) -> tuple[str, str | None, str, list[str], list[dict[str, Any]]]:
    """ASR単語存在を主軸にしたfusion v2判定。

    本文は冒頭欠けを避けるためmixを基本にし、話者別トラックは話者判定と候補確認へ使う。
    """
    reasons: list[str] = []
    mix_text = mix_seg.text.strip()
    mix_score = _asr_score([mix_seg])
    rms_scores = dict(mix_seg.speaker_scores)
    active_speakers = list(mix_seg.overlap_speakers)
    dominant = max(rms_scores.items(), key=lambda kv: kv[1])[0] if rms_scores else None
    mix_repetition_suspect = bool(mix_seg.asr.get("suspected_repetition_hallucination"))
    if mix_repetition_suspect:
        reasons.append("mix_repetition_hallucination_suspected")

    speaker_candidates: list[dict[str, Any]] = []
    for cand in candidates:
        if cand.get("source") == "mix" or not cand.get("speaker"):
            continue
        fusion_score, fusion_reasons = _track_asr_support_score(
            mix_seg,
            cand,
            rms_scores=rms_scores,
            active_speakers=active_speakers,
            dominant_speaker=dominant,
        )
        enriched = dict(cand)
        enriched["fusion_score"] = fusion_score
        enriched["fusion_reasons"] = fusion_reasons
        speaker_candidates.append(enriched)

    if not speaker_candidates:
        reasons.append("no_speaker_track_candidate")
        if mix_repetition_suspect:
            reasons.append("mix_text_suppressed_due_to_repetition_v2")
            return "", mix_seg.primary_speaker, "low", reasons, candidates
        return mix_text, mix_seg.primary_speaker, "medium", reasons, candidates

    speaker_candidates.sort(key=lambda c: float(c.get("fusion_score", -99.0)), reverse=True)
    best = speaker_candidates[0]
    second = speaker_candidates[1] if len(speaker_candidates) >= 2 else None
    best_speaker = str(best.get("speaker"))
    best_score = float(best.get("fusion_score", -99.0))
    second_score = float(second.get("fusion_score", -99.0)) if second else -99.0
    assigned: str | None = best_speaker

    overlap_speakers: list[str] = []
    if second is not None:
        second_speaker = str(second.get("speaker"))
        close_scores = best_score - second_score <= 0.9
        both_active = best_speaker in active_speakers and second_speaker in active_speakers
        if (mix_seg.is_overlap and close_scores) or (both_active and close_scores):
            assigned = "OVERLAP"
            overlap_speakers = [best_speaker, second_speaker]
            reasons.append("multiple_speaker_asr_candidates")

    best_text = str(best.get("text", "")).strip()
    best_track_score = float(best.get("score", -99.0))
    sim = _text_similarity(mix_text, best_text)
    chosen_text = mix_text
    best_norm = _normalize_candidate_text(best_text)
    mix_norm = _normalize_candidate_text(mix_text)

    if mix_repetition_suspect:
        usable_tracks = [
            c for c in speaker_candidates
            if _normalize_candidate_text(str(c.get("text", "")))
            and _normalize_candidate_text(str(c.get("text", ""))) != mix_norm
            and len(_normalize_candidate_text(str(c.get("text", "")))) >= 2
            and (
                float(c.get("score", -99.0)) >= -3.0
                or float(c.get("fusion_score", -99.0)) >= 2.0
            )
        ][:2]
        if assigned == "OVERLAP" and len(usable_tracks) >= 2:
            chosen_text = " / ".join(str(c.get("text", "")).strip() for c in usable_tracks)
            reasons.append("track_text_adopted_due_to_mix_repetition_overlap_v2")
        elif usable_tracks:
            chosen_text = str(usable_tracks[0].get("text", "")).strip()
            reasons.append("track_text_adopted_due_to_mix_repetition_v2")
        else:
            chosen_text = ""
            reasons.append("mix_text_suppressed_due_to_repetition_v2")
    elif (
        assigned != "OVERLAP"
        and best_text
        and sim >= 0.70
        and best_track_score >= mix_score - 0.2
        and len(best_norm) >= len(mix_norm) * 0.75
    ):
        chosen_text = best_text
        reasons.append("track_text_adopted_v2")
    else:
        reasons.append("mix_text_preserved_v2")
        if best_text and sim < 0.35:
            reasons.append("mix_track_text_mismatch")

    if assigned == "OVERLAP":
        confidence = "low"
    elif mix_repetition_suspect:
        confidence = "medium" if chosen_text != mix_text else "low"
    elif best_score >= 4.0 and sim >= 0.45:
        confidence = "high"
    else:
        confidence = "medium"

    enriched_candidates = [c for c in candidates if c.get("source") == "mix"] + speaker_candidates
    if overlap_speakers:
        mix_seg.overlap_speakers = overlap_speakers
        mix_seg.is_overlap = True
    return chosen_text, assigned, confidence, reasons, enriched_candidates


def fuse_mix_and_speaker_track_transcripts(
    mix_segments: list[TranscriptSegment],
    speaker_track_segments: dict[str, list[TranscriptSegment]],
    speaker_tracks: list[SpeakerTrack],
    active_db: AutoFloat = "auto",
    overlap_db: AutoFloat = "auto",
    margin: AutoFloat = "auto",
    cache_dir: str | Path | None = None,
    fusion_version: str = "v1",
) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    """mix文字起こしに、話者別文字起こし候補とRMS/VAD判定を統合する。"""
    mix_segments, diarization_meta = assign_speakers_from_track_rms(
        mix_segments,
        speaker_tracks=speaker_tracks,
        active_db=active_db,
        overlap_db=overlap_db,
        margin=margin,
        cache_dir=cache_dir,
    )

    for label, segments in speaker_track_segments.items():
        for seg in segments:
            seg.source = label

    version = "v2" if str(fusion_version).lower() in {"v2", "fusion_v2"} else "v1"
    repetition_flags, repetition_meta = _detect_mix_repetition_hallucinations(mix_segments)
    fused: list[TranscriptSegment] = []
    for seg in mix_segments:
        assigned = seg.primary_speaker
        candidates: list[dict[str, Any]] = []
        mix_candidate = _candidate_from_segments("mix", assigned, [seg])
        if mix_candidate:
            candidates.append(mix_candidate)

        for label, track_segments in speaker_track_segments.items():
            overlaps = _overlapping_segments(track_segments, seg.start, seg.end)
            if not overlaps:
                continue
            if version == "v1" and assigned is not None and assigned != "OVERLAP" and assigned != label and label not in seg.overlap_speakers:
                continue
            # 話者確定時、トラックセグメントがmixウィンドウにほぼ収まる場合はクリップしない。
            # 収まらない（長いトラックセグが複数mixウィンドウにまたがる）場合はクリップして重複を防ぐ。
            if assigned == label:
                track_start = min(float(s.start) for s in overlaps)
                track_end = max(float(s.end) for s in overlaps)
                track_dur = track_end - track_start
                ov = _overlap_seconds(track_start, track_end, seg.start, seg.end)
                if track_dur <= 0.001 or ov / track_dur >= 0.70:
                    cand = _candidate_from_segments(label, label, overlaps)
                else:
                    cand = _candidate_from_segments(label, label, overlaps, clip_start=seg.start, clip_end=seg.end)
            else:
                cand = _candidate_from_segments(label, label, overlaps, clip_start=seg.start, clip_end=seg.end)
            if cand:
                candidates.append(cand)

        if version == "v2":
            text, new_assigned, confidence, reasons, candidates = _choose_fused_text_v2(seg, candidates)
            seg.primary_speaker = new_assigned
        else:
            text, confidence, reasons = _choose_fused_text(seg, assigned, candidates)
        seg.text = text
        seg.candidates = candidates
        seg.confidence = confidence
        seg.confidence_reasons = reasons
        if seg.segment_id in repetition_flags and "mix_repetition_hallucination_suspected" not in seg.confidence_reasons:
            seg.confidence_reasons.append("mix_repetition_hallucination_suspected")
            if seg.confidence == "high":
                seg.confidence = "medium"
        seg.source = f"fusion_{version}"
        fused.append(seg)

    quality_meta = apply_transcript_quality_postprocessing(fused)
    meta = {
        **diarization_meta,
        "fusion_version": version,
        "fusion_method": "mix_plus_speaker_track_whisper",
        "fusion_note": "v1=RMS主導。v2=話者別ASR単語存在を主軸にRMS/mix類似度で補助し、本文は通常mix優先。ただしmixの異常反復は話者別トラック候補へ退避する",
        "mix_repetition_hallucination": repetition_meta,
        "quality_postprocess": quality_meta,
    }
    return fused, meta


def transcribe_with_speaker_track_fusion(
    audio_path: str | Path,
    output_path: str | Path,
    speaker_tracks: list[SpeakerTrack],
    diarization_speaker_tracks: list[SpeakerTrack] | None = None,
    model_size: str = "large-v3",
    language: str = "ja",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    active_db: AutoFloat = "auto",
    overlap_db: AutoFloat = "auto",
    margin: AutoFloat = "auto",
    progress: ProgressCallback | None = None,
    fusion_version: str = "v1",
    extra_meta: dict[str, Any] | None = None,
    enable_crosstalk_cancellation: bool = False,
) -> tuple[list[TranscriptSegment], dict[str, list[TranscriptSegment]]]:
    """mix + 各話者トラックを文字起こしし、RMS/VADで統合する高精度モード。

    Args:
        enable_crosstalk_cancellation: Trueの場合、ASR前に各話者トラックから
            他話者の漏れ込み（クロストーク）を除去する。RMS/VAD計算には
            元のトラックを使う（漏れ込みもRMS基準として有効なため）。

    Returns: (fused_segments, speaker_track_segments)
    """
    normalized_tracks = _as_speaker_tracks(speaker_tracks)
    rms_tracks = _as_speaker_tracks(diarization_speaker_tracks or speaker_tracks)
    cache_root = Path(output_path).parent / ".transcribe_cache"

    # クロストーク除去: 話者トラックから他話者の漏れ成分を除去した版を作る。
    # RMS/VAD（rms_tracks）は元のままにして、ASR用トラックのみ差し替える。
    crosstalk_meta: dict[str, Any] | None = None
    if enable_crosstalk_cancellation and len(normalized_tracks) >= 2:
        from core.review.crosstalk_cancellation import cancel_crosstalk_on_tracks

        crosstalk_dir = Path(output_path).parent / "crosstalk_cleaned"
        cleaned_tracks, crosstalk_meta = cancel_crosstalk_on_tracks(
            [(label, path) for label, path in normalized_tracks],
            crosstalk_dir,
            progress=progress,
            progress_base=7.0,
            progress_span=1.5,
        )
        normalized_tracks = _as_speaker_tracks(cleaned_tracks)

    track_count = max(1, len(normalized_tracks))
    mix_base = 10.0
    mix_span = 35.0
    tracks_base = mix_base + mix_span
    tracks_span = 40.0
    per_track_span = tracks_span / track_count

    _progress(progress, "mix音声の文字起こし開始", audio=str(audio_path), progress_percent=mix_base)
    mix_segments = _faster_whisper_transcribe_only(
        audio_path,
        model_size=model_size,
        language=language,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        progress=progress,
        progress_label="mix音声",
        progress_base=mix_base,
        progress_span=mix_span,
        cache_dir=cache_root,
    )
    _progress(progress, "mix音声の文字起こし完了", segments=len(mix_segments), progress_percent=tracks_base)
    track_segments: dict[str, list[TranscriptSegment]] = {}
    for index, (label, path) in enumerate(normalized_tracks, start=1):
        base = tracks_base + per_track_span * (index - 1)
        _progress(
            progress,
            "話者別トラックの文字起こし開始",
            index=index,
            total=len(normalized_tracks),
            speaker=label,
            audio=str(path),
            progress_percent=round(base, 1),
        )
        track_segments[label] = _faster_whisper_transcribe_only(
            path,
            model_size=model_size,
            language=language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            progress=progress,
            progress_label=f"話者別トラック {label}",
            progress_base=base,
            progress_span=per_track_span,
            cache_dir=cache_root,
        )
        _progress(
            progress,
            "話者別トラックの文字起こし完了",
            index=index,
            total=len(normalized_tracks),
            speaker=label,
            segments=len(track_segments[label]),
            progress_percent=round(base + per_track_span, 1),
        )

    _progress(progress, "RMS/VAD統合開始", progress_percent=87.0)
    segments, fusion_meta = fuse_mix_and_speaker_track_transcripts(
        mix_segments,
        track_segments,
        speaker_tracks=rms_tracks,
        active_db=active_db,
        overlap_db=overlap_db,
        margin=margin,
        cache_dir=cache_root,
        fusion_version=fusion_version,
    )
    _progress(
        progress,
        "RMS/VAD統合完了",
        segments=len(segments),
        active_db=fusion_meta.get("speaker_track_active_db"),
        overlap_db=fusion_meta.get("speaker_track_overlap_db"),
        rms_cache_hit=fusion_meta.get("cache_hit"),
        progress_percent=93.0,
    )
    _progress(progress, "JSON保存開始", output=str(output_path), progress_percent=95.0)
    save_transcript(
        segments,
        output_path,
        meta={
            "transcriber": "faster-whisper+speaker-track-fusion",
            "model": model_size,
            "language": language,
            "unit": "word+speaker+candidates",
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
            "fusion_version": fusion_meta.get("fusion_version", fusion_version),
            **fusion_meta,
            **({"crosstalk_cancellation": crosstalk_meta} if crosstalk_meta else {}),
            **(extra_meta or {}),
        },
    )
    _progress(progress, "JSON保存完了", output=str(output_path), progress_percent=97.0)
    return segments, track_segments


def transcribe_with_diarization(
    audio_path: str | Path,
    output_path: str | Path,
    model_size: str = "large-v3",
    language: str = "ja",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[TranscriptSegment]:
    """faster-whisper で文字起こし + pyannote.audio で話者識別を統合する。

    - Whisper segment ごとに speaker embedding を取り、短い発話単位でクラスタリング
    - segment 内の word に speaker を割り当て
    - 出力 JSON のメタに transcriber=faster-whisper+pyannote を記録
    """
    import os
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN が必要です (pyannote モデル利用のため)。\n"
            "1. https://huggingface.co/settings/tokens で read トークン作成\n"
            "2. https://huggingface.co/pyannote/speaker-diarization-3.1 で利用同意\n"
            "3. https://huggingface.co/pyannote/segmentation-3.0 でも利用同意\n"
            "4. 環境変数 HF_TOKEN にトークンをセット (例: setx HF_TOKEN hf_xxx)"
        )

    segments = _faster_whisper_transcribe_only(
        audio_path,
        model_size=model_size,
        language=language,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )
    diarization = _run_pyannote_segment_embedding_diarization(
        audio_path, segments,
        hf_token=hf_token, min_speakers=min_speakers, max_speakers=max_speakers,
    )
    if diarization.segment_speakers:
        segments = assign_segment_speakers_to_words(segments, diarization.segment_speakers)
    else:
        # embedding が取れない場合だけ従来の広い pyannote 区間割当へフォールバック
        intervals = _run_pyannote_diarization(
            audio_path, hf_token=hf_token, min_speakers=min_speakers, max_speakers=max_speakers,
        )
        diarization = DiarizationResult(
            intervals=intervals,
            meta={
                **diarization.meta,
                "diarization_fallback": "pyannote_timeline",
            },
        )
        segments = assign_speakers_to_segments(segments, intervals)

    save_transcript(
        segments, output_path,
        meta={
            "transcriber": "faster-whisper+pyannote",
            "model": model_size,
            "language": language,
            "unit": "word+speaker",
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            **diarization.meta,
            "speaker_intervals": [
                {"start": s, "end": e, "speaker": sp} for s, e, sp in diarization.intervals
            ],
        },
    )
    return segments
