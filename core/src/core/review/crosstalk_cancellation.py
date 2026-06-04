"""話者別トラック間のクロストーク（ブリードイン）を除去する。

各話者は近接マイクを持っているが、同じ部屋で収録されているため相手の声も
小さく入り込んでいる。このモジュールは、参照トラック（相手のマイク）の
信号を使って目的トラックから漏れ成分を引き算する。

実装方針:
- 周波数領域 Wiener フィルタ。FFT で高速に処理できる。
- 部屋の伝達関数は収録中ほぼ一定とみなし、全体平均から推定。
- 推定の信頼性のため、参照トラックが活発な区間のみで H を学習する。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

ProgressCallback = Callable[[str, dict[str, Any]], None]


def _progress(progress: ProgressCallback | None, event: str, **data: Any) -> None:
    if progress is None:
        return
    try:
        progress(event, dict(data))
    except Exception:
        pass


def estimate_crosstalk_filter(
    desired: Any,
    reference: Any,
    sample_rate: int,
    frame_ms: float = 32.0,
    hop_ms: float = 8.0,
    active_percentile: float = 50.0,
) -> Any:
    """desired から reference への伝達関数 H(f) を周波数領域で推定する。

    H(f) = sum(D X*) / sum(|X|^2), 参照が活発な区間のみで集計。
    返り値は複素 ndarray (n_freq,)。
    """
    import numpy as np  # type: ignore
    from scipy import signal as sps  # type: ignore

    n_fft = int(sample_rate * frame_ms / 1000.0)
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    nperseg = n_fft
    noverlap = max(0, n_fft - hop)

    _, _, D = sps.stft(desired, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    _, _, X = sps.stft(reference, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    if D.shape[1] == 0 or X.shape[1] == 0:
        return np.zeros(D.shape[0], dtype=np.complex128)

    # 参照が活発なフレームだけを伝達関数推定に使う
    x_pow_per_frame = np.sum(np.abs(X) ** 2, axis=0)
    threshold = float(np.percentile(x_pow_per_frame, active_percentile))
    active_mask = x_pow_per_frame > threshold
    if active_mask.sum() < 10:
        return np.zeros(D.shape[0], dtype=np.complex128)

    D_active = D[:, active_mask]
    X_active = X[:, active_mask]
    cross = np.sum(D_active * np.conj(X_active), axis=1)
    x_pow = np.sum(np.abs(X_active) ** 2, axis=1) + 1e-10
    H = cross / x_pow
    return H


def apply_crosstalk_filter(
    desired: Any,
    reference: Any,
    H: Any,
    sample_rate: int,
    frame_ms: float = 32.0,
    hop_ms: float = 8.0,
) -> Any:
    """推定済み伝達関数 H で reference 成分を desired から除去する。"""
    import numpy as np  # type: ignore
    from scipy import signal as sps  # type: ignore

    n_fft = int(sample_rate * frame_ms / 1000.0)
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    nperseg = n_fft
    noverlap = max(0, n_fft - hop)

    _, _, D = sps.stft(desired, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    _, _, X = sps.stft(reference, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    D_cleaned = D - H[:, np.newaxis] * X
    _, d_cleaned = sps.istft(D_cleaned, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    return d_cleaned[: len(desired)]


def measure_leakage_reduction(
    original: Any,
    cleaned: Any,
    reference: Any,
    sample_rate: int,
) -> dict[str, float]:
    """クロストーク除去がどれだけ漏れを減らしたかを測定する。

    参照が活発で目的が静かな区間（=漏れだけが鳴っている区間）で
    original と cleaned の RMS を比較する。
    """
    import numpy as np  # type: ignore

    win = max(1, int(sample_rate * 0.05))  # 50ms 窓
    n = min(len(original), len(cleaned), len(reference))
    n = (n // win) * win
    if n == 0:
        return {"reduction_db": 0.0, "n_leak_windows": 0}
    o = original[:n].reshape(-1, win)
    c = cleaned[:n].reshape(-1, win)
    r = reference[:n].reshape(-1, win)

    rms_o = np.sqrt(np.mean(o ** 2, axis=1) + 1e-12)
    rms_c = np.sqrt(np.mean(c ** 2, axis=1) + 1e-12)
    rms_r = np.sqrt(np.mean(r ** 2, axis=1) + 1e-12)

    # 参照活発 (上位30%) かつ 目的が静か (下位50%) な窓 = 漏れ窓
    ref_active = rms_r > np.percentile(rms_r, 70)
    desired_quiet_threshold = np.percentile(rms_o, 50)
    desired_quiet = rms_o < desired_quiet_threshold
    leak_mask = ref_active & desired_quiet

    if leak_mask.sum() < 5:
        return {"reduction_db": 0.0, "n_leak_windows": int(leak_mask.sum())}

    o_leak_rms = float(np.sqrt(np.mean(rms_o[leak_mask] ** 2)))
    c_leak_rms = float(np.sqrt(np.mean(rms_c[leak_mask] ** 2)))
    if o_leak_rms < 1e-9:
        reduction_db = 0.0
    else:
        import math

        reduction_db = 20.0 * math.log10(max(c_leak_rms, 1e-9) / o_leak_rms)
    return {
        "reduction_db": round(reduction_db, 2),
        "n_leak_windows": int(leak_mask.sum()),
        "original_leak_rms": round(o_leak_rms, 6),
        "cleaned_leak_rms": round(c_leak_rms, 6),
    }


def cancel_crosstalk_on_tracks(
    speaker_tracks: list[tuple[str, str | Path]],
    output_dir: str | Path,
    progress: ProgressCallback | None = None,
    progress_base: float | None = None,
    progress_span: float | None = None,
    frame_ms: float = 32.0,
    hop_ms: float = 8.0,
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    """各トラックから他トラックの漏れを除去した版を生成する。

    Args:
        speaker_tracks: [(label, path), ...] 話者別音声トラック
        output_dir: 出力先ディレクトリ
        progress: 進捗コールバック

    Returns:
        cleaned_tracks: [(label, cleaned_path), ...]
        meta: 処理メタデータ（reduction_db 等を含む）
    """
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(speaker_tracks) < 2:
        # 2本未満ならクロストーク除去できない
        return (
            [(label, Path(path)) for label, path in speaker_tracks],
            {"applied": False, "reason": "tracks < 2"},
        )

    def safe_label(label: str, index: int) -> str:
        value = re.sub(r'[<>:"/\\|?*\s]+', "_", str(label).strip())[:48].strip("._")
        return value or f"TRACK_{index}"

    # 全トラック読み込み
    _progress(progress, "クロストーク除去開始", tracks=len(speaker_tracks), progress_percent=progress_base)
    audio_list: list[Any] = []
    sample_rates: list[int] = []
    for label, path in speaker_tracks:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)
        else:
            data = data[:, 0]
        audio_list.append(data)
        sample_rates.append(int(sr))

    sample_rate = sample_rates[0]
    if any(sr != sample_rate for sr in sample_rates):
        raise ValueError(f"sample rate mismatch: {sample_rates}")

    min_len = min(len(a) for a in audio_list)
    audio_list = [a[:min_len] for a in audio_list]

    output_paths: list[tuple[str, Path]] = []
    measurements: list[dict[str, Any]] = []
    n_tracks = len(audio_list)

    for i in range(n_tracks):
        label_i = speaker_tracks[i][0]
        d = audio_list[i]
        # 他のすべてのトラックを参照にして順番に漏れを引く
        cleaned = d.copy()
        per_ref_meta: list[dict[str, Any]] = []
        for j in range(n_tracks):
            if i == j:
                continue
            x = audio_list[j]
            H = estimate_crosstalk_filter(cleaned, x, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
            new_cleaned = apply_crosstalk_filter(cleaned, x, H, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
            del H  # 周波数領域フィルタは使い終わり次第解放
            metric = measure_leakage_reduction(cleaned, new_cleaned, x, sample_rate)
            per_ref_meta.append(
                {
                    "reference_label": speaker_tracks[j][0],
                    "reduction_db": metric.get("reduction_db", 0.0),
                    "n_leak_windows": metric.get("n_leak_windows", 0),
                }
            )
            # 漏れ除去で逆に悪化した（reduction_db > 0 = 大きくなった）場合はrevert
            if metric.get("reduction_db", 0.0) > 0.5:
                _progress(
                    progress,
                    "クロストーク除去スキップ",
                    target=label_i,
                    reference=speaker_tracks[j][0],
                    reduction_db=metric.get("reduction_db"),
                )
                del new_cleaned  # revertした版は即解放
                continue
            cleaned = new_cleaned
            del new_cleaned  # cleaned に参照を渡したので別名は破棄
        # クリッピング防止
        peak = float(np.max(np.abs(cleaned)))
        if peak > 0.99:
            cleaned = cleaned * (0.99 / peak)

        out_path = output_dir / f"{i + 1:02d}_{safe_label(label_i, i + 1)}.crosstalk_clean.wav"
        sf.write(str(out_path), cleaned.astype(np.float32), sample_rate, subtype="PCM_16")
        del cleaned  # 書き出し済み。次トラックは audio_list[i+1] を参照する
        output_paths.append((label_i, out_path))
        measurements.append(
            {
                "label": label_i,
                "output": str(out_path),
                "per_reference": per_ref_meta,
            }
        )
        _progress(
            progress,
            "クロストーク除去進捗",
            target=label_i,
            track_index=i + 1,
            total=n_tracks,
            progress_percent=(
                round(progress_base + progress_span * ((i + 1) / n_tracks), 1)
                if progress_base is not None and progress_span is not None
                else None
            ),
        )

    meta = {
        "applied": True,
        "tracks": measurements,
        "sample_rate": sample_rate,
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
    }
    (output_dir / ".crosstalk_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _progress(progress, "クロストーク除去完了", tracks=len(output_paths), progress_percent=progress_base)
    return output_paths, meta
