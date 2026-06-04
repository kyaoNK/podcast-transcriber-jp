"""タイムコード変換ユーティリティ。

既存テロップCSV (例: `260221(2)_パスワードの死.csv`) のタイムコードは
`HH;MM;SS;ff` 形式で、末尾 `ff` は **1/100秒精度** (centisecond)。
内部処理ではすべて float 秒に正規化する。
"""
from __future__ import annotations

import re

DEFAULT_TC_UNITS_PER_SECOND = 100

_SMPTE_RE = re.compile(r"^(\d{1,2})[;:](\d{1,2})[;:](\d{1,2})[;:](\d{1,3})$")
_HHMMSS_MS_RE = re.compile(r"^(\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?$")
_MMSS_MS_RE = re.compile(r"^(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?$")
_FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def smpte_to_seconds(tc: str, units_per_second: int = DEFAULT_TC_UNITS_PER_SECOND) -> float:
    """`HH;MM;SS;ff` または `HH:MM:SS:ff` を秒に変換。

    `ff` は 1/units_per_second 精度。デフォルト 100 (1/100秒)。
    """
    s = tc.strip()
    m = _SMPTE_RE.match(s)
    if not m:
        raise ValueError(f"unsupported SMPTE timecode: {tc!r}")
    hh, mm, ss, ff = (int(g) for g in m.groups())
    if ff >= units_per_second:
        raise ValueError(
            f"frame value {ff} exceeds units_per_second={units_per_second} in {tc!r}"
        )
    return hh * 3600 + mm * 60 + ss + ff / units_per_second


def seconds_to_smpte(
    seconds: float,
    units_per_second: int = DEFAULT_TC_UNITS_PER_SECOND,
    sep: str = ";",
) -> str:
    """秒を `HH;MM;SS;ff` 形式に変換。"""
    if seconds < 0:
        raise ValueError(f"negative seconds not supported: {seconds}")
    total_units = round(seconds * units_per_second)
    ff = total_units % units_per_second
    total_seconds = total_units // units_per_second
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    width = len(str(units_per_second - 1))
    return f"{hh:02d}{sep}{mm:02d}{sep}{ss:02d}{sep}{ff:0{width}d}"


def format_hhmmss_ms(seconds: float) -> str:
    """秒を `HH:MM:SS.fff` (ミリ秒精度) に変換。人間確認用。"""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def parse_any(text: str, units_per_second: int = DEFAULT_TC_UNITS_PER_SECOND) -> float:
    """文字列を秒に変換。SMPTE / `HH:MM:SS.fff` / `MM:SS.fff` / 秒数 を受け付ける。"""
    s = str(text).strip()
    if not s:
        raise ValueError("empty timecode")

    if _SMPTE_RE.match(s):
        return smpte_to_seconds(s, units_per_second=units_per_second)

    m = _HHMMSS_MS_RE.match(s)
    if m:
        hh, mm, ss, ms = m.group(1), m.group(2), m.group(3), m.group(4) or "0"
        ms_padded = (ms + "000")[:3]
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms_padded) / 1000

    m = _MMSS_MS_RE.match(s)
    if m:
        mm, ss, ms = m.group(1), m.group(2), m.group(3) or "0"
        ms_padded = (ms + "000")[:3]
        return int(mm) * 60 + int(ss) + int(ms_padded) / 1000

    if _FLOAT_RE.match(s):
        return float(s)

    raise ValueError(f"unsupported timecode format: {text!r}")
