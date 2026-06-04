from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager
from typing import Generator


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def prevent_sleep() -> Generator[None, None, None]:
    """処理中にOSのスリープを抑制する。Windows以外は何もしない。

    with prevent_sleep():
        long_running_task()
    """
    if sys.platform != "win32":
        yield
        return

    ctypes.windll.kernel32.SetThreadExecutionState(
        _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    )
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
