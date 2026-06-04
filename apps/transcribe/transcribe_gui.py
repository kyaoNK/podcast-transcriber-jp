"""話者別トラックfusion専用の文字起こしGUI。

Tkinterだけで動く薄いラッパー。
`transcribe` の話者別トラック高精度fusionだけを呼ぶ。
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from tkinter import BooleanVar, END, StringVar, Text, Tk, filedialog, messagebox
from tkinter import ttk

from core.review.transcribe_audio import atomic_write_text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = (
    "これは日本語の会話音声です。"
    "コンピュータ科学、情報セキュリティ、ソフトウェア工学に関する専門用語を正確に書き起こしてください。"
    "話者名や固有名詞は、入力された用語リストの表記を優先してください。"
)
AUDIO_FILETYPES = [("Audio", "*.wav *.mp3 *.m4a *.mp4 *.mov *.mkv"), ("All", "*.*")]
SPEAKER_TRACK_FILETYPES = [("WAV", "*.wav"), ("All", "*.*")]
PRESET_FILETYPES = [("JSON", "*.json"), ("All", "*.*")]
MODEL_CHOICES = ("large-v3", "medium", "small", "base", "tiny")
LANGUAGE_CHOICES = ("ja", "en")
FUSION_VERSION_CHOICES = ("v1（安定版・現状）", "v2（試験版・ASR+RMS）")
DURATION_MISMATCH_TOLERANCE = 2.0
DEFAULT_OUTPUT_NAME = "transcript_speaker_tracks"
HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{10,}")
SETTINGS_PATH = ROOT / ".transcribe_gui_settings.json"
HOTWORDS_TOKEN_LIMIT = 223
HOTWORDS_WARN_TOKENS = 180
HOTWORDS_WARN_LINES = 80


def format_command_for_log(cmd: list[str]) -> str:
    """ログに出すコマンドからsecretを伏せる。"""
    safe: list[str] = []
    redact_next = False
    for part in cmd:
        if redact_next:
            safe.append("***")
            redact_next = False
            continue
        if part == "--hf-token":
            safe.append(part)
            redact_next = True
            continue
        if part.startswith("--hf-token="):
            safe.append("--hf-token=***")
            continue
        safe.append(HF_TOKEN_RE.sub("hf_***", part))
    return " ".join(safe)


class TranscribeGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("話者別トラック文字起こし + 話者識別")
        root.geometry("1180x860")
        root.minsize(960, 700)
        self.output_queue: queue.Queue[str | None] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.current_proc: subprocess.Popen[str] | None = None
        self.current_run_started: float | None = None
        self.current_run_kind = ""
        self.run_duration_history: list[str] = []
        self.stop_requested = False
        self.paused = False
        self.progress_total = 0
        self.progress_current = 0
        self.final_status: tuple[str, str] | None = None
        self.temp_files: list[Path] = []
        self.speaker_tracks: list[tuple[str, Path]] = []
        self.batch_jobs: list[dict[str, object]] = []
        self.batch_running = False
        self.current_batch_index = 0
        self.current_batch_total = 0
        self.run_lock_widgets: list[object] = []
        self.run_lock_text_widgets: list[Text] = []
        self.output_dir_auto = True

        self.mode_help = StringVar()
        self.crosstalk_cancel = BooleanVar(value=True)
        self.output_dir = StringVar()
        self.output_name = StringVar(value=DEFAULT_OUTPUT_NAME)
        self.output = StringVar()
        self.text_output = StringVar()
        self.selected_speaker_label = StringVar()

        self.model = StringVar(value="large-v3")
        self.language = StringVar(value="ja")
        self.fusion_version = StringVar(value=FUSION_VERSION_CHOICES[0])
        self.diarize = BooleanVar(value=False)
        self.min_speakers = StringVar(value="2")
        self.max_speakers = StringVar(value="2")
        self.hf_token = StringVar()
        self.hotwords_file = StringVar()
        self.speaker_track_active_db = StringVar(value="auto")
        self.speaker_track_overlap_db = StringVar(value="auto")
        self.speaker_track_margin = StringVar(value="auto")
        self.allow_duration_mismatch = BooleanVar(value=False)

        self.status_text = StringVar(value="待機中")
        self.progress_text = StringVar(value="")
        self.input_check_text = StringVar(value="話者別トラックを2本以上追加してください")
        self.hotwords_status_text = StringVar(value="用語: 0件 / 推定0 tokens")
        self.runtime_text = StringVar(value="実行環境: 確認中")
        self.batch_status_text = StringVar(value="バッチキュー: 0件")
        self.current_tab_text = StringVar(value="現在の画面: ① 入力")
        self.tab_group_text = StringVar(value="設定タブ: 実行前に使います")
        self.run_duration_text = StringVar(value="処理時間履歴: まだありません")

        self.batch_folder = StringVar()
        self.batch_glob = StringVar(value="*.wav")
        self.batch_output_dir = StringVar()
        self.batch_recursive = BooleanVar(value=False)
        self.batch_skip_existing = BooleanVar(value=True)
        self.retry_json = StringVar()
        self.retry_audio = StringVar()
        self.retry_output = StringVar()
        self.retry_start = StringVar()
        self.retry_end = StringVar()
        self.retry_force_replace = BooleanVar(value=False)

        self._build()
        self._load_settings()
        self.refresh_runtime_status()
        self.on_mode_change()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 5}
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Yu Gothic UI", 16, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Yu Gothic UI", 10, "bold"))
        style.configure("Hint.TLabel", foreground="#555")
        style.configure("Primary.TButton", font=("Yu Gothic UI", 10, "bold"))
        style.configure("TNotebook.Tab", font=("Yu Gothic UI", 11, "bold"), padding=(18, 8))
        style.configure("SetupBanner.TLabel", background="#e8f1ff", foreground="#003f8c", font=("Yu Gothic UI", 11, "bold"), padding=(10, 8))
        style.configure("MonitorBanner.TLabel", background="#fff4d6", foreground="#7a4b00", font=("Yu Gothic UI", 11, "bold"), padding=(10, 8))
        style.configure("CurrentSetup.TLabel", background="#dbeafe", foreground="#003f8c", font=("Yu Gothic UI", 10, "bold"), padding=(8, 6))
        style.configure("CurrentMonitor.TLabel", background="#fdecc8", foreground="#7a4b00", font=("Yu Gothic UI", 10, "bold"), padding=(8, 6))

        main = ttk.Frame(self.root)
        self.frame = main
        main.pack(fill="both", expand=True, padx=12, pady=12)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="話者別トラック文字起こし + 話者識別", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", **pad
        )
        ttk.Label(
            header,
            textvariable=self.mode_help,
            style="Hint.TLabel",
            wraplength=980,
        ).grid(row=1, column=0, sticky="ew", **pad)

        self.tab_group_label = ttk.Label(main, textvariable=self.tab_group_text, style="SetupBanner.TLabel")
        self.tab_group_label.grid(row=1, column=0, sticky="ew", padx=4, pady=(4, 4))
        self.current_tab_label = ttk.Label(main, textvariable=self.current_tab_text, style="CurrentSetup.TLabel")
        self.current_tab_label.grid(
            row=2, column=0, sticky="ew", padx=4, pady=(0, 4)
        )

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=3, column=0, sticky="nsew", pady=(4, 8))
        input_tab = ttk.Frame(self.notebook)
        batch_tab = ttk.Frame(self.notebook)
        advanced_tab = ttk.Frame(self.notebook)
        retry_tab = ttk.Frame(self.notebook)
        log_tab = ttk.Frame(self.notebook)
        self.input_tab = input_tab
        self.batch_tab = batch_tab
        self.advanced_tab = advanced_tab
        self.retry_tab = retry_tab
        self.log_tab = log_tab
        self.notebook.add(input_tab, text="① 入力")
        self.notebook.add(batch_tab, text="② バッチ")
        self.notebook.add(advanced_tab, text="③ 詳細設定")
        self.notebook.add(retry_tab, text="④ 手動retry")
        self.notebook.add(log_tab, text="⑤ ログ")
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self.update_current_tab_label())
        self.update_current_tab_label()

        input_tab.columnconfigure(0, weight=3)
        input_tab.columnconfigure(1, weight=2)
        input_tab.rowconfigure(0, weight=1)
        input_tab.rowconfigure(1, weight=0)

        input_box = ttk.LabelFrame(input_tab, text="1. 話者別トラック（必須）", style="Section.TLabelframe")
        input_box.grid(row=0, column=0, sticky="nsew", **pad)
        input_box.columnconfigure(0, weight=1)
        input_box.rowconfigure(1, weight=1)
        ttk.Label(
            input_box,
            text="同時に録った話者別音声を2本以上追加します。mix音声は通常不要です。",
            style="Hint.TLabel",
            wraplength=950,
        ).grid(row=0, column=0, columnspan=4, sticky="ew", **pad)

        self.speaker_tree = ttk.Treeview(
            input_box,
            columns=("speaker", "path"),
            show="headings",
            height=3,
            selectmode="extended",
        )
        self.speaker_tree.heading("speaker", text="話者名")
        self.speaker_tree.heading("path", text="音声ファイル")
        self.speaker_tree.column("speaker", width=140, anchor="w", stretch=False)
        self.speaker_tree.column("path", width=760, anchor="w", stretch=True)
        self.speaker_tree.grid(row=1, column=0, columnspan=4, sticky="nsew", **pad)
        self.speaker_tree.bind("<<TreeviewSelect>>", self.on_speaker_select)
        yscroll = ttk.Scrollbar(input_box, orient="vertical", command=self.speaker_tree.yview)
        yscroll.grid(row=1, column=4, sticky="ns", pady=5)
        self.speaker_tree.configure(yscrollcommand=yscroll.set)

        self.add_speaker_button = ttk.Button(input_box, text="音声を追加", command=self.add_speaker_track)
        self.add_speaker_button.grid(row=2, column=0, sticky="w", **pad)
        self.remove_speaker_button = ttk.Button(input_box, text="選択を削除", command=self.remove_selected_speaker_tracks)
        self.remove_speaker_button.grid(row=2, column=1, sticky="w", **pad)
        self.clear_speaker_button = ttk.Button(input_box, text="全削除", command=self.clear_speaker_tracks)
        self.clear_speaker_button.grid(row=2, column=2, sticky="w", **pad)
        self.check_input_button = ttk.Button(input_box, text="入力チェック", command=self.check_inputs)
        self.check_input_button.grid(row=2, column=3, sticky="e", **pad)

        edit_row = ttk.Frame(input_box)
        edit_row.grid(row=3, column=0, columnspan=4, sticky="ew", **pad)
        edit_row.columnconfigure(1, weight=1)
        ttk.Label(edit_row, text="選択した話者名").grid(row=0, column=0, sticky="w")
        self.speaker_label_entry = ttk.Entry(edit_row, textvariable=self.selected_speaker_label, width=24)
        self.speaker_label_entry.grid(row=0, column=1, sticky="w", padx=6)
        self.apply_speaker_label_button = ttk.Button(edit_row, text="名前を反映", command=self.apply_selected_speaker_label)
        self.apply_speaker_label_button.grid(row=0, column=2, sticky="w")
        ttk.Label(edit_row, text="例: 話者A / HORIMOTO / MIZUNO", style="Hint.TLabel").grid(row=0, column=3, sticky="w", padx=12)

        self.input_check_label = ttk.Label(input_box, textvariable=self.input_check_text, foreground="#555", wraplength=960)
        self.input_check_label.grid(row=4, column=0, columnspan=4, sticky="ew", **pad)

        words_box = ttk.LabelFrame(input_tab, text="3. 専門用語・固有名詞（任意）", style="Section.TLabelframe")
        words_box.grid(row=0, column=1, rowspan=2, sticky="nsew", **pad)
        words_box.columnconfigure(0, weight=1)
        words_box.rowconfigure(1, weight=1)
        ttk.Label(
            words_box,
            text="Whisperに優先してほしい単語。1行1語推奨。空欄でも可。",
            style="Hint.TLabel",
            wraplength=380,
        ).grid(row=0, column=0, sticky="ew", **pad)
        self.hotwords_text = Text(words_box, height=10, wrap="word")
        self.hotwords_text.grid(row=1, column=0, sticky="nsew", **pad)
        self.hotwords_text.bind("<<Modified>>", self.on_hotwords_modified)
        self.hotwords_text.edit_modified(False)
        words_button_row = ttk.Frame(words_box)
        words_button_row.grid(row=2, column=0, sticky="ew", **pad)
        self.load_hotwords_button = ttk.Button(words_button_row, text="読込", command=self.load_hotwords_text_file)
        self.load_hotwords_button.grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.save_hotwords_button = ttk.Button(words_button_row, text="保存", command=self.save_hotwords_text_file)
        self.save_hotwords_button.grid(row=0, column=1, sticky="w")
        self.hotwords_status_label = ttk.Label(words_button_row, textvariable=self.hotwords_status_text, style="Hint.TLabel")
        self.hotwords_status_label.grid(row=0, column=2, sticky="w", padx=12)

        output_box = ttk.LabelFrame(input_tab, text="2. 保存先", style="Section.TLabelframe")
        output_box.grid(row=1, column=0, sticky="ew", **pad)
        output_box.columnconfigure(0, weight=1)
        ttk.Label(output_box, text="保存先フォルダ（結果フォルダ）").grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        self.output_dir_entry = ttk.Entry(output_box, textvariable=self.output_dir)
        self.output_dir_entry.grid(row=1, column=0, sticky="ew", **pad)
        self.output_dir_entry.bind("<KeyRelease>", lambda _event: setattr(self, "output_dir_auto", False))
        self.output_dir_button = ttk.Button(output_box, text="参照", command=self.browse_output_dir)
        self.output_dir_button.grid(row=1, column=1, **pad)
        ttk.Label(output_box, text="ファイル名（JSON/レビューMD/ログを同じ名前で保存）").grid(
            row=2, column=0, columnspan=2, sticky="w", **pad
        )
        self.output_name_entry = ttk.Entry(output_box, textvariable=self.output_name)
        self.output_name_entry.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        batch_tab.columnconfigure(0, weight=1)
        batch_tab.rowconfigure(0, weight=1)
        queue_box = ttk.LabelFrame(batch_tab, text="バッチキュー", style="Section.TLabelframe")
        queue_box.grid(row=0, column=0, sticky="nsew", **pad)
        queue_box.columnconfigure(0, weight=1)
        queue_box.rowconfigure(1, weight=1)
        ttk.Label(
            queue_box,
            text="夜間処理用です。入力タブで設定した内容を1ジョブとして追加し、順番に実行します。",
            style="Hint.TLabel",
            wraplength=950,
        ).grid(row=0, column=0, columnspan=6, sticky="ew", **pad)
        self.batch_tree = ttk.Treeview(
            queue_box,
            columns=("status", "name", "tracks", "model", "lang", "fusion", "crosstalk", "output"),
            show="headings",
            height=10,
            selectmode="extended",
        )
        self.batch_tree.heading("status", text="状態")
        self.batch_tree.heading("name", text="ジョブ名")
        self.batch_tree.heading("tracks", text="トラック")
        self.batch_tree.heading("model", text="モデル")
        self.batch_tree.heading("lang", text="言語")
        self.batch_tree.heading("fusion", text="融合")
        self.batch_tree.heading("crosstalk", text="クロストーク")
        self.batch_tree.heading("output", text="保存先")
        self.batch_tree.column("status", width=80, anchor="center", stretch=False)
        self.batch_tree.column("name", width=180, anchor="w", stretch=False)
        self.batch_tree.column("tracks", width=60, anchor="center", stretch=False)
        self.batch_tree.column("model", width=90, anchor="center", stretch=False)
        self.batch_tree.column("lang", width=50, anchor="center", stretch=False)
        self.batch_tree.column("fusion", width=50, anchor="center", stretch=False)
        self.batch_tree.column("crosstalk", width=80, anchor="center", stretch=False)
        self.batch_tree.column("output", width=400, anchor="w", stretch=True)
        # 状態に応じて行の色を変える
        self.batch_tree.tag_configure("status_pending", foreground="#808080")
        self.batch_tree.tag_configure("status_running", foreground="#0066cc", background="#e6f2ff")
        self.batch_tree.tag_configure("status_done", foreground="#2a8a2a")
        self.batch_tree.tag_configure("status_failed", foreground="#c00000", background="#fff0f0")
        self.batch_tree.tag_configure("status_stopped", foreground="#c08000")
        self.batch_tree.tag_configure("status_paused", foreground="#8000c0")
        self.batch_tree.grid(row=1, column=0, columnspan=6, sticky="nsew", **pad)
        self.add_batch_job_button = ttk.Button(queue_box, text="現在の設定を追加", command=self.add_current_to_batch)
        self.add_batch_job_button.grid(row=2, column=0, sticky="w", **pad)
        self.remove_batch_job_button = ttk.Button(queue_box, text="選択を削除", command=self.remove_selected_batch_jobs)
        self.remove_batch_job_button.grid(row=2, column=1, sticky="w", **pad)
        self.clear_batch_jobs_button = ttk.Button(queue_box, text="全削除", command=self.clear_batch_jobs)
        self.clear_batch_jobs_button.grid(row=2, column=2, sticky="w", **pad)
        self.save_batch_button = ttk.Button(queue_box, text="キュー保存", command=self.save_batch_queue)
        self.save_batch_button.grid(row=2, column=3, sticky="w", **pad)
        self.load_batch_button = ttk.Button(queue_box, text="キュー読込", command=self.load_batch_queue)
        self.load_batch_button.grid(row=2, column=4, sticky="w", **pad)
        ttk.Label(queue_box, textvariable=self.batch_status_text, style="Hint.TLabel").grid(row=2, column=5, sticky="e", **pad)

        advanced_tab.columnconfigure(0, weight=1)
        advanced_tab.rowconfigure(0, weight=1)
        self.advanced_box = ttk.Frame(advanced_tab)
        self.advanced_box.grid(row=0, column=0, sticky="nsew", **pad)
        # 2カラム配置: 左=ASR, 右=話者識別+音声前処理
        self.advanced_box.columnconfigure(0, weight=1)
        self.advanced_box.columnconfigure(1, weight=1)

        # === グループ1: ASR設定 (左カラム、rowspan=2で高さフル) ===
        asr_box = ttk.LabelFrame(self.advanced_box, text="ASR（Whisper）設定", style="Section.TLabelframe")
        asr_box.grid(row=0, column=0, rowspan=2, sticky="nsew", **pad)
        asr_box.columnconfigure(1, weight=0)
        asr_box.columnconfigure(3, weight=1)

        ttk.Label(asr_box, text="モデル").grid(row=0, column=0, sticky="w", **pad)
        self.model_combo = ttk.Combobox(asr_box, textvariable=self.model, values=MODEL_CHOICES, width=12, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(asr_box, text="言語").grid(row=0, column=2, sticky="w", **pad)
        self.language_combo = ttk.Combobox(asr_box, textvariable=self.language, values=LANGUAGE_CHOICES, width=6)
        self.language_combo.grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(
            asr_box,
            text="large-v3 / ja 推奨。速度優先なら medium 以下。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky="w", **pad)

        ttk.Label(asr_box, text="基本プロンプト").grid(row=2, column=0, columnspan=4, sticky="w", **pad)
        self.prompt_text = Text(asr_box, height=3, wrap="word")
        self.prompt_text.grid(row=3, column=0, columnspan=4, sticky="ew", **pad)
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)

        ttk.Label(asr_box, text="hotwordsファイル（入力タブの用語欄と併用可）").grid(row=4, column=0, columnspan=4, sticky="w", **pad)
        hw_row = ttk.Frame(asr_box)
        hw_row.grid(row=5, column=0, columnspan=4, sticky="ew", **pad)
        hw_row.columnconfigure(0, weight=1)
        self.hotwords_file_entry = ttk.Entry(hw_row, textvariable=self.hotwords_file)
        self.hotwords_file_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.hotwords_file_button = ttk.Button(hw_row, text="参照", command=self.browse_hotwords_file)
        self.hotwords_file_button.grid(row=0, column=1, sticky="w")

        # === グループ2: 話者識別 (右上) ===
        speaker_box = ttk.LabelFrame(self.advanced_box, text="話者識別", style="Section.TLabelframe")
        speaker_box.grid(row=0, column=1, sticky="nsew", **pad)
        speaker_box.columnconfigure(1, weight=1)

        ttk.Label(speaker_box, text="話者識別方式").grid(row=0, column=0, sticky="w", **pad)
        self.fusion_version_combo = ttk.Combobox(
            speaker_box,
            textvariable=self.fusion_version,
            values=FUSION_VERSION_CHOICES,
            width=22,
            state="readonly",
        )
        self.fusion_version_combo.grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(
            speaker_box,
            text="v1は現状維持。v2は試験版で、本文はmix優先・話者判定はASR+RMS。",
            style="Hint.TLabel",
            wraplength=620,
        ).grid(row=1, column=0, columnspan=2, sticky="w", **pad)

        rms_row = ttk.Frame(speaker_box)
        rms_row.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)
        ttk.Label(rms_row, text="RMS/VAD").pack(side="left")
        ttk.Label(rms_row, text="active dB").pack(side="left", padx=(12, 3))
        self.active_db_entry = ttk.Entry(rms_row, width=8, textvariable=self.speaker_track_active_db)
        self.active_db_entry.pack(side="left")
        ttk.Label(rms_row, text="overlap dB").pack(side="left", padx=(12, 3))
        self.overlap_db_entry = ttk.Entry(rms_row, width=8, textvariable=self.speaker_track_overlap_db)
        self.overlap_db_entry.pack(side="left")
        ttk.Label(rms_row, text="margin秒").pack(side="left", padx=(12, 3))
        self.margin_entry = ttk.Entry(rms_row, width=8, textvariable=self.speaker_track_margin)
        self.margin_entry.pack(side="left")
        ttk.Label(rms_row, text="通常はすべて auto", style="Hint.TLabel").pack(side="left", padx=12)

        self.allow_mismatch_check = ttk.Checkbutton(
            speaker_box,
            text="⚠ 長さ不一致でも実行する（話者割当が壊れるリスクあり）",
            variable=self.allow_duration_mismatch,
        )
        self.allow_mismatch_check.grid(row=3, column=0, columnspan=2, sticky="w", **pad)

        # === グループ3: 音声前処理 (右下) ===
        preprocess_box = ttk.LabelFrame(self.advanced_box, text="音声前処理", style="Section.TLabelframe")
        preprocess_box.grid(row=1, column=1, sticky="nsew", **pad)
        preprocess_box.columnconfigure(1, weight=1)

        self.crosstalk_check = ttk.Checkbutton(
            preprocess_box,
            text="クロストーク除去（話者間のブリードイン除去）",
            variable=self.crosstalk_cancel,
        )
        self.crosstalk_check.grid(row=0, column=0, columnspan=3, sticky="w", **pad)

        retry_tab.columnconfigure(0, weight=1)
        retry_box = ttk.LabelFrame(retry_tab, text="手動ASR retry（既存JSONの一部を再文字起こし）", style="Section.TLabelframe")
        retry_box.grid(row=0, column=0, sticky="ew", **pad)
        retry_box.columnconfigure(1, weight=1)
        ttk.Label(
            retry_box,
            text="自動検出で拾えない違和感がある区間だけ、元JSONを残したまま *.manual_retry.json として再出力します。",
            style="Hint.TLabel",
            wraplength=950,
        ).grid(row=0, column=0, columnspan=4, sticky="ew", **pad)
        ttk.Label(retry_box, text="既存JSON").grid(row=1, column=0, sticky="w", **pad)
        self.retry_json_entry = ttk.Entry(retry_box, textvariable=self.retry_json)
        self.retry_json_entry.grid(row=1, column=1, sticky="ew", **pad)
        self.retry_json_button = ttk.Button(retry_box, text="参照", command=self.browse_retry_json)
        self.retry_json_button.grid(row=1, column=2, sticky="w", **pad)
        self.retry_current_button = ttk.Button(retry_box, text="現在の出力をセット", command=self.set_retry_from_current_output)
        self.retry_current_button.grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(retry_box, text="ASR対象音声/mix").grid(row=2, column=0, sticky="w", **pad)
        self.retry_audio_entry = ttk.Entry(retry_box, textvariable=self.retry_audio)
        self.retry_audio_entry.grid(row=2, column=1, sticky="ew", **pad)
        self.retry_audio_button = ttk.Button(retry_box, text="参照", command=self.browse_retry_audio)
        self.retry_audio_button.grid(row=2, column=2, sticky="w", **pad)
        ttk.Label(retry_box, text="通常は *_mix.wav", style="Hint.TLabel").grid(row=2, column=3, sticky="w", **pad)

        ttk.Label(retry_box, text="出力JSON").grid(row=3, column=0, sticky="w", **pad)
        self.retry_output_entry = ttk.Entry(retry_box, textvariable=self.retry_output)
        self.retry_output_entry.grid(row=3, column=1, sticky="ew", **pad)
        self.retry_output_button = ttk.Button(retry_box, text="参照", command=self.browse_retry_output)
        self.retry_output_button.grid(row=3, column=2, sticky="w", **pad)

        range_row = ttk.Frame(retry_box)
        range_row.grid(row=4, column=0, columnspan=4, sticky="ew", **pad)
        ttk.Label(range_row, text="範囲 秒").pack(side="left")
        ttk.Label(range_row, text="start").pack(side="left", padx=(12, 3))
        self.retry_start_entry = ttk.Entry(range_row, width=10, textvariable=self.retry_start)
        self.retry_start_entry.pack(side="left")
        ttk.Label(range_row, text="end").pack(side="left", padx=(12, 3))
        self.retry_end_entry = ttk.Entry(range_row, width=10, textvariable=self.retry_end)
        self.retry_end_entry.pack(side="left")
        ttk.Label(range_row, text="空欄なら品質異常の自動検出区間だけretry", style="Hint.TLabel").pack(side="left", padx=12)

        self.retry_force_check = ttk.Checkbutton(
            retry_box,
            text="強めに差し替える（通常OFF・候補が重大異常でない場合に差し替えやすくする）",
            variable=self.retry_force_replace,
        )
        self.retry_force_check.grid(row=5, column=0, columnspan=4, sticky="w", **pad)
        self.retry_run_button = ttk.Button(retry_box, text="手動retry実行", command=self.run_manual_retry, style="Primary.TButton")
        self.retry_run_button.grid(row=6, column=0, sticky="w", **pad)

        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(0, weight=1)
        self.log_box = ttk.LabelFrame(log_tab, text="処理ログ", style="Section.TLabelframe")
        self.log_box.grid(row=0, column=0, sticky="nsew", **pad)
        self.log_box.columnconfigure(0, weight=1)
        self.log_box.rowconfigure(0, weight=1)
        self.log = Text(self.log_box, height=16, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew", **pad)
        log_scroll = ttk.Scrollbar(self.log_box, orient="vertical", command=self.log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns", pady=5)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.clear_log_button = ttk.Button(self.log_box, text="ログ消去", command=lambda: self.log.delete("1.0", END))
        self.clear_log_button.grid(row=1, column=0, sticky="w", **pad)

        control_box = ttk.LabelFrame(main, text="実行", style="Section.TLabelframe")
        control_box.grid(row=4, column=0, sticky="ew")
        control_box.columnconfigure(1, weight=1)
        buttons = ttk.Frame(control_box)
        buttons.grid(row=0, column=0, columnspan=3, sticky="ew", **pad)
        self.run_button = ttk.Button(buttons, text="入力チェックして実行", command=self.run, style="Primary.TButton")
        self.run_button.pack(side="left")
        self.run_batch_button = ttk.Button(buttons, text="キューを順番に実行", command=self.run_batch)
        self.run_batch_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(buttons, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.pause_button = ttk.Button(buttons, text="一時停止", command=self.toggle_pause, state="disabled")
        self.pause_button.pack(side="left", padx=8)
        ttk.Separator(buttons, orient="vertical").pack(side="left", fill="y", padx=8)
        self.save_preset_button = ttk.Button(buttons, text="プリセット保存", command=self.save_preset)
        self.save_preset_button.pack(side="left")
        self.load_preset_button = ttk.Button(buttons, text="読込", command=self.load_preset)
        self.load_preset_button.pack(side="left", padx=6)

        ttk.Label(control_box, textvariable=self.status_text, width=28).grid(row=1, column=0, sticky="w", **pad)
        self.progress_bar = ttk.Progressbar(control_box, mode="determinate")
        self.progress_bar.grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Label(control_box, textvariable=self.progress_text, width=28).grid(row=1, column=2, sticky="e", **pad)
        ttk.Label(control_box, textvariable=self.runtime_text, style="Hint.TLabel", wraplength=950).grid(
            row=2, column=0, columnspan=3, sticky="ew", **pad
        )
        ttk.Label(control_box, textvariable=self.run_duration_text, style="Hint.TLabel", wraplength=950).grid(
            row=3, column=0, columnspan=3, sticky="ew", **pad
        )

        self.batch_box = ttk.LabelFrame(main, text="一括処理")
        self.batch_text = Text(self.batch_box, height=5, wrap="none")
        self.pyannote_options = ttk.Frame(main)
        self.hf_label = ttk.Label(main, text="HF token任意")
        self.hf_entry = ttk.Entry(main, textvariable=self.hf_token, show="*")
        self.hf_widgets = [self.hf_label, self.hf_entry]

        self._setup_drag_and_drop()
        self.run_lock_widgets = [
            self.speaker_tree,
            self.add_speaker_button,
            self.remove_speaker_button,
            self.clear_speaker_button,
            self.check_input_button,
            self.speaker_label_entry,
            self.apply_speaker_label_button,
            self.output_dir_entry,
            self.output_dir_button,
            self.output_name_entry,
            self.batch_tree,
            self.add_batch_job_button,
            self.remove_batch_job_button,
            self.clear_batch_jobs_button,
            self.save_batch_button,
            self.load_batch_button,
            self.run_batch_button,
            self.retry_json_entry,
            self.retry_json_button,
            self.retry_current_button,
            self.retry_audio_entry,
            self.retry_audio_button,
            self.retry_output_entry,
            self.retry_output_button,
            self.retry_start_entry,
            self.retry_end_entry,
            self.retry_force_check,
            self.retry_run_button,
            self.save_preset_button,
            self.load_preset_button,
            self.model_combo,
            self.language_combo,
            self.fusion_version_combo,
            self.crosstalk_check,
            self.load_hotwords_button,
            self.save_hotwords_button,
            self.hotwords_file_entry,
            self.hotwords_file_button,
            self.prompt_text,
            self.active_db_entry,
            self.overlap_db_entry,
            self.margin_entry,
            self.allow_mismatch_check,
        ]
        self.run_lock_text_widgets = [self.hotwords_text, self.prompt_text]

    def _on_scroll_container_configure(self, _event: object | None = None) -> None:
        return

    def _on_canvas_configure(self, event: object) -> None:
        return

    def _on_mousewheel(self, event: object) -> None:
        return

    def update_current_tab_label(self) -> None:
        try:
            index = self.notebook.index("current")
        except Exception:
            index = 0
        labels = {
            0: "現在の画面: ① 入力（素材・用語・保存先）",
            1: "現在の画面: ② バッチ（夜間まとめ処理）",
            2: "現在の画面: ③ 詳細設定（通常は変更不要）",
            3: "現在の画面: ④ 手動retry（既存結果の部分再処理）",
            4: "現在の画面: ⑤ ログ（処理状況）",
        }
        self.current_tab_text.set(labels.get(index, "現在の画面"))
        if index in (0, 1, 2, 3):
            self.tab_group_text.set("設定タブ: 実行前に使います。①入力 → 必要なら ②バッチ/③詳細設定 → 下の実行ボタン")
            self.tab_group_label.config(style="SetupBanner.TLabel")
            self.current_tab_label.config(style="CurrentSetup.TLabel")
        else:
            self.tab_group_text.set("確認タブ: 実行中・実行後に見る画面です。処理ログと進捗確認用")
            self.tab_group_label.config(style="MonitorBanner.TLabel")
            self.current_tab_label.config(style="CurrentMonitor.TLabel")

    def is_running(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def set_inputs_locked(self, locked: bool) -> None:
        for widget in self.run_lock_widgets:
            try:
                if widget is self.model_combo:
                    widget.config(state="disabled" if locked else "readonly")  # type: ignore[attr-defined]
                else:
                    widget.config(state="disabled" if locked else "normal")  # type: ignore[attr-defined]
            except Exception:
                pass
        for widget in self.run_lock_text_widgets:
            try:
                widget.config(state="disabled" if locked else "normal")
            except Exception:
                pass

    def refresh_runtime_status(self) -> None:
        try:
            from core.review.transcribe_audio import detect_compute_backend, format_compute_backend

            backend = detect_compute_backend()
            suffix = ""
            if backend.get("faster_whisper_device") == "cuda":
                suffix = "（WhisperはGPUで高速化）"
            else:
                suffix = "（WhisperはCPU実行）"
            model = self.model.get().strip() or "large-v3"
            fusion = self._fusion_version_value()
            lang = self.language.get().strip() or "ja"
            self.runtime_text.set(
                "実行環境: " + format_compute_backend(backend) + suffix
                + f" / model={model}, fusion={fusion}, lang={lang}"
            )
        except Exception as e:
            self.runtime_text.set(f"実行環境: 確認できませんでした: {e}")

    def on_mode_change(self) -> None:
        self.mode_help.set(
            "必要な入力は『話者別トラック2本以上』だけです。保存先は音声ファイル横の結果フォルダに自動設定されます。"
        )
        self.sync_output_paths()
        self.root.update_idletasks()

    def browse_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir_auto = False
            self.output_dir.set(path)
            self.sync_output_paths()

    def add_batch_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=AUDIO_FILETYPES)
        for path in paths:
            self.batch_text.insert(END, str(Path(path)) + "\n")

    def browse_batch_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.batch_folder.set(path)

    def browse_batch_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.batch_output_dir.set(path)

    def clear_batch_targets(self) -> None:
        self.batch_text.delete("1.0", END)
        self.batch_folder.set("")

    def browse_hotwords_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Text", "*.txt *.md"), ("All", "*.*")])
        if path:
            self.hotwords_file.set(path)

    def browse_retry_json(self) -> None:
        path = filedialog.askopenfilename(title="retry対象JSON", filetypes=PRESET_FILETYPES)
        if not path:
            return
        self.retry_json.set(path)
        self.update_retry_defaults_from_json(Path(path))

    def browse_retry_audio(self) -> None:
        path = filedialog.askopenfilename(title="retry用ASR対象音声/mix", filetypes=AUDIO_FILETYPES)
        if path:
            self.retry_audio.set(path)

    def browse_retry_output(self) -> None:
        initial = self.retry_output.get().strip()
        kwargs = {
            "title": "retry出力JSON",
            "defaultextension": ".json",
            "initialfile": Path(initial).name if initial else "transcript.manual_retry.json",
            "filetypes": PRESET_FILETYPES,
        }
        if initial:
            kwargs["initialdir"] = str(Path(initial).parent)
        path = filedialog.asksaveasfilename(**kwargs)
        if path:
            self.retry_output.set(path)

    def _infer_retry_audio_from_json(self, json_path: Path) -> Path | None:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            meta = payload.get("meta") if isinstance(payload, dict) else {}
            if isinstance(meta, dict):
                mix = meta.get("speaker_track_mix")
                if isinstance(mix, dict) and mix.get("path"):
                    p = Path(str(mix.get("path")))
                    if p.exists():
                        return p
                audio = meta.get("audio")
                if isinstance(audio, str) and audio:
                    p = Path(audio)
                    if p.exists():
                        return p
        except Exception:
            pass
        guessed = json_path.with_name(json_path.stem + "_mix.wav")
        if guessed.exists():
            return guessed
        return None

    def update_retry_defaults_from_json(self, json_path: Path) -> None:
        if not self.retry_output.get().strip():
            self.retry_output.set(str(json_path.with_name(json_path.stem + ".manual_retry.json")))
        if not self.retry_audio.get().strip():
            audio = self._infer_retry_audio_from_json(json_path)
            if audio is not None:
                self.retry_audio.set(str(audio))

    def set_retry_from_current_output(self) -> None:
        self.sync_output_paths()
        json_path = Path(self.output.get().strip())
        self.retry_json.set(str(json_path))
        self.retry_output.set(str(json_path.with_name(json_path.stem + ".manual_retry.json")))
        self.retry_audio.set("")
        audio = self._infer_retry_audio_from_json(json_path)
        if audio is not None:
            self.retry_audio.set(str(audio))
        self.notebook.select(self.retry_tab)

    def _hotwords_lines(self) -> list[str]:
        return [line.strip() for line in self.hotwords_text.get("1.0", END).splitlines() if line.strip()]

    def _estimate_hotwords_tokens(self, lines: list[str]) -> int:
        # faster-whisper は hotwords を最大約223 tokensまでpromptに入れる。
        # GUIではモデルロードを避けるため、文字種ベースの安全側推定にする。
        total = 0
        for line in lines:
            ascii_chunks = re.findall(r"[A-Za-z0-9_+./:-]+", line)
            ascii_chars = sum(len(chunk) for chunk in ascii_chunks)
            non_ascii_chars = len(re.sub(r"[A-Za-z0-9_+./:-]|\s", "", line))
            # 英数字はざっくり4文字/token、日本語・記号は安全側に1文字/token。
            total += max(1, (ascii_chars + 3) // 4 + non_ascii_chars)
            total += 1  # 区切りぶんの余裕
        return total

    def hotwords_warnings(self) -> list[str]:
        lines = self._hotwords_lines()
        token_estimate = self._estimate_hotwords_tokens(lines)
        warnings: list[str] = []
        if token_estimate >= HOTWORDS_TOKEN_LIMIT:
            warnings.append(
                f"用語リストが多すぎる可能性があります: 推定{token_estimate} tokens / 目安上限{HOTWORDS_TOKEN_LIMIT}。後半は効かない場合があります"
            )
        elif token_estimate >= HOTWORDS_WARN_TOKENS:
            warnings.append(
                f"用語リストが多めです: 推定{token_estimate} tokens / 推奨{HOTWORDS_WARN_TOKENS}以下"
            )
        if len(lines) > HOTWORDS_WARN_LINES:
            warnings.append(f"用語数が多めです: {len(lines)}件 / 推奨{HOTWORDS_WARN_LINES}件以下")
        return warnings

    def update_hotwords_status(self) -> None:
        lines = self._hotwords_lines()
        token_estimate = self._estimate_hotwords_tokens(lines)
        self.hotwords_status_text.set(
            f"用語: {len(lines)}件 / 推定{token_estimate} tokens（推奨{HOTWORDS_WARN_TOKENS}以下）"
        )
        if token_estimate >= HOTWORDS_TOKEN_LIMIT or len(lines) > HOTWORDS_WARN_LINES:
            self.hotwords_status_label.config(foreground="#b00020")
        elif token_estimate >= HOTWORDS_WARN_TOKENS:
            self.hotwords_status_label.config(foreground="#b26b00")
        else:
            self.hotwords_status_label.config(foreground="#555")

    def on_hotwords_modified(self, _event: object | None = None) -> None:
        if self.hotwords_text.edit_modified():
            self.hotwords_text.edit_modified(False)
            self.update_hotwords_status()

    def warn_hotwords_if_needed(self) -> None:
        warnings = self.hotwords_warnings()
        if warnings:
            messagebox.showwarning("用語リスト警告", "\n".join(warnings) + "\n\n実行は継続できます。")

    def load_hotwords_text_file(self) -> None:
        path = filedialog.askopenfilename(
            title="専門用語・固有名詞リストを読み込み",
            filetypes=[("Text", "*.txt *.md"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError:
            text = Path(path).read_text(encoding="cp932").strip()
        self.hotwords_text.delete("1.0", END)
        self.hotwords_text.insert("1.0", text)
        self.hotwords_text.edit_modified(False)
        self.update_hotwords_status()
        self.hotwords_file.set(path)
        self._save_settings()
        messagebox.showinfo("用語リスト読込", f"読み込みました:\n{path}")

    def save_hotwords_text_file(self) -> None:
        path = filedialog.asksaveasfilename(
            title="専門用語・固有名詞リストを保存",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("All", "*.*")],
        )
        if not path:
            return
        text = self.hotwords_text.get("1.0", END).strip()
        atomic_write_text(path, text + ("\n" if text else ""), encoding="utf-8")
        self.hotwords_file.set(path)
        self.update_hotwords_status()
        self._save_settings()
        messagebox.showinfo("用語リスト保存", f"保存しました:\n{path}")

    def add_speaker_track(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=SPEAKER_TRACK_FILETYPES)
        for path in paths:
            self._add_speaker_track_path(path, refresh=False)
        self.refresh_speaker_tree()
        self.on_mode_change()

    def save_as(self, var: StringVar, suffix: str) -> None:
        path = filedialog.asksaveasfilename(defaultextension=suffix)
        if path:
            var.set(path)


    def _add_speaker_track_path(self, path: str | Path, *, refresh: bool = True) -> None:
        audio = Path(path)
        if audio.suffix.lower() != ".wav":
            messagebox.showerror("話者別トラック", f"話者別トラックはWAVのみ対応です:\n{audio}")
            return
        label = f"TRACK_{len(self.speaker_tracks) + 1}"
        self.speaker_tracks.append((label, audio))
        if not self.output_dir.get().strip():
            self.output_dir_auto = True
        self.input_check_text.set("入力未チェック")
        if refresh:
            self.refresh_speaker_tree()
            self.on_mode_change()

    def refresh_speaker_tree(self) -> None:
        for item in self.speaker_tree.get_children():
            self.speaker_tree.delete(item)
        for i, (label, path) in enumerate(self.speaker_tracks):
            self.speaker_tree.insert("", END, iid=str(i), values=(label, str(path)))
        if len(self.speaker_tracks) >= 2:
            self.input_check_text.set("入力未チェック")
        self.sync_output_paths()

    def on_speaker_select(self, _event: object | None = None) -> None:
        selected = self.speaker_tree.selection()
        if not selected:
            self.selected_speaker_label.set("")
            return
        idx = int(selected[0])
        if 0 <= idx < len(self.speaker_tracks):
            self.selected_speaker_label.set(self.speaker_tracks[idx][0])

    def apply_selected_speaker_label(self) -> None:
        selected = self.speaker_tree.selection()
        if not selected:
            messagebox.showinfo("話者名", "名前を変える行を選択してください")
            return
        label = self.selected_speaker_label.get().strip()
        if not label:
            messagebox.showerror("話者名", "話者名を入力してください")
            return
        idx = int(selected[0])
        if 0 <= idx < len(self.speaker_tracks):
            _old_label, path = self.speaker_tracks[idx]
            self.speaker_tracks[idx] = (label, path)
            self.refresh_speaker_tree()
            self.speaker_tree.selection_set(str(idx))
            self.input_check_text.set("入力未チェック")

    def remove_selected_speaker_tracks(self) -> None:
        selected = sorted((int(i) for i in self.speaker_tree.selection()), reverse=True)
        if not selected:
            return
        for idx in selected:
            if 0 <= idx < len(self.speaker_tracks):
                del self.speaker_tracks[idx]
        self.selected_speaker_label.set("")
        self.refresh_speaker_tree()
        self.input_check_text.set("入力未チェック")

    def clear_speaker_tracks(self) -> None:
        self.speaker_tracks.clear()
        self.selected_speaker_label.set("")
        self.refresh_speaker_tree()
        self.input_check_text.set("話者別トラックを2本以上追加してください")
        self.output_dir_auto = True
        self.output_dir.set("")
        self.sync_output_paths()

    def _event_file_paths(self, data: str) -> list[Path]:
        try:
            items = self.root.tk.splitlist(data)
        except Exception:
            items = [data]
        paths: list[Path] = []
        for item in items:
            value = str(item).strip().strip("{}")
            if value:
                paths.append(Path(value))
        return paths

    def _on_speaker_tracks_drop(self, event: object) -> str:
        if self.is_running():
            return "break"
        data = getattr(event, "data", "")
        for path in self._event_file_paths(str(data)):
            self._add_speaker_track_path(path, refresh=False)
        self.refresh_speaker_tree()
        self.on_mode_change()
        return "break"

    def _setup_drag_and_drop(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES  # type: ignore
        except Exception:
            self.input_check_text.set("話者別トラックを2本以上追加してください（ドラッグ&ドロップは未有効）")
            return
        try:
            self.speaker_tree.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
            self.speaker_tree.dnd_bind("<<Drop>>", self._on_speaker_tracks_drop)  # type: ignore[attr-defined]
        except Exception as e:
            self.input_check_text.set(f"ドラッグ&ドロップ初期化失敗: {e}")

    def _set_speaker_tracks_from_lines(self, lines: list[object]) -> None:
        self.speaker_tracks.clear()
        invalid: list[str] = []
        for i, raw in enumerate(lines, start=1):
            line = str(raw).strip().strip('"')
            if not line:
                continue
            if "=" in line:
                label, path = line.split("=", 1)
                label = label.strip() or f"TRACK_{i}"
            else:
                label, path = f"TRACK_{i}", line
            audio_path = Path(path.strip().strip('"'))
            if audio_path.suffix.lower() != ".wav":
                invalid.append(str(audio_path))
                continue
            self.speaker_tracks.append((label, audio_path))
        self.refresh_speaker_tree()
        if invalid:
            messagebox.showwarning("話者別トラック", "WAV以外は読み込みませんでした:\n" + "\n".join(invalid[:8]))

    def _preset_payload(self) -> dict[str, object]:
        self.sync_output_paths()
        return {
            "version": 2,
            "speaker_tracks": self._speaker_tracks(),
            "output_dir": self.output_dir.get().strip(),
            "output_dir_auto": self.output_dir_auto,
            "output_name": self.output_name.get().strip(),
            "crosstalk_cancel": self.crosstalk_cancel.get(),
            "model": self.model.get().strip(),
            "language": self.language.get().strip(),
            "fusion_version": self._fusion_version_value(),
            "initial_prompt": self.prompt_text.get("1.0", END).strip(),
            "hotwords_file": self.hotwords_file.get().strip(),
            "hotwords": self.hotwords_text.get("1.0", END).strip(),
            "speaker_track_active_db": self.speaker_track_active_db.get().strip(),
            "speaker_track_overlap_db": self.speaker_track_overlap_db.get().strip(),
            "speaker_track_margin": self.speaker_track_margin.get().strip(),
            "allow_duration_mismatch": self.allow_duration_mismatch.get(),
        }

    def save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            title="プリセット保存",
            defaultextension=".json",
            filetypes=PRESET_FILETYPES,
        )
        if not path:
            return
        atomic_write_text(path, json.dumps(self._preset_payload(), ensure_ascii=False, indent=2), encoding="utf-8-sig")
        messagebox.showinfo("プリセット保存", f"保存しました:\n{path}")

    def load_preset(self) -> None:
        path = filedialog.askopenfilename(title="プリセット読込", filetypes=PRESET_FILETYPES)
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        self.apply_job_payload(data)
        messagebox.showinfo("プリセット読込", f"読み込みました:\n{path}")

    def apply_job_payload(self, data: dict[str, object]) -> None:
        self.output_dir.set(str(data.get("output_dir", "")))
        self.output_dir_auto = bool(data.get("output_dir_auto", not bool(data.get("output_dir"))))
        self.output_name.set(str(data.get("output_name", DEFAULT_OUTPUT_NAME)) or DEFAULT_OUTPUT_NAME)
        self.crosstalk_cancel.set(bool(data.get("crosstalk_cancel", True)))
        self.model.set(str(data.get("model", "large-v3")))
        self.language.set(str(data.get("language", "ja")))
        self.fusion_version.set(FUSION_VERSION_CHOICES[1] if str(data.get("fusion_version", "v1")).lower() == "v2" else FUSION_VERSION_CHOICES[0])
        self.hotwords_file.set(str(data.get("hotwords_file", "")))
        self.speaker_track_active_db.set(str(data.get("speaker_track_active_db", "auto")))
        self.speaker_track_overlap_db.set(str(data.get("speaker_track_overlap_db", "auto")))
        self.speaker_track_margin.set(str(data.get("speaker_track_margin", "auto")))
        self.allow_duration_mismatch.set(bool(data.get("allow_duration_mismatch", False)))

        self.prompt_text.delete("1.0", END)
        self.prompt_text.insert("1.0", str(data.get("initial_prompt", DEFAULT_PROMPT)))
        self.hotwords_text.delete("1.0", END)
        self.hotwords_text.insert("1.0", str(data.get("hotwords", "")))
        self.hotwords_text.edit_modified(False)
        self.update_hotwords_status()
        tracks = data.get("speaker_tracks", [])
        if isinstance(tracks, list):
            self._set_speaker_tracks_from_lines(tracks)
        self.sync_output_paths()
        self.input_check_text.set("プリセット読込済み。実行前に入力チェックしてください")
        self.input_check_label.config(foreground="#555")

    def add_current_to_batch(self) -> None:
        try:
            self.validate_inputs(show_success=False)
        except Exception as e:
            messagebox.showerror("キュー追加エラー", str(e))
            return
        job = self._preset_payload()
        job["job_name"] = self.output_name.get().strip() or f"job_{len(self.batch_jobs) + 1}"
        self.batch_jobs.append(job)
        self.refresh_batch_tree()
        self.batch_status_text.set(f"バッチキュー: {len(self.batch_jobs)}件")

    _STATUS_LABELS: dict[str, str] = {
        "pending": "未実行",
        "running": "実行中",
        "done": "完了",
        "failed": "失敗",
        "stopped": "停止",
        "paused": "一時停止",
    }

    def refresh_batch_tree(self) -> None:
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        for i, job in enumerate(self.batch_jobs):
            tracks = job.get("speaker_tracks") if isinstance(job.get("speaker_tracks"), list) else []
            name = str(job.get("job_name") or job.get("output_name") or f"job_{i + 1}")
            output_dir = str(job.get("output_dir") or "")
            output_name = str(job.get("output_name") or DEFAULT_OUTPUT_NAME)
            output = str(Path(output_dir) / f"{output_name}.json") if output_dir else output_name
            status = str(job.get("status") or "pending").lower()
            status_label = self._STATUS_LABELS.get(status, status)
            model = str(job.get("model") or "large-v3")
            lang = str(job.get("language") or "ja")
            fusion = "v2" if str(job.get("fusion_version", "v1")).lower() == "v2" else "v1"
            crosstalk = "ON" if bool(job.get("crosstalk_cancel", True)) else "-"
            self.batch_tree.insert(
                "",
                END,
                iid=str(i),
                values=(status_label, name, len(tracks), model, lang, fusion, crosstalk, output),
                tags=(f"status_{status}",),
            )
        self.batch_status_text.set(f"バッチキュー: {len(self.batch_jobs)}件")

    def remove_selected_batch_jobs(self) -> None:
        selected = sorted((int(i) for i in self.batch_tree.selection()), reverse=True)
        for idx in selected:
            if 0 <= idx < len(self.batch_jobs):
                del self.batch_jobs[idx]
        self.refresh_batch_tree()

    def clear_batch_jobs(self) -> None:
        self.batch_jobs.clear()
        self.refresh_batch_tree()

    def save_batch_queue(self) -> None:
        path = filedialog.asksaveasfilename(
            title="キュー保存",
            defaultextension=".json",
            filetypes=PRESET_FILETYPES,
        )
        if not path:
            return
        payload = {"version": 1, "jobs": self.batch_jobs}
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
        messagebox.showinfo("キュー保存", f"保存しました:\n{path}")

    def load_batch_queue(self) -> None:
        path = filedialog.askopenfilename(title="キュー読込", filetypes=PRESET_FILETYPES)
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        jobs = payload.get("jobs") if isinstance(payload, dict) else payload
        if not isinstance(jobs, list):
            messagebox.showerror("キュー読込", "jobs 配列が見つかりません")
            return
        self.batch_jobs = [dict(job) for job in jobs if isinstance(job, dict)]
        self.refresh_batch_tree()
        messagebox.showinfo("キュー読込", f"{len(self.batch_jobs)}件読み込みました")

    def _settings_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "output_dir": "" if self.output_dir_auto else self.output_dir.get().strip(),
            "output_dir_auto": self.output_dir_auto,
            "output_name": self.output_name.get().strip() or DEFAULT_OUTPUT_NAME,
            "model": self.model.get().strip() or "large-v3",
            "language": self.language.get().strip() or "ja",
            "fusion_version": self._fusion_version_value(),
            "crosstalk_cancel": self.crosstalk_cancel.get(),
            "speaker_track_active_db": self.speaker_track_active_db.get().strip(),
            "speaker_track_overlap_db": self.speaker_track_overlap_db.get().strip(),
            "speaker_track_margin": self.speaker_track_margin.get().strip(),
            "allow_duration_mismatch": self.allow_duration_mismatch.get(),
            "hotwords_file": self.hotwords_file.get().strip(),
            "batch_output_dir": self.batch_output_dir.get().strip(),
            "batch_recursive": self.batch_recursive.get(),
            "batch_skip_existing": self.batch_skip_existing.get(),
            "retry_json": self.retry_json.get().strip(),
            "retry_audio": self.retry_audio.get().strip(),
            "retry_output": self.retry_output.get().strip(),
        }

    def _save_settings(self) -> None:
        try:
            atomic_write_text(SETTINGS_PATH, json.dumps(self._settings_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            self.sync_output_paths()
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            self.sync_output_paths()
            return
        if not isinstance(data, dict):
            self.sync_output_paths()
            return
        self.output_name.set(str(data.get("output_name") or DEFAULT_OUTPUT_NAME))
        self.output_dir_auto = bool(data.get("output_dir_auto", True))
        self.output_dir.set(str(data.get("output_dir") or ""))
        self.model.set(str(data.get("model") or "large-v3"))
        self.language.set(str(data.get("language") or "ja"))
        self.fusion_version.set(FUSION_VERSION_CHOICES[1] if str(data.get("fusion_version", "v1")).lower() == "v2" else FUSION_VERSION_CHOICES[0])
        self.crosstalk_cancel.set(bool(data.get("crosstalk_cancel", True)))
        self.speaker_track_active_db.set(str(data.get("speaker_track_active_db") or "auto"))
        self.speaker_track_overlap_db.set(str(data.get("speaker_track_overlap_db") or "auto"))
        self.speaker_track_margin.set(str(data.get("speaker_track_margin") or "auto"))
        self.allow_duration_mismatch.set(bool(data.get("allow_duration_mismatch", False)))
        self.hotwords_file.set(str(data.get("hotwords_file") or ""))
        self.batch_output_dir.set(str(data.get("batch_output_dir") or ""))
        self.batch_recursive.set(bool(data.get("batch_recursive", False)))
        self.batch_skip_existing.set(bool(data.get("batch_skip_existing", True)))
        self.retry_json.set(str(data.get("retry_json") or ""))
        self.retry_audio.set(str(data.get("retry_audio") or ""))
        self.retry_output.set(str(data.get("retry_output") or ""))
        self.sync_output_paths()
        self.update_hotwords_status()

    def on_close(self) -> None:
        self._save_settings()
        self.root.destroy()

    def toggle_section(self, name: str) -> None:
        if name == "advanced":
            self.notebook.select(self.advanced_tab)
        elif name == "log":
            self.notebook.select(self.log_tab)
        elif name == "batch":
            self.notebook.select(self.batch_tab)
        else:
            self.notebook.select(self.input_tab)
        self.root.update_idletasks()

    def _temp_text_file(self, content: str, suffix: str) -> Path:
        f = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=False)
        with f:
            f.write(content.strip())
        path = Path(f.name)
        self.temp_files.append(path)
        return path

    def _source_parent_for_default_output(self) -> Path:
        if self.speaker_tracks:
            return self.speaker_tracks[0][1].parent
        return ROOT

    def sync_output_paths(self) -> None:
        name = self.output_name.get().strip() or DEFAULT_OUTPUT_NAME
        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "_")
        self.output_name.set(name)
        out_dir = self.output_dir.get().strip()
        if self.output_dir_auto or not out_dir:
            out_dir = str(self._source_parent_for_default_output() / name)
            self.output_dir.set(out_dir)
        base = Path(out_dir) / name
        self.output.set(str(base.with_suffix(".json")))
        self.text_output.set(str(base.with_name(base.name + ".review.md")))

    def _common_options(
        self,
        cmd: list[str],
        *,
        include_pyannote: bool = True,
        include_speaker_tracks: bool = False,
    ) -> list[str]:
        cmd += [
            "--whisper-model",
            self.model.get().strip() or "large-v3",
            "--language",
            self.language.get().strip() or "ja",
        ]
        if include_pyannote and self.diarize.get():
            cmd.append("--diarize")
        if include_pyannote and self.min_speakers.get().strip():
            cmd += ["--min-speakers", self.min_speakers.get().strip()]
        if include_pyannote and self.max_speakers.get().strip():
            cmd += ["--max-speakers", self.max_speakers.get().strip()]
        if include_pyannote and self.hf_token.get().strip():
            cmd += ["--hf-token", self.hf_token.get().strip()]

        prompt = self.prompt_text.get("1.0", END).strip()
        if prompt:
            cmd += ["--prompt-file", str(self._temp_text_file(prompt, "_prompt.txt"))]

        hotwords_file = self.hotwords_file.get().strip()
        hotwords = self.hotwords_text.get("1.0", END).strip()
        if hotwords:
            cmd += ["--hotwords-file", str(self._temp_text_file(hotwords, "_hotwords.txt"))]
        elif hotwords_file:
            cmd += ["--hotwords-file", hotwords_file]

        if include_speaker_tracks:
            cmd = self._append_speaker_track_options(cmd)
            cmd += ["--fusion-version", self._fusion_version_value()]
            if not self.crosstalk_cancel.get():
                cmd.append("--no-crosstalk-cancel")
        return cmd

    def _append_speaker_track_options(self, cmd: list[str]) -> list[str]:
        for line in self._speaker_tracks():
            cmd += ["--speaker-track", line]
        if self.speaker_track_active_db.get().strip():
            cmd += ["--speaker-track-active-db", self.speaker_track_active_db.get().strip()]
        if self.speaker_track_overlap_db.get().strip():
            cmd += ["--speaker-track-overlap-db", self.speaker_track_overlap_db.get().strip()]
        if self.speaker_track_margin.get().strip():
            cmd += ["--speaker-track-margin", self.speaker_track_margin.get().strip()]
        cmd += ["--duration-mismatch-tolerance", str(DURATION_MISMATCH_TOLERANCE)]
        if self.allow_duration_mismatch.get():
            cmd.append("--allow-duration-mismatch")
        return cmd

    def _batch_files(self) -> list[str]:
        lines = [line.strip().strip('"') for line in self.batch_text.get("1.0", END).splitlines()]
        return [line for line in lines if line]

    def _speaker_tracks(self) -> list[str]:
        return [f"{label}={path}" for label, path in self.speaker_tracks]

    def _speaker_track_pairs(self) -> list[tuple[str, Path]]:
        return list(self.speaker_tracks)

    def _fusion_version_value(self) -> str:
        value = self.fusion_version.get().strip().lower()
        return "v2" if value.startswith("v2") else "v1"

    def _format_alignment_summary(self, meta: dict[str, object]) -> str:
        audio = meta.get("audio") if isinstance(meta.get("audio"), dict) else None
        track_duration = meta.get("track_duration_sec")
        parts: list[str] = []
        if audio:
            parts.append(f"mix {float(audio.get('duration_sec', 0.0)):.1f}秒")
        else:
            parts.append("mix 自動作成")
        if isinstance(track_duration, (int, float)):
            parts.append(f"tracks {float(track_duration):.1f}秒")
        tracks = meta.get("speaker_tracks")
        if isinstance(tracks, list):
            parts.append(f"{len(tracks)}トラック")
        return " / ".join(parts)

    def inspect_inputs(self) -> dict[str, object]:
        from core.review.transcribe_audio import inspect_audio_alignment

        speaker_tracks = self._speaker_track_pairs()
        if len(speaker_tracks) < 2:
            raise ValueError("話者別トラックを2本以上追加してください")
        return inspect_audio_alignment(
            None,
            speaker_tracks,
            tolerance_sec=DURATION_MISMATCH_TOLERANCE,
        )

    def validate_inputs(self, *, show_success: bool = False) -> dict[str, object]:
        from core.review.transcribe_audio import validate_audio_alignment

        speaker_tracks = self._speaker_track_pairs()
        if len(speaker_tracks) < 2:
            raise ValueError("話者別トラックを2本以上追加してください")
        meta = validate_audio_alignment(
            None,
            speaker_tracks,
            tolerance_sec=DURATION_MISMATCH_TOLERANCE,
            allow_mismatch=self.allow_duration_mismatch.get(),
        )
        summary = self._format_alignment_summary(meta)
        warnings = meta.get("warnings") or []
        warnings = [*warnings, *self.hotwords_warnings()]
        errors = meta.get("errors") or []
        if errors and self.allow_duration_mismatch.get():
            self.input_check_text.set("強制実行: " + summary + " / " + " / ".join(str(e) for e in errors))
            self.input_check_label.config(foreground="#b26b00")
        elif warnings:
            self.input_check_text.set("警告: " + summary + " / " + " / ".join(str(w) for w in warnings))
            self.input_check_label.config(foreground="#b26b00")
        else:
            self.input_check_text.set("OK: " + summary)
            self.input_check_label.config(foreground="#1f7a1f")
            if show_success:
                messagebox.showinfo("入力チェック", "OK: " + summary)
        return meta

    def check_inputs(self) -> None:
        try:
            meta = self.inspect_inputs()
        except Exception as e:
            self.input_check_text.set("NG: " + str(e))
            self.input_check_label.config(foreground="#b00020")
            messagebox.showerror("入力チェック", str(e))
            return
        summary = self._format_alignment_summary(meta)
        errors = meta.get("errors") or []
        warnings = [*(meta.get("warnings") or []), *self.hotwords_warnings()]
        if errors:
            self.input_check_text.set("NG: " + summary + " / " + " / ".join(str(e) for e in errors))
            self.input_check_label.config(foreground="#b00020")
            messagebox.showerror("入力チェック", "\n".join(str(e) for e in errors))
        elif warnings:
            self.input_check_text.set("警告: " + summary + " / " + " / ".join(str(w) for w in warnings))
            self.input_check_label.config(foreground="#b26b00")
            messagebox.showwarning("入力チェック", "\n".join(str(w) for w in warnings))
        else:
            self.input_check_text.set("OK: " + summary)
            self.input_check_label.config(foreground="#1f7a1f")
            messagebox.showinfo("入力チェック", "OK: " + summary)

    def has_batch_targets(self) -> bool:
        return bool(self._batch_files() or self.batch_folder.get().strip())

    def build_single_command(self) -> list[str]:
        self.sync_output_paths()
        speaker_tracks = self._speaker_tracks()

        cmd = [sys.executable, "-m", "transcribe"]
        if len(speaker_tracks) < 2:
            raise ValueError("話者別トラックを2本以上追加してください")
        cmd.append("--mix-from-speaker-tracks")
        cmd = self._common_options(cmd, include_pyannote=False, include_speaker_tracks=True)

        if self.output.get().strip():
            cmd += ["--output", self.output.get().strip()]
        if self.text_output.get().strip():
            cmd += ["--text-output", self.text_output.get().strip()]
        return cmd

    def build_command_from_job(self, job: dict[str, object]) -> list[str]:
        speaker_tracks = [str(v) for v in job.get("speaker_tracks", []) if str(v).strip()] if isinstance(job.get("speaker_tracks"), list) else []
        if len(speaker_tracks) < 2:
            raise ValueError(f"{job.get('job_name') or job.get('output_name')}: 話者別トラックを2本以上指定してください")

        cmd = [sys.executable, "-m", "transcribe"]
        cmd.append("--mix-from-speaker-tracks")

        cmd += [
            "--whisper-model",
            str(job.get("model") or "large-v3"),
            "--language",
            str(job.get("language") or "ja"),
        ]
        prompt = str(job.get("initial_prompt") or "").strip()
        if prompt:
            cmd += ["--prompt-file", str(self._temp_text_file(prompt, "_prompt.txt"))]
        hotwords = str(job.get("hotwords") or "").strip()
        hotwords_file = str(job.get("hotwords_file") or "").strip()
        if hotwords:
            cmd += ["--hotwords-file", str(self._temp_text_file(hotwords, "_hotwords.txt"))]
        elif hotwords_file:
            cmd += ["--hotwords-file", hotwords_file]

        for line in speaker_tracks:
            cmd += ["--speaker-track", line]
        cmd += ["--fusion-version", "v2" if str(job.get("fusion_version", "v1")).lower() == "v2" else "v1"]
        for key, arg, default in [
            ("speaker_track_active_db", "--speaker-track-active-db", "auto"),
            ("speaker_track_overlap_db", "--speaker-track-overlap-db", "auto"),
            ("speaker_track_margin", "--speaker-track-margin", "auto"),
        ]:
            value = str(job.get(key) or default).strip()
            if value:
                cmd += [arg, value]
        cmd += ["--duration-mismatch-tolerance", str(DURATION_MISMATCH_TOLERANCE)]
        if bool(job.get("allow_duration_mismatch", False)):
            cmd.append("--allow-duration-mismatch")
        if not bool(job.get("crosstalk_cancel", True)):
            cmd.append("--no-crosstalk-cancel")

        out_dir = str(job.get("output_dir") or "").strip()
        out_name = str(job.get("output_name") or DEFAULT_OUTPUT_NAME).strip() or DEFAULT_OUTPUT_NAME
        for ch in '<>:"/\\|?*':
            out_name = out_name.replace(ch, "_")
        if out_dir:
            base = Path(out_dir) / out_name
            cmd += ["--output", str(base.with_suffix(".json"))]
            cmd += ["--text-output", str(base.with_suffix(".md"))]
        return cmd

    def build_batch_command(self) -> list[str]:
        files = self._batch_files()
        folder = self.batch_folder.get().strip()
        if not files and not folder:
            raise ValueError("一括処理: ファイル追加またはフォルダ指定をしてください")
        cmd = [sys.executable, "-m", "transcribe_batch"]
        cmd = self._common_options(cmd, include_pyannote=True, include_speaker_tracks=False)
        if files:
            cmd += ["--audio", *files]
        if folder:
            cmd += ["--input-dir", folder, "--glob", self.batch_glob.get().strip() or "*.wav"]
            if self.batch_recursive.get():
                cmd.append("--recursive")
        if self.batch_output_dir.get().strip():
            cmd += ["--output-dir", self.batch_output_dir.get().strip()]
        if self.batch_skip_existing.get():
            cmd.append("--skip-existing")
        return cmd

    def build_command(self) -> list[str]:
        return self.build_single_command()

    def _format_duration(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h:
            return f"{h}時間{m:02d}分{s:02d}秒"
        if m:
            return f"{m}分{s:02d}秒"
        return f"{s}秒"

    def _mark_run_started(self, kind: str) -> None:
        self.current_run_started = time.perf_counter()
        self.current_run_kind = kind

    def _record_run_duration(self, status: str) -> None:
        if self.current_run_started is None:
            return
        elapsed = time.perf_counter() - self.current_run_started
        timestamp = time.strftime("%H:%M")
        label = self.current_run_kind or "処理"
        line = f"{timestamp} {label}: {self._format_duration(elapsed)}（{status}）"
        self.run_duration_history.insert(0, line)
        self.run_duration_history = self.run_duration_history[:5]
        self.run_duration_text.set("処理時間履歴: " + " / ".join(self.run_duration_history))
        self.current_run_started = None
        self.current_run_kind = ""

    def build_manual_retry_command(self) -> list[str]:
        json_path = self.retry_json.get().strip()
        audio_path = self.retry_audio.get().strip()
        if not json_path:
            raise ValueError("retry対象JSONを指定してください")
        if not audio_path:
            raise ValueError("retry用ASR対象音声/mixを指定してください")
        if not Path(json_path).exists():
            raise ValueError(f"retry対象JSONが見つかりません: {json_path}")
        if not Path(audio_path).exists():
            raise ValueError(f"retry用音声が見つかりません: {audio_path}")
        output = self.retry_output.get().strip()
        if not output:
            output = str(Path(json_path).with_name(Path(json_path).stem + ".manual_retry.json"))
            self.retry_output.set(output)
        cmd = [
            sys.executable,
            "-m",
            "retry_transcript",
            "--json",
            json_path,
            "--audio",
            audio_path,
            "--output",
            output,
            "--whisper-model",
            self.model.get().strip() or "large-v3",
            "--language",
            self.language.get().strip() or "ja",
        ]
        start = self.retry_start.get().strip()
        end = self.retry_end.get().strip()
        if start:
            float(start)
            cmd += ["--start", start]
        if end:
            float(end)
            cmd += ["--end", end]
        if self.retry_force_replace.get():
            cmd.append("--force-replace")
        return cmd

    def run_manual_retry(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("実行中", "すでに実行中です")
            return
        try:
            cmd = self.build_manual_retry_command()
        except Exception as e:
            messagebox.showerror("手動retry入力エラー", str(e))
            return
        self.batch_running = False
        self.current_batch_index = 0
        self.current_batch_total = 0
        self.stop_requested = False
        self.paused = False
        self.final_status = None
        self.refresh_runtime_status()
        self.log.insert(END, "[gui] manual retry\n")
        self.log.insert(END, "[gui] " + format_command_for_log(cmd) + "\n")
        self._mark_run_started("手動retry")
        self.run_button.config(state="disabled")
        self.run_batch_button.config(state="disabled")
        self.retry_run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.pause_button.config(state="normal", text="一時停止")
        self.set_inputs_locked(True)
        self.reset_progress(batch=False)
        self.worker = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        self.worker.start()
        self.root.after(100, self.poll_output)

    def _ensure_output_dir(self, out_dir: str) -> None:
        """出力先フォルダが無ければ作成する。失敗時は例外。"""
        path = Path(out_dir).expanduser()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self.log.insert(END, f"[gui] 出力フォルダを作成: {path}\n")

    def run(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("実行中", "すでに実行中です")
            return
        try:
            self.validate_inputs(show_success=False)
            cmd = self.build_command()
            out_dir = self.output_dir.get().strip()
            if out_dir:
                self._ensure_output_dir(out_dir)
        except Exception as e:
            messagebox.showerror("入力エラー", str(e))
            return
        self.warn_hotwords_if_needed()
        self.batch_running = False
        self.current_batch_index = 0
        self.current_batch_total = 0
        self.stop_requested = False
        self.paused = False
        self.final_status = None
        self.refresh_runtime_status()
        self.log.insert(END, "[gui] " + self.runtime_text.get() + "\n")
        self.log.insert(END, "[gui] " + format_command_for_log(cmd) + "\n")
        self._mark_run_started("文字起こし")
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.pause_button.config(state="normal", text="一時停止")
        self.set_inputs_locked(True)
        self.reset_progress(batch=False)
        self.worker = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        self.worker.start()
        self.root.after(100, self.poll_output)

    def run_batch(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("実行中", "すでに実行中です")
            return
        if not self.batch_jobs:
            messagebox.showerror("バッチ実行", "キューが空です。現在の設定を追加してください")
            return
        missing: list[str] = []
        for job in self.batch_jobs:
            tracks = job.get("speaker_tracks", [])
            if isinstance(tracks, list):
                for track in tracks:
                    s = str(track)
                    _, _, path_part = s.partition("=")
                    path_str = (path_part if path_part else s).strip().strip('"')
                    if path_str and Path(path_str).suffix.lower() != ".wav":
                        name = str(job.get("job_name") or job.get("output_name") or "?")
                        missing.append(f"{name}: WAV以外は未対応 {path_str}")
                        continue
                    if path_str and not Path(path_str).exists():
                        name = str(job.get("job_name") or job.get("output_name") or "?")
                        missing.append(f"{name}: {path_str}")
        if missing:
            messagebox.showerror(
                "バッチ実行エラー",
                "以下のファイルが見つかりません:\n" + "\n".join(missing[:8]),
            )
            return

        # 全ジョブの出力フォルダを事前に作成
        try:
            for job in self.batch_jobs:
                out_dir = str(job.get("output_dir") or "").strip()
                if out_dir:
                    self._ensure_output_dir(out_dir)
        except Exception as e:
            messagebox.showerror("出力フォルダ作成エラー", str(e))
            return

        try:
            queue_path = self._temp_text_file(
                json.dumps({"version": 1, "jobs": self.batch_jobs}, ensure_ascii=False, indent=2),
                "_job_queue.json",
            )
            cmd = [sys.executable, "-m", "transcribe_job_queue", "--job-queue", str(queue_path)]
        except Exception as e:
            messagebox.showerror("バッチ実行エラー", str(e))
            return

        self.batch_running = True
        self.current_batch_index = 0
        self.current_batch_total = len(self.batch_jobs)
        self.stop_requested = False
        self.paused = False
        self.final_status = None
        # ジョブステータスをリセット
        for job in self.batch_jobs:
            job["status"] = "pending"
        self.refresh_batch_tree()
        self.refresh_runtime_status()
        self.log.insert(END, "[gui] " + self.runtime_text.get() + "\n")
        self.log.insert(END, f"[gui] batch start: {len(self.batch_jobs)} jobs (single process/model reuse)\n")
        self.log.insert(END, "[gui] " + format_command_for_log(cmd) + "\n")
        self._mark_run_started(f"バッチ{len(self.batch_jobs)}件")
        self.run_button.config(state="disabled")
        self.run_batch_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.pause_button.config(state="normal", text="一時停止")
        self.set_inputs_locked(True)
        self.reset_progress(batch=True)
        self.progress_bar.config(mode="determinate", maximum=100, value=0)
        self.progress_text.set(f"0 / {len(self.batch_jobs)}")
        self.worker = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        self.worker.start()
        self.root.after(100, self.poll_output)

    def reset_progress(self, batch: bool) -> None:
        self.progress_total = 0
        self.progress_current = 0
        self.status_text.set("実行中…")
        self.progress_text.set("一括: 準備中" if batch else "0% 準備中")
        self.progress_bar.stop()
        if batch:
            self.progress_bar.config(mode="determinate", maximum=1, value=0)
        else:
            self.progress_bar.config(mode="determinate", maximum=100, value=0)

    def update_progress_from_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith("[progress] "):
            self.update_progress_from_json(stripped.removeprefix("[progress] ").strip())
            return

        if (
            stripped.startswith("[gui] batch job ")
            and not stripped.startswith("[gui] batch job done ")
            and not stripped.startswith("[gui] batch job failed ")
            and "/" in stripped
        ):
            try:
                marker = stripped.removeprefix("[gui] batch job ").split(":", 1)[0]
                current_text, total_text = marker.split("/", 1)
                self.current_batch_index = int(current_text)
                self.current_batch_total = int(total_text)
                self.status_text.set(f"バッチ実行中… {self.current_batch_index}/{self.current_batch_total}")
                self.progress_text.set(f"{self.current_batch_index - 1} / {self.current_batch_total}")
            except ValueError:
                pass
            return

        if stripped.startswith("[gui] batch job done ") and "/" in stripped:
            try:
                marker = stripped.removeprefix("[gui] batch job done ").split(":", 1)[0]
                current_text, total_text = marker.split("/", 1)
                current = int(current_text)
                total = int(total_text)
                self.current_batch_index = current
                self.current_batch_total = total
                value = (current / max(1, total)) * 100.0
                self.progress_bar.config(mode="determinate", maximum=100, value=value)
                self.progress_text.set(f"{current} / {total}")
            except ValueError:
                pass
            return

        if stripped.startswith("[gui] batch job failed ") and "/" in stripped:
            try:
                marker = stripped.removeprefix("[gui] batch job failed ").split(":", 1)[0]
                current_text, total_text = marker.split("/", 1)
                current = int(current_text)
                total = int(total_text)
                self.current_batch_index = current
                self.current_batch_total = total
                value = (current / max(1, total)) * 100.0
                self.progress_bar.config(mode="determinate", maximum=100, value=value)
                self.progress_text.set(f"{current} / {total}")
            except ValueError:
                pass
            return

        if stripped.startswith("[batch] files:"):
            try:
                self.progress_total = int(stripped.rsplit(":", 1)[1].strip())
            except ValueError:
                return
            self.progress_current = 0
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", maximum=max(1, self.progress_total), value=0)
            self.progress_text.set(f"0 / {self.progress_total}")
            return

        if stripped.startswith("[batch] (") and "/" in stripped:
            marker = stripped.removeprefix("[batch] (").removesuffix(")")
            try:
                current_text, total_text = marker.split("/", 1)
                current = int(current_text)
                total = int(total_text)
            except ValueError:
                return
            self.progress_total = total
            self.progress_current = max(0, current - 1)
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", maximum=max(1, total), value=self.progress_current)
            self.status_text.set(f"実行中… {current}/{total}")
            self.progress_text.set(f"{self.progress_current} / {total}")
            return

        if stripped.startswith("[batch] saved:") or stripped.startswith("[batch] skip existing:"):
            if self.progress_total:
                self.progress_current = min(self.progress_total, self.progress_current + 1)
                self.progress_bar.config(value=self.progress_current)
                self.progress_text.set(f"{self.progress_current} / {self.progress_total}")
            return

        if stripped.startswith("[batch] done:") and self.progress_total:
            self.progress_bar.config(value=self.progress_total)
            self.progress_text.set(f"{self.progress_total} / {self.progress_total}")
            return


    def update_progress_from_json(self, payload_text: str) -> None:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            if payload_text:
                self.status_text.set(payload_text[:34])
            return
        message = str(payload.get("message") or "処理中")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        percent = data.get("progress_percent")
        stage_percent = data.get("stage_percent")
        stage = str(data.get("stage") or message)

        self.progress_bar.stop()
        self.progress_bar.config(mode="determinate", maximum=100)
        if isinstance(percent, (int, float)):
            job_value = max(0.0, min(100.0, float(percent)))
            if self.batch_running and self.current_batch_total > 0 and self.current_batch_index > 0:
                value = (
                    (self.current_batch_index - 1 + (job_value / 100.0))
                    / self.current_batch_total
                ) * 100.0
                value = max(0.0, min(100.0, value))
                percent_text = f"全体 {value:.1f}%"
            else:
                value = job_value
                percent_text = f"{value:.1f}%"
            self.progress_bar.config(value=value)
        else:
            value = float(self.progress_bar["value"] or 0)
            percent_text = f"全体 {value:.1f}%" if self.batch_running else f"{value:.1f}%"

        elapsed = str(payload.get("elapsed_text") or "")
        if self.paused:
            self.status_text.set("一時停止中")
        elif self.batch_running and self.current_batch_total > 0 and self.current_batch_index > 0:
            self.status_text.set(f"{stage[:20]} {self.current_batch_index}/{self.current_batch_total}")
        else:
            self.status_text.set(stage[:34])

        detail_parts = [percent_text]
        if isinstance(stage_percent, (int, float)):
            detail_parts.append(f"{stage_percent:.0f}%/{stage[:10]}")
        current_sec = data.get("current_sec") or data.get("written_sec")
        duration_sec = data.get("duration_sec")
        if isinstance(current_sec, (int, float)) and isinstance(duration_sec, (int, float)) and duration_sec > 0:
            detail_parts.append(f"{int(current_sec)}/{int(duration_sec)}秒")
        if elapsed:
            detail_parts.append(elapsed.split()[0])
        self.progress_text.set(" ".join(detail_parts)[:28])

    def update_final_status_from_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith("[gui] stopped:"):
            self.final_status = ("stopped", "停止しました")
        elif stripped.startswith("[gui] failed:"):
            self.final_status = ("failed", "失敗しました")
        elif stripped.startswith("[gui] finished: exit code"):
            code = stripped.rsplit(" ", 1)[-1]
            if code == "0":
                self.final_status = ("ok", "完了しました")
            else:
                self.final_status = ("failed", f"失敗しました exit code {code}")

    def finish_progress(self) -> None:
        self.progress_bar.stop()
        kind, message = self.final_status or ("ok", "完了しました")
        duration_status = {"ok": "完了", "stopped": "停止"}.get(kind, "失敗")
        self._record_run_duration(duration_status)
        if kind == "ok":
            self.status_text.set("完了")
            if self.progress_total:
                self.progress_bar.config(mode="determinate", value=self.progress_total)
            else:
                self.progress_bar.config(mode="determinate", maximum=100, value=100)
            self.progress_text.set("完了")
            self.root.bell()
            messagebox.showinfo("完了", message)
        elif kind == "stopped":
            self.status_text.set("停止")
            self.progress_text.set("停止")
            self.root.bell()
            messagebox.showwarning("停止", message)
        else:
            self.status_text.set("失敗")
            self.progress_text.set("失敗")
            self.root.bell()
            messagebox.showerror("失敗", message)

    def stop(self) -> None:
        self.stop_requested = True
        proc = self.current_proc
        if self.paused and proc is not None and proc.poll() is None:
            try:
                self._set_subprocess_paused(proc.pid, False)
            except Exception as e:
                self.output_queue.put(f"[gui] resume before stop failed: {e}\n")
            self.paused = False
            self.pause_button.config(text="一時停止")
        if proc is None or proc.poll() is not None:
            self.output_queue.put("[gui] 停止対象のプロセスはありません\n")
            return
        self.output_queue.put("[gui] 停止要求: 実行中プロセスを終了します\n")
        try:
            proc.terminate()
        except Exception as e:
            self.output_queue.put(f"[gui] terminate failed: {e}\n")
            try:
                proc.kill()
            except Exception as kill_error:
                self.output_queue.put(f"[gui] kill failed: {kill_error}\n")
        self.root.after(5000, self.force_kill_if_running)


    def toggle_pause(self) -> None:
        proc = self.current_proc
        if proc is None or proc.poll() is not None:
            messagebox.showinfo("一時停止", "実行中の処理がありません")
            return
        try:
            self._set_subprocess_paused(proc.pid, not self.paused)
        except Exception as e:
            messagebox.showerror("一時停止に失敗", str(e))
            return
        self.paused = not self.paused
        if self.paused:
            self.status_text.set("一時停止中")
            self.pause_button.config(text="再開")
            self.output_queue.put("[gui] 一時停止しました\n")
            # バッチ実行中なら running ジョブを paused に
            if self.batch_running:
                for job in self.batch_jobs:
                    if str(job.get("status") or "").lower() == "running":
                        job["status"] = "paused"
                self.refresh_batch_tree()
        else:
            self.status_text.set("実行中…")
            self.pause_button.config(text="一時停止")
            self.output_queue.put("[gui] 再開しました\n")
            if self.batch_running:
                for job in self.batch_jobs:
                    if str(job.get("status") or "").lower() == "paused":
                        job["status"] = "running"
                self.refresh_batch_tree()

    def _set_subprocess_paused(self, pid: int, pause: bool) -> None:
        if os.name != "nt":
            raise RuntimeError("一時停止/再開はこのGUIではWindowsのみ対応です")

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        TH32CS_SNAPTHREAD = 0x00000004
        THREAD_SUSPEND_RESUME = 0x0002
        INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

        class THREADENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_ulong),
                ("cntUsage", ctypes.c_ulong),
                ("th32ThreadID", ctypes.c_ulong),
                ("th32OwnerProcessID", ctypes.c_ulong),
                ("tpBasePri", ctypes.c_long),
                ("tpDeltaPri", ctypes.c_long),
                ("dwFlags", ctypes.c_ulong),
            ]

        kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
        kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        kernel32.Thread32First.argtypes = [ctypes.c_void_p, ctypes.POINTER(THREADENTRY32)]
        kernel32.Thread32First.restype = ctypes.c_int
        kernel32.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(THREADENTRY32)]
        kernel32.Thread32Next.restype = ctypes.c_int
        kernel32.OpenThread.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenThread.restype = ctypes.c_void_p
        kernel32.SuspendThread.argtypes = [ctypes.c_void_p]
        kernel32.SuspendThread.restype = ctypes.c_ulong
        kernel32.ResumeThread.argtypes = [ctypes.c_void_p]
        kernel32.ResumeThread.restype = ctypes.c_ulong
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int

        snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if snapshot == INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())
        affected = 0
        try:
            entry = THREADENTRY32()
            entry.dwSize = ctypes.sizeof(THREADENTRY32)
            ok = kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while ok:
                if int(entry.th32OwnerProcessID) == int(pid):
                    thread = kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                    if thread:
                        try:
                            if pause:
                                result = kernel32.SuspendThread(thread)
                                if result == 0xFFFFFFFF:
                                    raise ctypes.WinError(ctypes.get_last_error())
                            else:
                                result = kernel32.ResumeThread(thread)
                                if result == 0xFFFFFFFF:
                                    raise ctypes.WinError(ctypes.get_last_error())
                                while result > 1:
                                    result = kernel32.ResumeThread(thread)
                                    if result == 0xFFFFFFFF:
                                        raise ctypes.WinError(ctypes.get_last_error())
                            affected += 1
                        finally:
                            kernel32.CloseHandle(thread)
                ok = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)
        if affected == 0:
            raise RuntimeError("対象プロセスのスレッドを見つけられませんでした")

    def force_kill_if_running(self) -> None:
        proc = self.current_proc
        if proc is not None and proc.poll() is None:
            self.output_queue.put("[gui] 強制終了します\n")
            try:
                proc.kill()
            except Exception as e:
                self.output_queue.put(f"[gui] force kill failed: {e}\n")

    def _run_subprocess(self, cmd: list[str]) -> None:
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.current_proc = proc
            if proc.stdout is None:
                self.output_queue.put("[gui] failed: subprocess stdout is None\n")
                try:
                    proc.terminate()
                except Exception:
                    pass
                return
            for line in proc.stdout:
                self.output_queue.put(line)
            code = proc.wait()
            if self.stop_requested:
                self.output_queue.put(f"[gui] stopped: exit code {code}\n")
            else:
                self.output_queue.put(f"[gui] finished: exit code {code}\n")
        except Exception as e:
            self.output_queue.put(f"[gui] failed: {e}\n")
        finally:
            self.current_proc = None
            for path in self.temp_files:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            self.temp_files.clear()
            self.output_queue.put(None)

    def _run_batch_subprocesses(self, commands: list[tuple[str, list[str]]]) -> None:
        failures = 0
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            total = len(commands)
            for index, (name, cmd) in enumerate(commands, start=1):
                if self.stop_requested:
                    break
                self.output_queue.put(f"[gui] batch job {index}/{total}: {name}\n")
                self.output_queue.put("[gui] " + format_command_for_log(cmd) + "\n")
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self.current_proc = proc
                if proc.stdout is None:
                    self.output_queue.put(f"[gui] batch job failed {index}/{total}: {name} stdout is None\n")
                    failures += 1
                    self.current_proc = None
                    continue
                for line in proc.stdout:
                    self.output_queue.put(line)
                code = proc.wait()
                self.current_proc = None
                if self.stop_requested:
                    self.output_queue.put(f"[gui] batch stopped at {index}/{total}: exit code {code}\n")
                    break
                if code == 0:
                    self.output_queue.put(f"[gui] batch job done {index}/{total}: {name}\n")
                else:
                    failures += 1
                    self.output_queue.put(f"[gui] batch job failed {index}/{total}: {name} exit code {code}\n")
            if self.stop_requested:
                self.output_queue.put("[gui] stopped: exit code 1\n")
            elif failures:
                self.output_queue.put(f"[gui] batch finished with failures: {failures}/{len(commands)}\n")
                self.output_queue.put("[gui] finished: exit code 1\n")
            else:
                self.output_queue.put("[gui] finished: exit code 0\n")
        except Exception as e:
            self.output_queue.put(f"[gui] failed: {e}\n")
        finally:
            self.current_proc = None
            for path in self.temp_files:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            self.temp_files.clear()
            self.output_queue.put(None)

    def poll_output(self) -> None:
        done = False
        while True:
            try:
                item = self.output_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                done = True
            else:
                self.update_progress_from_line(item)
                self.update_final_status_from_line(item)
                self.update_batch_status_from_line(item)
                self.log.insert(END, item)
                line_count = int(float(self.log.index("end-1c")))
                if line_count > 3000:
                    self.log.delete("1.0", f"{line_count - 2500}.0")
                self.log.see(END)
        if done:
            self.run_button.config(state="normal")
            self.run_batch_button.config(state="normal")
            self.retry_run_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.pause_button.config(state="disabled", text="一時停止")
            self.set_inputs_locked(False)
            self.paused = False
            self._finalize_batch_status_on_end()
            self.finish_progress()
        else:
            self.root.after(100, self.poll_output)

    _BATCH_JOB_RE = re.compile(r"\[gui\] batch job (?P<kind>done |failed )?(?P<idx>\d+)/(?P<total>\d+):")

    def update_batch_status_from_line(self, line: str) -> None:
        if not self.batch_jobs:
            return
        m = self._BATCH_JOB_RE.search(line)
        if not m:
            return
        kind = (m.group("kind") or "").strip()
        idx = int(m.group("idx"))
        job_idx = idx - 1
        if not (0 <= job_idx < len(self.batch_jobs)):
            return
        if kind == "":
            # 開始 → 直前のジョブを done に、当該を running に
            for prev in range(job_idx):
                if self.batch_jobs[prev].get("status") == "running":
                    self.batch_jobs[prev]["status"] = "done"
            self.batch_jobs[job_idx]["status"] = "running"
        elif kind == "done":
            self.batch_jobs[job_idx]["status"] = "done"
        elif kind == "failed":
            self.batch_jobs[job_idx]["status"] = "failed"
        self.refresh_batch_tree()

    def _finalize_batch_status_on_end(self) -> None:
        if not self.batch_jobs:
            return
        for job in self.batch_jobs:
            status = str(job.get("status") or "pending").lower()
            if status == "running":
                if self.stop_requested:
                    job["status"] = "stopped"
                else:
                    # サブプロセスが終わったのに running のままなら done とみなす
                    job["status"] = "done"
        self.refresh_batch_tree()


def main() -> None:
    try:
        from tkinterdnd2 import TkinterDnD  # type: ignore
        root = TkinterDnD.Tk()
    except Exception:
        root = Tk()
    try:
        TranscribeGui(root)
        root.mainloop()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        crash_log = Path(__file__).parent / "transcribe_gui_crash.log"
        crash_log.write_text(tb, encoding="utf-8")
        try:
            messagebox.showerror("起動エラー", f"予期しないエラーが発生しました。\n\n{crash_log}\n\nを確認してください。\n\n{tb[:500]}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        crash_log = Path(__file__).parent / "transcribe_gui_crash.log"
        crash_log.write_text(tb, encoding="utf-8")
        print(f"[ERROR] クラッシュログ: {crash_log}", file=sys.stderr)
        print(tb, file=sys.stderr)
        sys.exit(1)

