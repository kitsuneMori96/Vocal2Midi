"""Sentence-level repair page: parameter gacha and reference curves.

The gacha reruns one lyric line with many GAME parameter sets and splices the
chosen take back into the base USTX; the curve export writes f0/vuv masked
curves so notes can be drawn by hand in OpenUtau.
"""

from __future__ import annotations

import pathlib

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from application.sentence_jobs import bar_beat_label, export_reference_curves, read_ustx_header
from gui.fluent_compat import (
    BodyLabel,
    CardWidget,
    ComboBox,
    DoubleSpinBox,
    FluentIcon,
    IndeterminateProgressRing,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    SpinBox,
    SubtitleLabel,
)
from gui.fluent_worker import FunctionWorker, HYBRID_AVAILABLE
from gui.i18n import tr
from gui.log_terminal import LogTerminal
from gui.option_tables import (
    DEFAULT_LYRIC_OUTPUT,
    LYRIC_OUTPUT_BY_LANGUAGE,
    TARGET_LANGUAGE_CHOICES,
    fill_combo,
)
from inference.device_utils import VISIBLE_RUNTIME_DEVICE_CHOICES, normalize_runtime_device

GACHA_STRATEGIES = [
    ("random", "repair_strategy_random"),
    ("grid", "repair_strategy_grid"),
    ("repeat", "repair_strategy_repeat"),
]


class SentenceRepairInterface(ScrollArea):
    def __init__(self, global_settings, model_config, parent=None):
        super().__init__(parent=parent)
        self.global_settings = global_settings
        self.model_config = model_config
        self._tr_bindings: list = []
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)
        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)
        self.vBoxLayout.setSpacing(20)
        self.view.setObjectName("view")
        self.setObjectName("sentenceRepairInterface")

        self.worker = None
        self._job_kind = None
        self._job_manifest_path = None
        self._job_output_dir = None
        self._job_base = None
        self._last_manifest = None
        self._take_labels: list[str] = []
        self._lyric_lines: list = []
        self._base_header: dict | None = None
        self._audio_touched = False
        self._output_touched = False

        title = SubtitleLabel(self)
        self._bind_tr(lambda: title.setText(tr("repair_title")))
        self.vBoxLayout.addWidget(title)
        hint = BodyLabel(self)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #808080; font-size: 12px;")
        self._bind_tr(lambda: hint.setText(tr("repair_hint")))
        self.vBoxLayout.addWidget(hint)

        self._build_input_card()
        self._build_target_card()
        self._build_action_card()
        self._build_results_card()
        self._build_status()

        self.vBoxLayout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

        self._refresh_language_options()
        self._refresh_lyric_combo()
        self._update_window_label()

    # ------------------------------------------------------------------ setup
    def _bind_tr(self, fn):
        self._tr_bindings.append(fn)
        fn()

    def retranslate_ui(self):
        for fn in self._tr_bindings:
            fn()
        self._refresh_lyric_combo(keep_value=True)
        self._refresh_language_options(keep_value=True)
        self._retranslate_headers()
        self._update_window_label()

    def _card(self, title_key: str) -> tuple[CardWidget, QVBoxLayout]:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        title = BodyLabel(self)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._bind_tr(lambda t=title, k=title_key: t.setText(tr(k)))
        layout.addWidget(title)
        self.vBoxLayout.addWidget(card)
        return card, layout

    def _label(self, key: str) -> BodyLabel:
        label = BodyLabel(self)
        self._bind_tr(lambda l=label, k=key: l.setText(tr(k)))
        return label

    def _path_row(self, layout: QHBoxLayout, label_key: str, browse_cb, edit=None, stretch=1):
        layout.addWidget(self._label(label_key))
        edit = edit or LineEdit(self)
        layout.addWidget(edit, stretch)
        btn = PushButton(tr("browse"), self, FluentIcon.FOLDER)
        self._bind_tr(lambda b=btn: b.setText(tr("browse")))
        btn.clicked.connect(browse_cb)
        layout.addWidget(btn)
        return edit

    def _build_input_card(self):
        _card, layout = self._card("repair_inputs")

        row = QHBoxLayout()
        self.audio_edit = self._path_row(row, "repair_audio", self._browse_audio)
        self.audio_edit.textChanged.connect(self._on_audio_changed)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.lyrics_edit = self._path_row(row, "repair_lyrics_file", self._browse_lyrics)
        self.lyrics_edit.textChanged.connect(self._on_lyrics_file_changed)
        self.btn_parse = PushButton(tr("repair_parse_lyrics"), self, FluentIcon.SYNC)
        self._bind_tr(lambda: self.btn_parse.setText(tr("repair_parse_lyrics")))
        self.btn_parse.clicked.connect(self._reload_lyrics)
        row.addWidget(self.btn_parse)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.base_edit = self._path_row(row, "repair_base", self._browse_base)
        self.base_edit.textChanged.connect(self._on_base_changed)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.output_edit = self._path_row(row, "repair_output_dir", self._browse_output)
        self.output_edit.textEdited.connect(lambda _t: setattr(self, "_output_touched", True))
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(self._label("repair_tempo"))
        self.tempo_spin = DoubleSpinBox(self)
        self.tempo_spin.setRange(10.0, 400.0)
        self.tempo_spin.setDecimals(2)
        self.tempo_spin.setValue(120.0)
        row.addWidget(self.tempo_spin)
        row.addSpacing(28)
        row.addWidget(self._label("device"))
        self.device_combo = ComboBox(self)
        self.device_combo.addItems(list(VISIBLE_RUNTIME_DEVICE_CHOICES))
        row.addWidget(self.device_combo)
        row.addStretch(1)
        layout.addLayout(row)

    def _build_target_card(self):
        _card, layout = self._card("repair_target")

        row = QHBoxLayout()
        self.line_combo = ComboBox(self)
        self.line_combo.setMinimumWidth(520)
        self.line_combo.currentIndexChanged.connect(self._update_window_label)
        row.addWidget(self.line_combo, 1)
        layout.addLayout(row)

        self.window_label = BodyLabel(self)
        self.window_label.setStyleSheet("color: #808080; font-size: 12px;")
        self.window_label.setWordWrap(True)
        layout.addWidget(self.window_label)

        row = QHBoxLayout()
        row.addWidget(self._label("repair_pad_before"))
        self.pad_before_spin = DoubleSpinBox(self)
        self.pad_before_spin.setRange(0.0, 30.0)
        self.pad_before_spin.setDecimals(2)
        self.pad_before_spin.setValue(0.5)
        self.pad_before_spin.valueChanged.connect(lambda _v: self._update_window_label())
        row.addWidget(self.pad_before_spin)
        row.addSpacing(20)
        row.addWidget(self._label("repair_pad_after"))
        self.pad_after_spin = DoubleSpinBox(self)
        self.pad_after_spin.setRange(0.0, 30.0)
        self.pad_after_spin.setDecimals(2)
        self.pad_after_spin.setValue(0.3)
        self.pad_after_spin.valueChanged.connect(lambda _v: self._update_window_label())
        row.addWidget(self.pad_after_spin)
        row.addSpacing(20)
        row.addWidget(self._label("repair_manual_start"))
        self.start_edit = LineEdit(self)
        self.start_edit.setFixedWidth(96)
        self._bind_tr(lambda: self.start_edit.setPlaceholderText(tr("repair_auto_hint")))
        self.start_edit.textChanged.connect(self._on_manual_window_changed)
        row.addWidget(self.start_edit)
        row.addSpacing(20)
        row.addWidget(self._label("repair_manual_end"))
        self.end_edit = LineEdit(self)
        self.end_edit.setFixedWidth(96)
        self._bind_tr(lambda: self.end_edit.setPlaceholderText(tr("repair_auto_hint")))
        self.end_edit.textChanged.connect(self._on_manual_window_changed)
        row.addWidget(self.end_edit)
        row.addStretch(1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(self._label("target_lang"))
        self.lang_combo = ComboBox(self)
        self.lang_combo.setMinimumWidth(110)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        row.addWidget(self.lang_combo)
        row.addSpacing(28)
        row.addWidget(self._label("lyric_output_format"))
        self.lyric_output_combo = ComboBox(self)
        self.lyric_output_combo.setMinimumWidth(120)
        row.addWidget(self.lyric_output_combo)
        row.addStretch(1)
        layout.addLayout(row)

    def _build_action_card(self):
        _card, layout = self._card("repair_gacha")

        row = QHBoxLayout()
        row.addWidget(self._label("repair_strategy"))
        self.strategy_combo = ComboBox(self)
        self._bind_tr(lambda: fill_combo(self.strategy_combo, GACHA_STRATEGIES, keep_value=True))
        row.addWidget(self.strategy_combo)
        row.addSpacing(20)
        row.addWidget(self._label("repair_takes"))
        self.takes_spin = SpinBox(self)
        self.takes_spin.setRange(1, 64)
        self.takes_spin.setValue(8)
        row.addWidget(self.takes_spin)
        row.addSpacing(20)
        row.addWidget(self._label("repair_seed"))
        self.seed_spin = SpinBox(self)
        self.seed_spin.setRange(0, 999999)
        row.addWidget(self.seed_spin)
        row.addSpacing(20)
        row.addWidget(self._label("repair_engines"))
        self.cb_tifa = QCheckBox("TiFA", self)
        self.cb_hfa = QCheckBox("HubertFA", self)
        self.cb_tifa.setChecked(self.model_config.alignment_engine() != "hfa")
        self.cb_hfa.setChecked(self.model_config.alignment_engine() == "hfa")
        row.addWidget(self.cb_tifa)
        row.addWidget(self.cb_hfa)
        row.addStretch(1)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.btn_gacha = PrimaryPushButton(tr("repair_run_gacha"), self, FluentIcon.PLAY)
        self._bind_tr(lambda: self.btn_gacha.setText(tr("repair_run_gacha")))
        self.btn_gacha.clicked.connect(self.run_gacha)
        row.addWidget(self.btn_gacha, 1)

        self.btn_curves = PushButton(tr("repair_export_curves"), self, FluentIcon.MUSIC)
        self._bind_tr(lambda: self.btn_curves.setText(tr("repair_export_curves")))
        self.btn_curves.clicked.connect(self.export_curves)
        row.addWidget(self.btn_curves, 1)

        self.btn_stop = PushButton(tr("stop"), self, FluentIcon.PAUSE)
        self._bind_tr(lambda: self.btn_stop.setText(tr("stop")))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_job)
        row.addWidget(self.btn_stop, 1)
        layout.addLayout(row)

        curves_hint = BodyLabel(self)
        curves_hint.setWordWrap(True)
        curves_hint.setStyleSheet("color: #808080; font-size: 12px;")
        self._bind_tr(lambda: curves_hint.setText(tr("repair_curves_hint")))
        layout.addWidget(curves_hint)

    def _build_results_card(self):
        _card, layout = self._card("repair_results")
        self.results_table = QTableWidget(0, 8, self)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setMinimumHeight(180)
        self._retranslate_headers()
        layout.addWidget(self.results_table)

        row = QHBoxLayout()
        self.btn_apply = PushButton(tr("repair_apply"), self, FluentIcon.SYNC)
        self._bind_tr(lambda: self.btn_apply.setText(tr("repair_apply")))
        self.btn_apply.clicked.connect(self.apply_selected)
        row.addWidget(self.btn_apply)
        self.btn_open_dir = PushButton(tr("repair_open_dir"), self, FluentIcon.FOLDER)
        self._bind_tr(lambda: self.btn_open_dir.setText(tr("repair_open_dir")))
        self.btn_open_dir.clicked.connect(self.open_output_dir)
        row.addWidget(self.btn_open_dir)
        row.addStretch(1)
        layout.addLayout(row)

    def _build_status(self):
        row = QHBoxLayout()
        self.progress_ring = IndeterminateProgressRing(self)
        self.progress_ring.setFixedSize(28, 28)
        self.progress_ring.setVisible(False)
        self.status_label = BodyLabel("", self)
        row.addWidget(self.progress_ring)
        row.addWidget(self.status_label, 1)
        self.vBoxLayout.addLayout(row)
        self.log_edit = LogTerminal(self)
        self.vBoxLayout.addWidget(self.log_edit)

    def _retranslate_headers(self):
        headers = [
            tr("repair_col_take"),
            tr("repair_col_engine"),
            tr("repair_col_notes"),
            tr("repair_col_min"),
            tr("repair_col_short"),
            tr("repair_col_repeat"),
            tr("repair_col_flags"),
            tr("repair_col_lyrics"),
        ]
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.resizeColumnsToContents()
        self.results_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)

    # ---------------------------------------------------------------- helpers
    def log_msg(self, message: str):
        self.log_edit.log(message)

    def _show_error(self, title: str, content: str):
        self.log_msg(f"{tr('error_prefix')}: {content}")
        InfoBar.error(
            title=title, content=content, orient=Qt.Horizontal, isClosable=True,
            position=InfoBarPosition.TOP, duration=6000, parent=self,
        )

    def _show_success(self, content: str):
        self.log_msg(content)
        InfoBar.success(
            title=tr("done_title"), content=content, orient=Qt.Horizontal, isClosable=True,
            position=InfoBarPosition.TOP, duration=6000, parent=self,
        )

    def _show_warning(self, content: str):
        InfoBar.warning(
            title=tr("err_cannot_start"), content=content, orient=Qt.Horizontal, isClosable=True,
            position=InfoBarPosition.TOP, duration=4000, parent=self,
        )

    def _require_file(self, edit: LineEdit, message: str) -> pathlib.Path:
        text = edit.text().strip()
        if not text:
            raise ValueError(message)
        path = pathlib.Path(text)
        if not path.is_file():
            raise ValueError(f"{message}: {text}")
        return path

    def _selected_engines(self) -> list[str]:
        engines = []
        if self.cb_tifa.isChecked():
            engines.append("tifa")
        if self.cb_hfa.isChecked():
            engines.append("hfa")
        return engines

    def _selected_language(self) -> str:
        return self.lang_combo.currentData() or "zh"

    def _lyric_output_mode(self) -> str:
        return self.lyric_output_combo.currentData() or DEFAULT_LYRIC_OUTPUT[self._selected_language()]

    def _selected_line(self):
        index = self.line_combo.currentData()
        if index is None:
            return None
        return next((line for line in self._lyric_lines if line.index == index), None)

    # ---------------------------------------------------------------- inputs
    def _browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("pick_files_dialog"), self.audio_edit.text(),
            "Audio (*.wav *.mp3 *.flac *.m4a *.ogg *.opus *.wma *.aif *.aiff)",
        )
        if path:
            self.audio_edit.setText(path)

    def _browse_lyrics(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("repair_lyrics_file"), self.lyrics_edit.text(),
            "Lyrics (*.lrc *.txt);;All Files (*)",
        )
        if path:
            self.lyrics_edit.setText(path)

    def _browse_base(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("repair_base"), self.base_edit.text(), "OpenUtau (*.ustx)",
        )
        if path:
            self.base_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, tr("choose_folder_dialog"), self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _on_audio_changed(self, _text: str):
        audio = self.audio_edit.text().strip()
        if audio and not self._output_touched:
            stem = pathlib.Path(audio).stem
            self.output_edit.setText(str(pathlib.Path(audio).with_name(f"{stem}_gacha")))
        self._update_window_label()

    def _on_lyrics_file_changed(self, _text: str):
        self._reload_lyrics()

    def _on_base_changed(self, _text: str):
        path = self.base_edit.text().strip()
        self._base_header = None
        if path and pathlib.Path(path).is_file():
            try:
                self._base_header = read_ustx_header(path)
            except Exception as error:  # noqa: BLE001 - surfaced in the log
                self.log_msg(tr("repair_base_read_failed", err=error))
        if self._base_header:
            self.tempo_spin.setValue(float(self._base_header["bpm"]))
        self._refresh_lyric_combo(keep_value=True)
        self._update_window_label()

    def _on_manual_window_changed(self, _text: str):
        self._update_window_label()

    def _reload_lyrics(self):
        path = self.lyrics_edit.text().strip()
        self._lyric_lines = []
        if path and pathlib.Path(path).is_file():
            from inference.pipeline.sentence_gacha import parse_lyric_lines

            try:
                text = pathlib.Path(path).read_text(encoding="utf-8")
                self._lyric_lines = parse_lyric_lines(text)
            except Exception as error:  # noqa: BLE001
                self._show_error(
                    tr("repair_lyrics_file"), tr("repair_lyrics_read_failed", err=error)
                )
            else:
                self._apply_detected_language(text)
        self._refresh_lyric_combo()
        self._update_window_label()

    def _apply_detected_language(self, text: str):
        from gui.lyric_language import detect_lyric_language

        detected = detect_lyric_language(text)
        index = self.lang_combo.findData(detected) if detected else -1
        if index >= 0 and index != self.lang_combo.currentIndex():
            self.lang_combo.setCurrentIndex(index)

    def _refresh_lyric_combo(self, keep_value: bool = False):
        current = self.line_combo.currentData() if keep_value else None
        self.line_combo.blockSignals(True)
        self.line_combo.clear()
        for line in self._lyric_lines:
            if line.start is None:
                stamp = "  --        --      "
                bar = "   -.-"
            else:
                end = line.end if line.end is not None else line.start
                stamp = f"{line.start:8.3f}-{end:8.3f}"
                label = bar_beat_label(line.start, self._base_header)
                bar = f"{label:>6}" if label else "   -.-"
            self.line_combo.addItem(f"{line.index + 1:3d}  {bar}  {stamp}  {line.text}", userData=line.index)
        if current is not None:
            index = self.line_combo.findData(current)
            if index >= 0:
                self.line_combo.setCurrentIndex(index)
        elif self.line_combo.count():
            self.line_combo.setCurrentIndex(0)
        self.line_combo.blockSignals(False)
        if not self.line_combo.count():
            self.line_combo.addItem(tr("repair_no_lyrics"), userData=None)
            self.line_combo.setCurrentIndex(0)

    def _refresh_language_options(self, keep_value: bool = False):
        fill_combo(self.lang_combo, TARGET_LANGUAGE_CHOICES, keep_value=keep_value)
        self._refresh_lyric_output_options(keep_value=keep_value)

    def _refresh_lyric_output_options(self, keep_value: bool = False):
        language = self._selected_language()
        choices = LYRIC_OUTPUT_BY_LANGUAGE.get(language, [("word", "opt_word")])
        fill_combo(self.lyric_output_combo, choices, keep_value=keep_value)
        if not keep_value:
            saved = str(
                self.global_settings.settings.value(
                    f"lyric_output_mode_{language}", DEFAULT_LYRIC_OUTPUT.get(language, "word")
                )
                or DEFAULT_LYRIC_OUTPUT.get(language, "word")
            )
            index = self.lyric_output_combo.findData(saved)
            if index >= 0:
                self.lyric_output_combo.setCurrentIndex(index)

    def _on_language_changed(self, *_args):
        if self._selected_language() == "yue" and self.cb_hfa.isChecked():
            self.cb_hfa.setChecked(False)
        self._refresh_lyric_output_options()
        self._update_window_label()

    def _manual_seconds(self, edit: LineEdit):
        text = edit.text().strip()
        if not text:
            return None
        return float(text)

    def _resolve_window(self, allow_whole: bool = False):
        """(line, window, splice_range); window is None for the whole song."""
        from inference.pipeline.sentence_gacha import resolve_window

        line = self._selected_line()
        start = self._manual_seconds(self.start_edit)
        end = self._manual_seconds(self.end_edit)
        if start is None and end is None and line is None:
            if allow_whole:
                return None, None, None
            raise ValueError(tr("repair_need_line"))
        window = resolve_window(
            self._lyric_lines, line=line, start=start, end=end,
            pad_before=float(self.pad_before_spin.value()),
            pad_after=float(self.pad_after_spin.value()),
            total_duration=float("inf"),
        )
        if start is not None and end is not None:
            splice = (min(start, end), max(start, end))
        elif line is not None and line.start is not None and line.end is not None:
            splice = (line.start, line.end)
        else:
            splice = window
        return line, window, splice

    def _update_window_label(self):
        try:
            line, window, splice = self._resolve_window(allow_whole=True)
        except (ValueError, TypeError):
            self.window_label.setText(tr("repair_window_unknown"))
            return
        if window is None:
            self.window_label.setText(tr("repair_window_whole"))
            return
        self.window_label.setText(
            tr(
                "repair_window",
                start=window[0], end=window[1],
                splice_start=splice[0], splice_end=splice[1],
            )
        )

    # ---------------------------------------------------------------- jobs
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def request_stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()

    def wait_for_worker(self, timeout_ms: int) -> bool:
        return self.worker is not None and self.worker.wait(timeout_ms)

    def on_worker_settled(self, callback) -> None:
        if self.worker is not None:
            self.worker.finished.connect(callback)

    def _start_job(self, fn, *, kind: str, needs_asr_session: bool = False) -> bool:
        if self.is_running():
            self._show_warning(tr("repair_busy"))
            return False
        self.log_edit.clear()
        self._job_kind = kind
        self._set_running_ui(True)
        self.worker = FunctionWorker(fn, needs_asr_session=needs_asr_session)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.finished_signal.connect(self._on_job_finished)
        self.worker.error_signal.connect(self._on_job_error)
        self.worker.start()
        return True

    def _set_running_ui(self, running: bool):
        for button in (self.btn_gacha, self.btn_curves, self.btn_apply):
            button.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.progress_ring.setVisible(running)
        if not running:
            self.status_label.setText("")

    def stop_job(self):
        if not self.worker:
            return
        self.worker.stop()
        self.status_label.setText(tr("stopping"))
        InfoBar.warning(
            title=tr("stop_requested_title"), content=tr("stop_requested_body"),
            orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP,
            duration=3000, parent=self,
        )
        if self.worker.isRunning():
            self.log_msg(tr("stop_hint"))

    def _on_job_error(self, message: str):
        self.log_msg(message)
        self._set_running_ui(False)
        InfoBar.error(
            title=tr("fail_title"), content=tr("fail_body"), orient=Qt.Horizontal,
            isClosable=True, position=InfoBarPosition.TOP, duration=6000, parent=self,
        )

    def _on_job_finished(self, _message: str):
        self._set_running_ui(False)
        kind = self._job_kind
        self._job_kind = None
        result = self.worker.result if self.worker is not None else None
        if kind == "gacha":
            results = result or []
            self._populate_results(results)
            self._last_manifest = self._load_manifest()
            self._show_success(tr("repair_gacha_done", n=len(results)))
        elif kind == "curves":
            output = ""
            if isinstance(result, dict):
                output = str(result.get("output", ""))
            self._show_success(tr("repair_curves_done", path=output))

    def _load_manifest(self):
        from inference.pipeline.sentence_gacha import load_manifest

        if self._job_manifest_path is None or not self._job_manifest_path.is_file():
            return None
        try:
            return load_manifest(self._job_manifest_path)
        except Exception as error:  # noqa: BLE001
            self.log_msg(f"manifest: {error}")
            return None

    def _output_dir(self, audio: pathlib.Path) -> pathlib.Path:
        text = self.output_edit.text().strip()
        if text:
            return pathlib.Path(text)
        return audio.with_name(f"{audio.stem}_gacha")

    def _build_gacha_config(self, audio: pathlib.Path, output_dir: pathlib.Path):
        from inference.pipeline.sentence_gacha import GachaConfig

        settings = self.global_settings
        return GachaConfig(
            audio_path=str(audio),
            output_dir=output_dir,
            output_key=audio.stem,
            language=self._selected_language(),
            lyric_output_mode=self._lyric_output_mode(),
            device=normalize_runtime_device(self.device_combo.currentText()),
            game_model_dir=self.model_config.model_path("game_model"),
            hfa_model_dir=self.model_config.model_path("hfa_model"),
            asr_model_path=self.model_config.model_path("asr_model"),
            tifa_model_path=self.model_config.model_path("tifa_model"),
            tempo=float(self.tempo_spin.value()),
            batch_size=settings.batch_size(),
            asr_batch_size=settings.asr_batch_size(),
            slicing_method="smart",
            slice_min_sec=settings.slice_min_sec(),
            slice_max_sec=settings.slice_max_sec(),
            pitch_curve=False,
            rmvpe_model_path=self.model_config.model_path("rmvpe_model"),
            phoneme_asr_model_path=self.model_config.model_path("phoneme_asr_model"),
            pinyin_asr_model_path=self.model_config.model_path("pinyin_asr_model"),
            chinese_asr_engine=self.model_config.chinese_asr_engine(),
            japanese_asr_engine=self.model_config.japanese_asr_engine(),
            japanese_g2p_engine=self.model_config.japanese_g2p_engine(),
            kashi_g2p_model_path=self.model_config.model_path("kashi_g2p_model"),
        )

    def run_gacha(self):
        if not HYBRID_AVAILABLE:
            self._show_warning(tr("err_hybrid"))
            return
        from inference.pipeline.sentence_gacha import (
            build_takes,
            clip_lyrics_for_window,
            run_gacha,
        )

        try:
            audio = self._require_file(self.audio_edit, tr("repair_need_audio"))
            self._require_file(self.base_edit, tr("repair_need_base"))
            if not self._lyric_lines:
                raise ValueError(tr("repair_need_lyrics"))
            engines = self._selected_engines()
            if not engines:
                raise ValueError(tr("repair_need_engine"))
            if self._selected_language() == "yue" and "hfa" in engines:
                raise ValueError(tr("repair_yue_hfa"))
            line, window, splice = self._resolve_window()
            clip_lyrics = clip_lyrics_for_window(
                self._lyric_lines, window[0], window[1],
                fallback=line.text if line is not None else "",
            )
            if line is not None and line.text not in clip_lyrics.splitlines():
                clip_lyrics = line.text
            takes = build_takes(
                engines=engines,
                strategy=self.strategy_combo.currentData() or "random",
                takes=int(self.takes_spin.value()),
                seed=int(self.seed_spin.value()),
                base={
                    "seg_threshold": self.global_settings.seg_threshold(),
                    "seg_radius": self.global_settings.seg_radius(),
                    "est_threshold": self.global_settings.est_threshold(),
                    "t0": self.global_settings.t0(),
                    "nsteps": self.global_settings.nsteps(),
                },
            )
            output_dir = self._output_dir(audio)
            config = self._build_gacha_config(audio, output_dir)
        except (ValueError, TypeError, FileNotFoundError) as error:
            self._show_error(tr("err_cannot_start"), str(error))
            return

        self._job_output_dir = output_dir
        self._job_base = pathlib.Path(self.base_edit.text().strip())
        self._job_manifest_path = output_dir / f"{audio.stem}_manifest.json"
        lyric_text = line.text if line is not None else clip_lyrics
        self.status_label.setText(tr("repair_gacha"))

        def job(session=None, cancel_checker=None):
            config.asr_session = session
            return run_gacha(
                config, takes, window,
                lyric_window_text=lyric_text,
                clip_lines_text=clip_lyrics,
                line=line,
                splice_range=splice,
                cancel_checker=cancel_checker,
            )

        self._start_job(job, kind="gacha", needs_asr_session=True)

    def export_curves(self):
        try:
            audio = self._require_file(self.audio_edit, tr("repair_need_audio"))
            base = self._require_file(self.base_edit, tr("repair_need_base"))
            _line, window, _splice = self._resolve_window(allow_whole=True)
        except (ValueError, TypeError, FileNotFoundError) as error:
            self._show_error(tr("err_cannot_start"), str(error))
            return

        out_path = base.with_name(f"{base.stem}_curves.ustx")
        device = normalize_runtime_device(self.device_combo.currentText())
        rmvpe_model = self.model_config.model_path("rmvpe_model")
        self.status_label.setText(tr("repair_export_curves"))

        def job(session=None, cancel_checker=None):
            return export_reference_curves(
                audio_path=audio,
                base_path=base,
                out_path=out_path,
                window=window,
                device=device,
                rmvpe_model_path=rmvpe_model,
                cancel_checker=cancel_checker,
            )

        self._start_job(job, kind="curves")

    # ---------------------------------------------------------------- results
    def _populate_results(self, results):
        self._take_labels = []
        self.results_table.setRowCount(0)
        for result in results:
            params = result.params
            diag = result.diagnostics
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            values = [
                params.label,
                params.alignment_engine,
                str(diag["note_count"]),
                f"{diag['min_duration']:.3f}",
                str(diag["short_notes"]),
                str(diag["repeated_lyric_notes"]),
                str(diag["flag_score"]),
                diag["lyric_sequence"],
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if 2 <= column <= 6:
                    item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row, column, item)
            self._take_labels.append(params.label)
        self._retranslate_headers()
        if self.results_table.rowCount():
            self.results_table.selectRow(0)

    def apply_selected(self):
        if self.is_running():
            self._show_warning(tr("repair_busy"))
            return
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self._take_labels):
            self._show_warning(tr("repair_select_take"))
            return
        manifest = self._last_manifest
        if not manifest or self._job_base is None or self._job_output_dir is None:
            self._show_warning(tr("repair_no_results"))
            return
        from inference.pipeline.sentence_gacha import splice_take_into_ustx

        label = self._take_labels[row]
        entry = next((item for item in manifest["takes"] if item["take"] == label), None)
        if entry is None:
            self._show_error(tr("fail_title"), tr("repair_no_results"))
            return
        take_path = pathlib.Path(self._job_output_dir) / entry["ustx"]
        window = (
            float(manifest.get("splice_start", manifest["window_start"])),
            float(manifest.get("splice_end", manifest["window_end"])),
        )
        out_path = self._job_base.with_name(f"{self._job_base.stem}_patched.ustx")
        try:
            removed, inserted = splice_take_into_ustx(self._job_base, take_path, out_path, window)
        except Exception as error:  # noqa: BLE001
            self._show_error(tr("fail_title"), str(error))
            return
        self._show_success(
            tr("repair_applied", take=label, removed=removed, inserted=inserted, path=out_path)
        )

    def open_output_dir(self):
        path = self.output_edit.text().strip()
        if not path or not pathlib.Path(path).is_dir():
            self._show_warning(tr("repair_output_dir"))
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
