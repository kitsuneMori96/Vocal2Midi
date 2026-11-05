import os
import pathlib

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from PySide6.QtCore import QEvent, Qt

from gui.fluent_compat import (
    ScrollArea,
    PushButton,
    PrimaryPushButton,
    CardWidget,
    IconWidget,
    BodyLabel,
    LineEdit,
    ComboBox,
    DoubleSpinBox,
    TextEdit,
    FluentIcon,
    SubtitleLabel,
    SwitchButton,
    InfoBar,
    InfoBarPosition,
    IndeterminateProgressRing,
    accent_color,
)

from application.config import PipelineConfig, validate_slice_bounds
from gui.audio_file_list import AudioFileList
from gui.bpm_spinbox import BpmSpinBox
from gui.fluent_utils import t0_nstep_to_ts
from gui.fluent_worker import WorkerThread, HYBRID_AVAILABLE
from gui.i18n import tr
from gui.log_terminal import LogTerminal
from gui.option_tables import (
    DEFAULT_LYRIC_OUTPUT,
    EXPORT_FORMAT_CHOICES,
    LYRIC_OUTPUT_BY_LANGUAGE,
    QUANT_MODE_CHOICES,
    QUANT_STEP_CHOICES,
    SLICE_METHOD_CHOICES,
    TARGET_LANGUAGE_CHOICES,
    fill_combo,
)
from gui.settings_utils import default_output_dir
from inference.device_utils import VISIBLE_RUNTIME_DEVICE_CHOICES, normalize_runtime_device


class AutoLyricInterface(ScrollArea):
    def __init__(self, global_settings, model_config, parent=None):
        super().__init__(parent=parent)
        self.global_settings = global_settings
        self.model_config = model_config
        self._tr_bindings: list = []
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)
        self.vBoxLayout.setSpacing(20)
        self.view.setObjectName('view')
        self.setObjectName('autoLyricInterface')

        title = SubtitleLabel(self)
        self._bind_tr(lambda: title.setText(tr("app_title")))
        self.vBoxLayout.addWidget(title)

        audio_card = CardWidget(self)
        audio_layout = QVBoxLayout(audio_card)
        header_layout = QHBoxLayout()

        music_icon = IconWidget(FluentIcon.MUSIC, self)
        music_icon.setFixedSize(16, 16)
        header_layout.addWidget(music_icon)

        title_label = BodyLabel(self)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._bind_tr(lambda: title_label.setText(tr("upload_audio")))
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        btn_add = PushButton(tr("pick_files"), self, FluentIcon.FOLDER)
        self._bind_tr(lambda: btn_add.setText(tr("pick_files")))
        btn_add.clicked.connect(self.add_audio_files)
        btn_clear = PushButton(tr("clear_files"), self, FluentIcon.DELETE)
        self._bind_tr(lambda: btn_clear.setText(tr("clear_files")))
        btn_clear.clicked.connect(self.clear_audio_files)
        header_layout.addWidget(btn_add)
        header_layout.addWidget(btn_clear)
        audio_layout.addLayout(header_layout)

        self.audio_list = AudioFileList(self)
        self.audio_list.setMaximumHeight(40)
        audio_layout.addWidget(self.audio_list, 1)  # stretch so batch mode can grow the list
        self._audio_card_stretch = audio_layout
        self.vBoxLayout.addWidget(audio_card)
        self.setAcceptDrops(True)
        self._is_running = False
        self._last_added_dir: pathlib.Path | None = None
        self.audio_list.filesAdded.connect(lambda n: self.log_msg(tr("files_added", n=n)))
        self.audio_list.pathsAdded.connect(self._on_audio_paths_added)
        self.audio_list.filesChanged.connect(self._on_audio_files_changed)
        self.audio_list.settingsRequested.connect(self._open_file_settings)
        if self.global_settings is not None:
            if hasattr(self.global_settings, "enable_batch_mode"):
                self.audio_list.set_batch_mode(self.global_settings.enable_batch_mode())
            if hasattr(self.global_settings, "batchModeChanged"):
                self.global_settings.batchModeChanged.connect(self._on_batch_mode_setting_changed)
        self._audio_card = audio_card
        self._combo_card = None  # set after cards are built
        self._output_card = None

        # Static hint: shown only while the list is empty. Drag feedback is
        # NOT done by toggling this (that reflows the card mid-drag); it uses
        # overlays with absolute geometry instead.
        self._drop_hint = BodyLabel(self)
        self._bind_tr(lambda: self._refresh_drop_hint())
        self._audio_card_stretch.insertWidget(1, self._drop_hint)

        self._drop_target = "list"
        self._drop_feedback_on = False
        self._card_drop_overlay = QWidget(audio_card)
        self._card_drop_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._card_drop_overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._card_drop_overlay.hide()

        self._drop_banner = BodyLabel(audio_card)
        self._drop_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_banner.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._drop_banner.setStyleSheet(
            "BodyLabel {"
            f"background: {accent_color()};"
            "color: white;"
            "border-radius: 6px;"
            "padding: 3px 10px;"
            "font-size: 12px;"
            "}"
        )
        self._drop_banner.hide()

        self.lyric_card = CardWidget(self)
        lyric_layout = QVBoxLayout(self.lyric_card)
        self.lyric_title = BodyLabel(self)
        self.lyric_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._bind_tr(lambda: self.lyric_title.setText(tr("ref_lyrics")))
        lyric_layout.addWidget(self.lyric_title)

        self.lyrics_edit = TextEdit(self)
        self.lyrics_edit.setPlaceholderText(tr("ref_lyrics_hint"))
        self._bind_tr(lambda: self.lyrics_edit.setPlaceholderText(tr("ref_lyrics_hint")))
        self.lyrics_edit.setMaximumHeight(80)
        self.lyrics_edit.setAcceptDrops(False)  # let drops fall through to the file list
        lyric_layout.addWidget(self.lyrics_edit)
        self.vBoxLayout.addWidget(self.lyric_card)

        combo_card = CardWidget(self)
        combo_layout = QVBoxLayout(combo_card)

        combo_row1 = QHBoxLayout()
        self.slicing_combo = ComboBox(self)
        fill_combo(self.slicing_combo, SLICE_METHOD_CHOICES)
        self._bind_tr(lambda: fill_combo(self.slicing_combo, SLICE_METHOD_CHOICES, keep_value=True))
        self._add_flow_pair(combo_row1, "slicing_method", self.slicing_combo)
        combo_row1.addSpacing(28)
        self.lang_combo = ComboBox(self)
        self._refresh_lang_combo(keep_value=False)
        self._bind_tr(lambda: self._refresh_lang_combo(keep_value=True))
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        self._add_flow_pair(combo_row1, "target_lang", self.lang_combo)
        combo_row1.addSpacing(28)
        self.lyric_output_label = BodyLabel(self)
        self.lyric_output_combo = ComboBox(self)
        self.lyric_output_combo.currentIndexChanged.connect(self.save_lyric_output_preference)
        self._add_flow_pair(combo_row1, "lyric_output_format", self.lyric_output_combo, label=self.lyric_output_label)
        combo_row1.addSpacing(28)
        self.device_combo = ComboBox(self)
        self.device_combo.addItems(list(VISIBLE_RUNTIME_DEVICE_CHOICES))
        self.device_combo.currentTextChanged.connect(self.apply_device_batch_defaults)
        self._add_flow_pair(combo_row1, "device", self.device_combo)
        combo_row1.addStretch(1)
        combo_layout.addLayout(combo_row1)

        combo_row2 = QHBoxLayout()
        self.cb_match_lyrics = SwitchButton("On", self)
        self.cb_match_lyrics.setOffText("Off")
        self.cb_match_lyrics.setChecked(self.global_settings.settings.value("enable_lyrics_match", False, type=bool))
        self.cb_match_lyrics.checkedChanged.connect(self.on_match_lyrics_changed)
        self._add_flow_pair(combo_row2, "match_lyrics", self.cb_match_lyrics)
        combo_row2.addSpacing(28)
        self.cb_output_lyrics = SwitchButton("On", self)
        self.cb_output_lyrics.setOffText("Off")
        self.cb_output_lyrics.setChecked(self.global_settings.settings.value("output_lyrics", True, type=bool))
        self.cb_output_lyrics.checkedChanged.connect(self.on_output_lyrics_changed)
        self._add_flow_pair(combo_row2, "output_lyrics", self.cb_output_lyrics)
        combo_row2.addSpacing(28)
        self.export_format_combo = ComboBox(self)
        fill_combo(self.export_format_combo, EXPORT_FORMAT_CHOICES)
        self.export_format_combo.setCurrentIndex(max(0, self.export_format_combo.findData(self._initial_export_format_value())))
        self.export_format_combo.currentIndexChanged.connect(self.on_export_format_changed)
        self._add_flow_pair(combo_row2, "export_format", self.export_format_combo)
        combo_row2.addSpacing(28)
        self.pitch_curve_label = BodyLabel(self)
        self.cb_pitch_curve = SwitchButton("On", self)
        self.cb_pitch_curve.setOffText("Off")
        self.cb_pitch_curve.setChecked(self.global_settings.settings.value("output_pitch_curve", True, type=bool))
        self.cb_pitch_curve.checkedChanged.connect(lambda v: self.global_settings.settings.setValue("output_pitch_curve", v))
        self._add_flow_pair(combo_row2, "pitch_curve", self.cb_pitch_curve, label=self.pitch_curve_label)
        combo_row2.addStretch(1)
        combo_layout.addLayout(combo_row2)

        self.vBoxLayout.addWidget(combo_card)
        self._combo_card = combo_card

        output_card = CardWidget(self)
        output_layout = QVBoxLayout(output_card)
        output_title = BodyLabel(self)
        output_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._bind_tr(lambda: output_title.setText(tr("output_settings")))
        output_layout.addWidget(output_title)

        opts_layout = QHBoxLayout()
        self.tempo_spin = BpmSpinBox(self, accept_drops=False)
        self.tempo_spin.setRange(10, 300)
        self.tempo_spin.setValue(120)
        self._bind_tr(lambda: self.tempo_spin.setToolTip(tr("tempo_bpm_tooltip")))
        self.tempo_spin.bpmDetectStarted.connect(self._on_bpm_detect_started)
        self.tempo_spin.bpmDetected.connect(self._on_bpm_detected)
        self.tempo_spin.bpmDetectFailed.connect(self._on_bpm_detect_failed)
        self._add_flow_pair(opts_layout, "tempo_bpm", self.tempo_spin)

        # Standing guidance next to the BPM box; swaps text while a file hovers
        # the box. Fixed width so the swap can never reflow the row.
        self._bpm_drop_active = False
        self._bpm_hint = BodyLabel(self)
        self._bpm_hint.setStyleSheet("font-size: 12px; color: #808080;")
        self._bind_tr(self._refresh_bpm_hint)
        opts_layout.addWidget(self._bpm_hint)
        opts_layout.addSpacing(28)
        self.quantize_combo = ComboBox(self)
        fill_combo(self.quantize_combo, QUANT_STEP_CHOICES)
        self._bind_tr(lambda: fill_combo(self.quantize_combo, QUANT_STEP_CHOICES, keep_value=True))
        initial_step = self._initial_quantization_step()
        self.quantize_combo.setCurrentIndex(max(0, self.quantize_combo.findData(initial_step)))
        self.quantize_combo.currentIndexChanged.connect(self.on_quantization_step_changed)
        self._add_flow_pair(opts_layout, "quant_step", self.quantize_combo)
        opts_layout.addSpacing(28)
        self.quantize_mode_combo = ComboBox(self)
        fill_combo(self.quantize_mode_combo, QUANT_MODE_CHOICES)
        self._bind_tr(lambda: fill_combo(self.quantize_mode_combo, QUANT_MODE_CHOICES, keep_value=True))
        initial_mode = self._initial_quantization_mode()
        self.quantize_mode_combo.setCurrentIndex(max(0, self.quantize_mode_combo.findData(initial_mode)))
        self.quantize_mode_combo.currentIndexChanged.connect(self.on_quantization_mode_changed)
        self._add_flow_pair(opts_layout, "quant_mode", self.quantize_mode_combo)
        opts_layout.addStretch(1)
        output_layout.addLayout(opts_layout)

        save_layout = QHBoxLayout()
        save_dir_label = BodyLabel(self)
        self._bind_tr(lambda: save_dir_label.setText(tr("save_dir")))
        save_layout.addWidget(save_dir_label)
        self.save_dir_edit = LineEdit(self)
        self._suppress_save_dir_follow = True
        self.save_dir_edit.setText(
            self.global_settings.settings.value("save_dir", str(default_output_dir(self.global_settings.project_root)))
        )
        self._suppress_save_dir_follow = False
        self.save_dir_edit.textChanged.connect(self._on_save_dir_text_changed)
        save_layout.addWidget(self.save_dir_edit, 1)
        self.follow_dir_label = BodyLabel(self)
        self._bind_tr(lambda: self.follow_dir_label.setText(tr("follow_audio_dir")))
        save_layout.addWidget(self.follow_dir_label)
        self.follow_dir_switch = SwitchButton("On", self)
        self.follow_dir_switch.setOffText("Off")
        self.follow_dir_switch.setChecked(
            self.global_settings.settings.value("save_dir_follow", True, type=bool)
        )
        self.follow_dir_switch.checkedChanged.connect(self._on_follow_dir_toggled)
        save_layout.addWidget(self.follow_dir_switch)
        btn_browse_save = PushButton(tr("browse"), self, FluentIcon.FOLDER)
        self._bind_tr(lambda: btn_browse_save.setText(tr("browse")))
        btn_browse_save.clicked.connect(lambda: self.browse_dir(self.save_dir_edit))
        save_layout.addWidget(btn_browse_save)
        output_layout.addLayout(save_layout)
        self.vBoxLayout.addWidget(output_card)
        self._output_card = output_card

        # selection changes drive batch mode; the retranslate refresh runs
        # after __init__ finishes (batch refresh is invoked there too)
        self.audio_list.itemSelectionChanged.connect(self._update_batch_mode)

        action_layout = QHBoxLayout()
        self.btn_run = PrimaryPushButton(tr("run"), self, FluentIcon.PLAY)
        self._bind_tr(lambda: self.btn_run.setText(tr("run")))
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_stop = PushButton(tr("stop"), self, FluentIcon.PAUSE)
        self._bind_tr(lambda: self.btn_stop.setText(tr("stop")))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_pipeline)
        self.progress_ring = IndeterminateProgressRing(self)
        self.progress_ring.setFixedSize(28, 28)
        self.progress_ring.setVisible(False)
        self.run_status_label = BodyLabel("", self)
        action_layout.addWidget(self.btn_run, 1)
        action_layout.addWidget(self.btn_stop, 1)
        action_layout.addWidget(self.progress_ring)
        action_layout.addWidget(self.run_status_label)
        self.vBoxLayout.addLayout(action_layout)

        self.log_edit = LogTerminal(self)
        self.vBoxLayout.addWidget(self.log_edit)

        self.vBoxLayout.addStretch(1)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()
        # The scroll area's viewport accepts drops and is the widget Qt hands
        # drags to, so the page-level handlers would never run. Forward them
        # here; positions are translated from viewport coordinates.
        self.viewport().installEventFilter(self)

        self.worker = None
        self._last_device = None
        self.update_lyrics_visibility()
        self.update_lyric_output_options()
        self.model_config.chinese_asr_engine_changed.connect(
            lambda _value: self.update_lyric_output_options()
        )
        # Switching the alignment engine enables/disables Cantonese ('yue') which HFA does not support.
        self.model_config.alignment_engine_changed.connect(
            lambda _value: self.update_language_availability()
        )
        self._last_device = self.device_combo.currentText()  # baseline; don't wipe saved batches on startup
        self.on_export_format_changed()
        self._update_batch_mode()

    def _bind_tr(self, fn):
        self._tr_bindings.append(fn)
        fn()

    def retranslate_ui(self):
        for fn in self._tr_bindings:
            fn()

    @staticmethod
    def _fill_combo(combo, choices: list[tuple], keep_value: bool = False):
        fill_combo(combo, choices, keep_value=keep_value)

    def _add_flow_pair(self, row, label_key, widget, label=None):
        label = label or BodyLabel(self)
        self._bind_tr(lambda l=label, k=label_key: l.setText(tr(k)))
        row.addWidget(label)
        row.addWidget(widget)

    @staticmethod
    def _audio_urls(event) -> list:
        return [
            url.toLocalFile() for url in event.mimeData().urls()
            if url.isLocalFile() and pathlib.Path(url.toLocalFile()).suffix.lower() in AudioFileList.AUDIO_EXTENSIONS
        ]

    def _refresh_drop_hint(self):
        """Static empty-state hint only; never toggled while dragging."""
        self._drop_hint.setText(tr("drop_audio_hint"))
        self._drop_hint.setVisible(self.audio_list.count() == 0)

    def _refresh_bpm_hint(self):
        text = tr("bpm_drop_now") if self._bpm_drop_active else tr("tempo_bpm_drop_hint")
        if self._bpm_hint.text() != text:
            self._bpm_hint.setText(text)
        self._bpm_hint.ensurePolished()
        fm = self._bpm_hint.fontMetrics()
        width = max(
            fm.horizontalAdvance(tr("tempo_bpm_drop_hint")),
            fm.horizontalAdvance(tr("bpm_drop_now")),
        ) + 4
        if self._bpm_hint.minimumWidth() != width:
            self._bpm_hint.setFixedWidth(width)

    def _drop_target_at(self, pos) -> str:
        """Which drop handler owns the cursor: the BPM box or the audio list."""
        spin = getattr(self, "tempo_spin", None)
        if spin is not None and spin.isVisible() and spin.rect().contains(spin.mapFrom(self, pos)):
            return "bpm"
        return "list"

    def _sync_card_overlays(self):
        rect = self._audio_card.rect().adjusted(1, 1, -1, -1)
        self._card_drop_overlay.setGeometry(rect)
        self._drop_banner.adjustSize()
        size = self._drop_banner.size()
        self._drop_banner.move(
            rect.center().x() - size.width() // 2,
            rect.center().y() - size.height() // 2,
        )

    def _update_drop_feedback(self, target: str, has_audio: bool):
        """Refresh highlight+banner. No layout item is added or hidden here."""
        if not has_audio:
            self._clear_drop_feedback()
            return
        if self._drop_feedback_on and target == self._drop_target:
            return
        self._drop_target = target
        self._drop_feedback_on = True
        is_bpm = target == "bpm"
        self.tempo_spin.setDropHighlight(is_bpm)
        if self._bpm_drop_active != is_bpm:
            self._bpm_drop_active = is_bpm
            self._refresh_bpm_hint()
        if is_bpm:
            self._card_drop_overlay.hide()
            self._drop_banner.hide()
            return
        self._card_drop_overlay.setStyleSheet(
            "QWidget {"
            "background: transparent;"
            f"border: 2px dashed {accent_color()};"
            "border-radius: 6px;"
            "}"
        )
        self._drop_banner.setText(tr("drop_now_hint"))
        self._sync_card_overlays()
        self._card_drop_overlay.show()
        self._card_drop_overlay.raise_()
        self._drop_banner.show()
        self._drop_banner.raise_()

    def _clear_drop_feedback(self):
        self._drop_feedback_on = False
        self.tempo_spin.setDropHighlight(False)
        self._card_drop_overlay.hide()
        self._drop_banner.hide()
        if self._bpm_drop_active:
            self._bpm_drop_active = False
            self._refresh_bpm_hint()

    def dragEnterEvent(self, event):
        self._handle_drag(event, event.position().toPoint())

    def dragLeaveEvent(self, event):
        self._clear_drop_feedback()

    def dragMoveEvent(self, event):
        self._handle_drag(event, event.position().toPoint())

    def dropEvent(self, event):
        self._handle_drop(event, event.position().toPoint())

    def eventFilter(self, watched, event):
        """Route viewport drags into the single handler set.

        Qt delivers drags to the scroll area's viewport (it accepts drops on
        its own), so without this the page-level handlers never run. Drop
        events are forwarded too; positions arrive in viewport coordinates.
        """
        if watched is self.viewport():
            kind = event.type()
            if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                self._handle_drag(event, self.viewport().mapTo(self, event.position().toPoint()))
                return True
            if kind == QEvent.Type.DragLeave:
                self._clear_drop_feedback()
                return True
            if kind == QEvent.Type.Drop:
                self._handle_drop(event, self.viewport().mapTo(self, event.position().toPoint()))
                return True
        return super().eventFilter(watched, event)

    def _handle_drag(self, event, pos):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        event.acceptProposedAction()
        self._update_drop_feedback(self._drop_target_at(pos), bool(self._audio_urls(event)))

    def _handle_drop(self, event, pos):
        target = self._drop_target_at(pos)
        paths = self._audio_urls(event)
        self._clear_drop_feedback()
        if not paths:
            if event.mimeData().hasUrls():
                self.log_msg(tr("drop_unsupported"))
            event.ignore()
            return
        event.acceptProposedAction()
        if target == "bpm":
            self.tempo_spin.recognize_bpm(paths[0])
            return
        self.audio_list.add_paths(paths)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._drop_feedback_on and self._card_drop_overlay.isVisible():
            self._sync_card_overlays()

    def _on_audio_files_changed(self):
        self._update_batch_mode()

    def _on_audio_paths_added(self, paths) -> None:
        """Remember the folder of the newest file and follow it.

        Only additions move the save dir here: removing a row leaves the
        folder alone.  Clearing the list resets it to the default output dir
        (see ``clear_audio_files``), because the folder being followed is gone.
        """
        if not paths:
            return
        self._last_added_dir = pathlib.Path(str(paths[-1])).expanduser().parent
        self._follow_audio_dir()

    def _on_save_dir_text_changed(self, text: str):
        self.global_settings.settings.setValue("save_dir", text)
        if self._suppress_save_dir_follow:
            return
        switch = getattr(self, "follow_dir_switch", None)
        if switch is not None and switch.isChecked():
            # A manual edit is an explicit override; stop fighting the user.
            self._suppress_save_dir_follow = True
            try:
                switch.setChecked(False)
            finally:
                self._suppress_save_dir_follow = False

    def _on_follow_dir_toggled(self, checked: bool):
        self.global_settings.settings.setValue("save_dir_follow", checked)
        if checked:
            self._follow_audio_dir()

    def _follow_target_dir(self) -> pathlib.Path | None:
        """Folder to follow: the most recent addition, else the last row."""
        if self._last_added_dir is not None:
            return self._last_added_dir
        paths = self.audio_list.all_paths()
        if not paths:
            return None
        # Turning the switch back on after a restart has no "last added" yet;
        # the newest row is the closest thing to it.
        return pathlib.Path(str(paths[-1])).expanduser().parent

    def _follow_audio_dir(self) -> None:
        """Point the save dir at the most recently added audio folder."""
        switch = getattr(self, "follow_dir_switch", None)
        if switch is None or not switch.isChecked():
            return
        target = self._follow_target_dir()
        if target is None:
            return
        text = str(target)
        if text and text != self.save_dir_edit.text().strip():
            self._suppress_save_dir_follow = True
            try:
                self.save_dir_edit.setText(text)
            finally:
                self._suppress_save_dir_follow = False

    def log_msg(self, msg):
        self.log_edit.log(msg)

    def _show_error(self, title: str, content: str):
        self.log_msg(f"{tr('error_prefix')}: {content}")
        InfoBar.error(
            title=title, content=content, orient=Qt.Horizontal, isClosable=True,
            position=InfoBarPosition.TOP, duration=5000, parent=self,
        )

    def _on_bpm_detect_started(self, path: str):
        self.log_msg(tr("bpm_detecting", f=pathlib.Path(path).name))

    def _on_bpm_detected(self, bpm: float, path: str):
        msg = tr("bpm_detected", f=pathlib.Path(path).name, bpm=f"{bpm:.2f}")
        self.log_msg(msg)
        InfoBar.success(
            title=tr("bpm_detect_title"),
            content=msg,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3500,
            parent=self,
        )

    def _on_bpm_detect_failed(self, error: str, path: str):
        msg = tr("bpm_failed", f=pathlib.Path(path).name, err=error)
        self._show_error(tr("bpm_detect_title"), msg)

    def _set_running_ui(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.progress_ring.setVisible(running)
        if not running:
            self.run_status_label.setText("")

    def on_progress(self, current: int, total: int, filename: str):
        self.run_status_label.setText(tr("processing", i=current, n=total, f=filename))

    def apply_device_batch_defaults(self, device: str):
        # Apply conservative batch defaults only when the device actually
        # changes; never wipe a value the user persisted in global settings.
        if device == self._last_device:
            return
        self._last_device = device
        self.global_settings.apply_batch_defaults(batch=1, asr_batch=2)

    def update_lyrics_visibility(self):
        batch = hasattr(self, "audio_list") and self.audio_list.count() > 1
        enabled = self.cb_match_lyrics.isChecked() and not batch
        self.lyric_card.setVisible(enabled)
        self.lyric_title.setVisible(enabled)
        self.lyrics_edit.setVisible(enabled)

    def _lyric_output_setting_key(self, language: str):
        return f"lyric_output_mode_{language}"

    def _selected_language(self) -> str:
        return self.lang_combo.currentData() or "zh"

    def _chinese_asr_engine(self) -> str:
        return self.model_config.chinese_asr_engine()

    def _japanese_asr_engine(self) -> str:
        return self.model_config.japanese_asr_engine()

    def _japanese_g2p_engine(self) -> str:
        return self.model_config.japanese_g2p_engine()

    def _alignment_engine(self) -> str:
        return self.model_config.alignment_engine()

    def _refresh_lang_combo(self, keep_value: bool = False):
        fill_combo(self.lang_combo, TARGET_LANGUAGE_CHOICES, keep_value=keep_value)
        self.update_language_availability()

    def update_language_availability(self):
        """Update language combo options based on alignment engine compatibility.

        HubertFA (HFA) does not support Cantonese ('yue'). When HFA is active,
        'yue' is disabled in the dropdown. If 'yue' was currently selected,
        it automatically falls back to 'zh'.
        """
        is_hfa = self._alignment_engine() == "hfa"
        yue_idx = self.lang_combo.findData("yue")
        if yue_idx >= 0:
            self.lang_combo.setItemEnabled(yue_idx, not is_hfa)
            label = tr("lang_name_yue_disabled") if is_hfa else tr("lang_name_yue")
            self.lang_combo.setItemText(yue_idx, label)
        if is_hfa and self._selected_language() == "yue":
            zh_idx = self.lang_combo.findData("zh")
            self.lang_combo.setCurrentIndex(max(0, zh_idx))
            self.update_lyric_output_options()

    def _on_language_changed(self, *_args):
        if self._selected_language() == "yue" and self._alignment_engine() == "hfa":
            zh_idx = self.lang_combo.findData("zh")
            if zh_idx >= 0 and self.lang_combo.currentIndex() != zh_idx:
                self.lang_combo.setCurrentIndex(zh_idx)
                return
        self.update_lyric_output_options()

    def _pinyin_output_locked(self) -> bool:
        """PinyinASR can only emit pinyin, so hanzi output is unavailable."""
        return self._selected_language() == "zh" and self._chinese_asr_engine() == "pinyin"

    def save_lyric_output_preference(self, *_args):
        language = self._selected_language()
        value = self.lyric_output_combo.currentData()
        if value:
            self.global_settings.settings.setValue(self._lyric_output_setting_key(language), value)

    def update_lyric_output_enabled_state(self):
        enabled = self.cb_output_lyrics.isChecked()
        locked = self._pinyin_output_locked()
        self.lyric_output_label.setEnabled(enabled)
        self.lyric_output_combo.setEnabled(enabled and not locked)
        if locked:
            hint = tr("lyric_output_locked_hint")
            self.lyric_output_combo.setToolTip(hint)
            self.lyric_output_label.setToolTip(hint)
        else:
            self.lyric_output_combo.setToolTip("")
            self.lyric_output_label.setToolTip("")

    def on_output_lyrics_changed(self, enabled: bool):
        self.global_settings.settings.setValue("output_lyrics", enabled)
        self.update_lyric_output_enabled_state()

    def on_match_lyrics_changed(self, enabled: bool):
        self.global_settings.settings.setValue("enable_lyrics_match", enabled)
        self.update_lyrics_visibility()

    def get_export_format(self) -> str:
        return self.export_format_combo.currentData() or "mid"

    def _initial_export_format_value(self) -> str:
        """Load the selected format, migrating the former export switches."""
        saved_format = str(self.global_settings.settings.value("export_format", "")).strip().lower()
        return saved_format if saved_format in {"mid", "ustx", "vsqx"} else "mid"

    def on_export_format_changed(self, *_args):
        self.global_settings.settings.setValue("export_format", self.get_export_format())
        self._update_pitch_curve_enabled()

    def _update_pitch_curve_enabled(self):
        enabled = self.get_export_format() in {"ustx", "vsqx"}
        self.pitch_curve_label.setEnabled(enabled)
        self.cb_pitch_curve.setEnabled(enabled)

    def _initial_quantization_mode(self) -> str:
        if self.global_settings is None:
            return "smart"
        saved = str(self.global_settings.settings.value("quantization_mode", "smart")).strip().lower()
        return saved if saved in {"smart", "simple"} else "smart"

    def on_quantization_mode_changed(self, *_args):
        if self.global_settings is not None:
            mode = self.quantize_mode_combo.currentData() or "smart"
            self.global_settings.settings.setValue("quantization_mode", mode)

    def _initial_quantization_step(self) -> int:
        if self.global_settings is None:
            return 0
        try:
            saved = int(self.global_settings.settings.value("quantization_step", 0))
        except (ValueError, TypeError):
            saved = 0
        valid_steps = {choice[0] for choice in QUANT_STEP_CHOICES}
        return saved if saved in valid_steps else 0

    def on_quantization_step_changed(self, *_args):
        if self.global_settings is not None:
            step = self.quantize_combo.currentData()
            if step is None:
                step = 0
            self.global_settings.settings.setValue("quantization_step", int(step))

    def update_lyric_output_options(self, *_args):
        language = self._selected_language()
        locked = self._pinyin_output_locked()
        choices = [
            (value, key)
            for value, key in LYRIC_OUTPUT_BY_LANGUAGE.get(language, LYRIC_OUTPUT_BY_LANGUAGE["zh"])
            if not locked or value == "pinyin"
        ]
        saved_value = str(self.global_settings.settings.value(
            self._lyric_output_setting_key(language),
            DEFAULT_LYRIC_OUTPUT.get(language, "hanzi"),
        ))

        self._fill_combo(self.lyric_output_combo, choices)
        values = [value for value, _ in choices]
        if saved_value not in values:
            saved_value = "pinyin" if locked else DEFAULT_LYRIC_OUTPUT.get(language, "hanzi")
        index = self.lyric_output_combo.findData(saved_value)
        self.lyric_output_combo.setCurrentIndex(max(0, index))

        self.save_lyric_output_preference()
        self.update_lyric_output_enabled_state()

    def get_lyric_output_mode(self):
        return self.lyric_output_combo.currentData() or DEFAULT_LYRIC_OUTPUT.get(self._selected_language(), "hanzi")

    def browse_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, tr("choose_folder_dialog"), line_edit.text())
        if dir_path:
            line_edit.setText(dir_path)

    def _on_batch_mode_setting_changed(self, enabled: bool):
        if self._is_running:
            self.audio_list.batch_mode = enabled
            return
        self.audio_list.set_batch_mode(enabled)

    def add_audio_files(self):
        batch_enabled = True
        if self.global_settings is not None and hasattr(self.global_settings, "enable_batch_mode"):
            batch_enabled = self.global_settings.enable_batch_mode()

        filter_str = "Audio Files (*.wav *.m4a *.flac *.mp3 *.ogg *.opus *.wma *.webm *.aif *.aiff)"
        start_dir = self._dialog_start_dir()
        if batch_enabled:
            files, _ = QFileDialog.getOpenFileNames(
                self, tr("pick_files_dialog"), start_dir, filter_str
            )
        else:
            file, _ = QFileDialog.getOpenFileName(
                self, tr("pick_files_dialog"), start_dir, filter_str
            )
            files = [file] if file else []
        if files:
            self.audio_list.add_paths(files)

    def _dialog_start_dir(self) -> str:
        """Open the picker where the user last was, not at an empty path.

        Follow mode means the audio folder is the working location; with it
        off the save dir was chosen by hand, so that is the better start.
        """
        switch = getattr(self, "follow_dir_switch", None)
        following = switch is not None and switch.isChecked()
        candidates = []
        if following:
            candidates.append(self._follow_target_dir())
        current = self.save_dir_edit.text().strip()
        if current:
            candidates.append(pathlib.Path(current).expanduser())
        if not following:
            candidates.append(self._follow_target_dir())
        for target in candidates:
            if target is not None and target.is_dir():
                return str(target)
        return ""

    def clear_audio_files(self):
        """Clear the list; with follow mode on the save dir returns to default."""
        had_files = self.audio_list.count() > 0
        self.audio_list.clear_all()
        self._last_added_dir = None
        if not had_files:
            return
        switch = getattr(self, "follow_dir_switch", None)
        if switch is None or not switch.isChecked():
            return
        target = str(default_output_dir(getattr(self.global_settings, "project_root", None)))
        if target != self.save_dir_edit.text().strip():
            self._suppress_save_dir_follow = True
            try:
                self.save_dir_edit.setText(target)
            finally:
                self._suppress_save_dir_follow = False

    def _update_batch_mode(self):
        """Single file keeps the panels; more than one file in the list enters
        batch mode: panels hide and the list grows to fill the freed space."""
        file_count = self.audio_list.count()
        batch = file_count > 1
        self._combo_card.setVisible(not batch)
        self._output_card.setVisible(not batch)
        self.update_lyrics_visibility()
        self.audio_list.setMaximumHeight(16777215 if batch else 40)  # unclamped in batch mode
        self.audio_list.setMinimumHeight(160 if batch else 40)
        if batch:
            self.run_status_label.setText(tr("batch_mode", n=file_count))
        elif not self._is_running:
            self.run_status_label.setText("")
        self._refresh_drop_hint()

    def _current_base_values(self) -> dict:
        """Snapshot every parameter the main UI currently controls.

        All entries are backend values; the per-file dialog round-trips them
        without any display-text conversion.
        """
        batch = hasattr(self, "audio_list") and self.audio_list.count() > 1
        return {
            "slicing_method": self.slicing_combo.currentData(),
            "language": self.lang_combo.currentData() or "zh",
            "lyric_output": self.lyric_output_combo.currentData() or "hanzi",
            "device": self.device_combo.currentText(),
            "match_lyrics": False if batch else self.cb_match_lyrics.isChecked(),
            "original_lyrics": "" if batch else self.lyrics_edit.toPlainText().strip(),
            "export_format": self.get_export_format(),
            "output_lyrics": self.cb_output_lyrics.isChecked(),
            "pitch_curve": self.cb_pitch_curve.isChecked(),
            "tempo": self.tempo_spin.value(),
            "quantization_step": self.quantize_combo.currentData(),
            "quantization_mode": self.quantize_mode_combo.currentData(),
            "batch_size": self.global_settings.batch_size(),
            "asr_batch_size": self.global_settings.asr_batch_size(),
            "output_dir": self.save_dir_edit.text(),
            "chinese_asr_engine": self._chinese_asr_engine(),
            "japanese_asr_engine": self._japanese_asr_engine(),
            "alignment_engine": self.model_config.alignment_engine(),
            "japanese_g2p_engine": self._japanese_g2p_engine(),
            "devices": VISIBLE_RUNTIME_DEVICE_CHOICES,
            "slice_min_sec": self.global_settings.slice_min_sec(),
            "slice_max_sec": self.global_settings.slice_max_sec(),
            "output_formats_extra": self.global_settings.extra_output_formats(),
            "pitch_format": self.global_settings.pitch_format(),
            "round_pitch": self.global_settings.round_pitch(),
            "seg_threshold": self.global_settings.seg_threshold(),
            "seg_radius": self.global_settings.seg_radius(),
            "est_threshold": self.global_settings.est_threshold(),
        }

    def _open_file_settings(self, filename: str):
        from gui.file_settings_dialog import FileSettingsDialog

        base = dict(self._current_base_values())
        saved = self.audio_list.file_settings.get(filename)
        if saved:
            base.update({k: v for k, v in saved.items() if k in base})
        dialog = FileSettingsDialog(filename, base, self)
        if dialog.exec():
            self.audio_list.file_settings[filename] = dialog.values()
            self.log_msg(f"{filename}: {tr('file_settings')} {tr('apply')}")

    def _build_file_config(self, filename: str, base: dict, ts_list: list, overrides: dict | None) -> PipelineConfig:
        """Build a PipelineConfig for one file, applying its saved overrides."""
        values = dict(base)
        if overrides:
            values.update(overrides)
        export_format = values["export_format"]
        output_formats = [export_format, *values["output_formats_extra"]]
        save_dir = values["output_dir"]
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
                self.log_msg(tr("info_dir_created", dir=save_dir))
            except Exception as e:
                self._show_error(tr("err_dir_create"), str(e))
                raise ValueError(f"cannot create save dir: {save_dir}") from e
        elif not os.path.isdir(save_dir):
            self._show_error(tr("err_cannot_start"), tr("err_dir_not_dir"))
            raise ValueError(f"save path is not a directory: {save_dir}")

        return PipelineConfig(
            audio_path="",  # set per-file in worker
            output_filename="",  # set per-file in worker
            output_dir=pathlib.Path(save_dir),
            game_model_dir=self.model_config.model_path("game_model"),
            hfa_model_dir=self.model_config.model_path("hfa_model"),
            asr_model_path=self.model_config.model_path("asr_model"),
            device=normalize_runtime_device(values["device"]),
            language=values["language"],
            ts=ts_list,
            lyric_output_mode=values["lyric_output"],
            original_lyrics=values["original_lyrics"] if values["match_lyrics"] else "",
            output_formats=output_formats,
            output_lyrics=values["output_lyrics"],
            output_pitch_curve=values["pitch_curve"] if export_format in {"ustx", "vsqx"} else False,
            slicing_method=values["slicing_method"],
            slice_min_sec=values["slice_min_sec"],
            slice_max_sec=values["slice_max_sec"],
            tempo=float(values["tempo"]),
            quantization_step=values["quantization_step"],
            quantization_mode=values["quantization_mode"],
            pitch_format=values["pitch_format"],
            round_pitch=values["round_pitch"],
            seg_threshold=values["seg_threshold"],
            seg_radius=values["seg_radius"],
            est_threshold=values["est_threshold"],
            batch_size=int(values["batch_size"]),
            asr_batch_size=int(values["asr_batch_size"]),
            rmvpe_model_path=self.model_config.model_path("rmvpe_model"),
            phoneme_asr_model_path=self.model_config.model_path("phoneme_asr_model"),
            pinyin_asr_model_path=self.model_config.model_path("pinyin_asr_model"),
            chinese_asr_engine=values.get("chinese_asr_engine", "qwen"),
            japanese_asr_engine=values.get("japanese_asr_engine", "romaji"),
            alignment_engine=values.get("alignment_engine", "hfa"),
            japanese_g2p_engine=values.get("japanese_g2p_engine", "kashi-g2p-onnx"),
            kashi_g2p_model_path=self.model_config.model_path("kashi_g2p_model"),
            tifa_model_path=self.model_config.model_path("tifa_model"),
        )

    def run_pipeline(self):
        if not HYBRID_AVAILABLE:
            self.log_msg(tr("err_hybrid"))
            return

        # Batch mode runs the current selection; otherwise every file in the list.
        audio_files = self.audio_list.selected_paths() or self.audio_list.all_paths()
        if not audio_files:
            self._show_error(tr("err_cannot_start"), tr("err_no_audio"))
            return

        base = self._current_base_values()
        ts_list = t0_nstep_to_ts(
            self.global_settings.t0(),
            self.global_settings.nsteps(),
        )
        try:
            validate_slice_bounds(base["slice_min_sec"], base["slice_max_sec"])
        except ValueError as exc:
            self.log_msg(f"Error: invalid slice duration settings: {exc}")
            return

        tasks: list[tuple[PipelineConfig, str]] = []
        try:
            for filename in audio_files:
                config = self._build_file_config(
                    filename, base, ts_list, self.audio_list.file_settings.get(filename)
                )
                tasks.append((config, filename))
        except ValueError:
            return  # error already reported

        self.log_edit.clear()
        self._set_running_ui(True)
        self.run_status_label.setText(tr("preparing", n=len(tasks)))
        self.worker = WorkerThread(tasks)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def stop_pipeline(self):
        if self.worker:
            self.worker.stop()
            self.run_status_label.setText(tr("stopping"))
            InfoBar.warning(
                title=tr("stop_requested_title"), content=tr("stop_requested_body"), orient=Qt.Horizontal,
                isClosable=True, position=InfoBarPosition.TOP, duration=3000, parent=self,
            )
            if self.worker.isRunning():
                self.log_msg(tr("stop_hint"))

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def request_stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()

    def wait_for_worker(self, timeout_ms: int) -> bool:
        """True when the worker thread finished within the timeout."""
        return self.worker is not None and self.worker.wait(timeout_ms)

    def on_worker_settled(self, callback) -> None:
        """Invoke callback once the current worker thread has finished."""
        if self.worker is not None:
            self.worker.finished.connect(callback)

    def on_finished(self, msg):
        self.log_msg(msg)
        self._set_running_ui(False)
        InfoBar.success(
            title=tr("done_title"), content=msg, orient=Qt.Horizontal, isClosable=True,
            position=InfoBarPosition.TOP, duration=6000, parent=self,
        )
        try:
            save_dir_path = self.save_dir_edit.text()
            if os.path.exists(save_dir_path):
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(save_dir_path))
        except Exception as e:
            self.log_msg(f"Cannot open output folder: {e}")

    def on_error(self, msg):
        self.log_msg(msg)
        self._set_running_ui(False)
        InfoBar.error(
            title=tr("fail_title"), content=tr("fail_body"), orient=Qt.Horizontal,
            isClosable=True, position=InfoBarPosition.TOP, duration=6000, parent=self,
        )
