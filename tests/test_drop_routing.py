"""Single-route drag & drop: one target at a time, and no layout churn.

The flicker came from the audio card reflowing while dragging (a hint row was
shown/hidden), which moved the widget under the cursor and re-triggered
enter/leave. These tests pin both halves: routing picks exactly one handler,
and toggling feedback never changes geometry or the card's own stylesheet.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def clean_settings():
    from gui.global_settings_view import GlobalSettingsInterface

    ui = GlobalSettingsInterface()
    for key in ("enable_batch_mode", "save_dir", "save_dir_follow"):
        ui.settings.remove(key)
    yield
    for key in ("enable_batch_mode", "save_dir", "save_dir_follow"):
        ui.settings.remove(key)


def _make_view(qapp):
    from gui.auto_lyric_view import AutoLyricInterface
    from gui.global_settings_view import GlobalSettingsInterface
    from gui.model_config_view import ModelConfigInterface

    settings_ui = GlobalSettingsInterface()
    model_cfg = ModelConfigInterface(settings_ui.settings, settings_ui.project_root)
    view = AutoLyricInterface(settings_ui, model_cfg)
    view.resize(1200, 900)
    view.show()
    qapp.processEvents()
    return view


def _drop_event(pos, local_file: str) -> QDropEvent:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(local_file)])
    event = QDropEvent(QPointF(pos), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    event._mime = mime  # QDropEvent does not own the QMimeData; keep it alive
    return event


def test_bpm_box_is_not_a_drop_target_in_main_view(qapp):
    view = _make_view(qapp)
    assert view.tempo_spin.acceptDrops() is False
    assert view.tempo_spin.lineEdit().acceptDrops() is False
    view.close()


def test_drop_on_bpm_recognizes_without_touching_the_list(qapp):
    view = _make_view(qapp)
    calls = []
    view.tempo_spin.recognize_bpm = lambda path: calls.append(("bpm", path))
    view.audio_list.add_paths = lambda paths: calls.append(("list", list(paths))) or 1

    pos = view.tempo_spin.mapTo(view, view.tempo_spin.rect().center())
    assert view._drop_target_at(pos) == "bpm"

    event = _drop_event(pos, "D:/music/song.wav")
    view.dropEvent(event)

    assert event.isAccepted()
    assert calls == [("bpm", "D:/music/song.wav")]
    view.close()


def test_drop_outside_bpm_adds_to_the_list(qapp):
    view = _make_view(qapp)
    calls = []
    view.tempo_spin.recognize_bpm = lambda path: calls.append(("bpm", path))
    view.audio_list.add_paths = lambda paths: calls.append(("list", list(paths))) or 1

    pos = view._audio_card.mapTo(view, QPoint(20, 20))
    assert view._drop_target_at(pos) == "list"

    event = _drop_event(pos, "D:/music/song.wav")
    view.dropEvent(event)

    assert event.isAccepted()
    assert calls == [("list", ["D:/music/song.wav"])]
    view.close()


def test_feedback_toggle_never_moves_widgets(qapp):
    """Regression: the highlight used to show/hide a layout row mid-drag."""
    view = _make_view(qapp)

    bpm_origin = view.tempo_spin.mapTo(view, QPoint(0, 0))
    card_height = view._audio_card.height()
    list_height = view.audio_list.height()

    view._update_drop_feedback("list", True)
    qapp.processEvents()
    assert view._card_drop_overlay.isVisible()
    assert view._drop_banner.isVisible()

    view._update_drop_feedback("bpm", True)
    qapp.processEvents()
    assert view.tempo_spin._drop_overlay.isVisible()
    assert not view._card_drop_overlay.isVisible()

    view._update_drop_feedback("list", True)
    view._clear_drop_feedback()
    qapp.processEvents()

    assert view.tempo_spin.mapTo(view, QPoint(0, 0)) == bpm_origin
    assert view._audio_card.height() == card_height
    assert view.audio_list.height() == list_height
    assert not view._card_drop_overlay.isVisible()
    assert not view._drop_banner.isVisible()
    assert not view.tempo_spin._drop_overlay.isVisible()
    view.close()


def test_feedback_does_not_wipe_card_stylesheet(qapp):
    """A transient highlight must not clear CardWidget's themed base style."""
    view = _make_view(qapp)
    base_style = view._audio_card.styleSheet()
    assert base_style, "CardWidget should carry its themed base stylesheet"

    view._update_drop_feedback("list", True)
    qapp.processEvents()
    view._clear_drop_feedback()
    qapp.processEvents()

    assert view._audio_card.styleSheet() == base_style
    view.close()


def test_drag_enter_on_view_shows_feedback(qapp):
    """Qt hands drags to the viewport; the filter must not swallow them."""
    from PySide6.QtGui import QDragEnterEvent

    view = _make_view(qapp)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile("D:/music/song.wav")])

    card_point = view._audio_card.mapTo(view, QPoint(20, 20))
    pos = view.viewport().mapFrom(view, card_point)
    event = QDragEnterEvent(pos, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(view.viewport(), event)

    assert event.isAccepted()
    assert view._card_drop_overlay.isVisible()
    assert view._drop_banner.isVisible()
    view.close()


def test_drop_on_bpm_routes_to_the_box(qapp):
    view = _make_view(qapp)
    calls = []
    view.tempo_spin.recognize_bpm = lambda path: calls.append(path)

    center = view.tempo_spin.mapTo(view, view.tempo_spin.rect().center())
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile("D:/music/bpm.wav")])
    event = QDropEvent(QPointF(center), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    event._mime = mime
    view.dropEvent(event)

    assert event.isAccepted()
    assert calls == ["D:/music/bpm.wav"]
    view.close()


def test_bpm_hint_keeps_a_stable_width(qapp):
    """Swapping the hint text mid-drag must not reflow the options row."""
    view = _make_view(qapp)
    width = view._bpm_hint.width()

    view._update_drop_feedback("bpm", True)
    qapp.processEvents()
    assert view._bpm_hint.width() == width
    assert view._bpm_drop_active is True

    view._clear_drop_feedback()
    qapp.processEvents()
    assert view._bpm_hint.width() == width
    view.close()


def test_non_audio_drop_is_ignored_with_a_note(qapp):
    view = _make_view(qapp)
    logs = []
    view.log_edit.log = logs.append

    pos = view._audio_card.mapTo(view, QPoint(20, 20))
    event = _drop_event(pos, "D:/music/notes.txt")
    view.dropEvent(event)

    assert not event.isAccepted()
    assert len(logs) == 1
    assert "仅支持音频文件" in logs[0] or "Only audio files" in logs[0]
    view.close()
