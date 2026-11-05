"""Window session: clamp against the screen the window is on, and save once.

Two behaviours are pinned here:

* a geometry restored on a secondary monitor must be clamped against that
  monitor, not the primary one (``self.screen()`` is the primary before show);
* a window that no longer fits must shrink, otherwise its title bar / close
  button stay off-screen;
* the session is written only when the close actually happens, so cancelling
  the "pipeline is running" prompt keeps the previous session, and the
  deferred ``hide()`` -> ``close()`` path does not overwrite it twice.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from gui.fluent_main import MainWindow

    w = MainWindow()
    w.app_settings.remove("main_geometry")
    w.app_settings.remove("main_maximized")
    yield w
    w.app_settings.remove("main_geometry")
    w.app_settings.remove("main_maximized")
    w.close()


def test_available_geometry_uses_the_screen_at_the_window(window, monkeypatch):
    class FakeScreen:
        def availableGeometry(self):
            return QRect(100, 50, 400, 300)

    monkeypatch.setattr(QApplication, "screenAt", lambda point: FakeScreen())
    assert window._available_geometry().getRect() == (100, 50, 400, 300)


def test_window_larger_than_the_screen_shrinks_and_stays_inside(window, monkeypatch):
    avail = QRect(0, 0, 1000, 700)
    monkeypatch.setattr(window, "_available_geometry", lambda: avail)
    window.resize(1600, 1000)
    window.move(0, 0)

    window._clamp_to_available_screen()

    assert window.width() <= avail.width()
    assert window.height() <= avail.height()
    geo = window.frameGeometry()
    assert geo.left() >= avail.left()
    assert geo.top() >= avail.top()
    assert geo.right() <= avail.right()
    assert geo.bottom() <= avail.bottom()


def test_shrink_accounts_for_the_window_frame(window, monkeypatch):
    """resize() takes the client size; the frame must still fit the screen."""
    avail = QRect(0, 0, 1000, 700)
    monkeypatch.setattr(window, "_available_geometry", lambda: avail)
    window.resize(1600, 1000)
    window.move(0, 0)

    deco_w, deco_h = 16, 39

    def fake_frame():
        return QRect(
            window.x() - deco_w // 2,
            window.y() - deco_h + 8,
            window.width() + deco_w,
            window.height() + deco_h,
        )

    monkeypatch.setattr(window, "frameGeometry", fake_frame)
    window._clamp_to_available_screen()

    assert window.width() <= avail.width() - deco_w
    assert window.height() <= avail.height() - deco_h


def test_window_off_every_screen_is_recentered(window, monkeypatch):
    avail = QRect(0, 0, 1000, 700)
    monkeypatch.setattr(window, "_available_geometry", lambda: avail)
    window.resize(900, 620)
    window.move(9000, 9000)

    window._clamp_to_available_screen()

    geo = window.frameGeometry()
    assert avail.contains(geo)
    assert abs(geo.center().x() - avail.center().x()) <= 1
    assert abs(geo.center().y() - avail.center().y()) <= 1


def test_cancelled_close_does_not_write_the_session(window, monkeypatch):
    from gui import fluent_main

    # Undo via monkeypatch: a plain attribute would survive into the fixture's
    # teardown, where close() would open a real modal prompt and block.
    monkeypatch.setattr(window.autoLyricInterface, "is_running", lambda: True)
    monkeypatch.setattr(
        fluent_main.QMessageBox, "question",
        staticmethod(lambda *args, **kwargs: fluent_main.QMessageBox.No),
    )
    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is False
    assert window.app_settings.value("main_geometry") is None


def test_close_writes_the_session_exactly_once(window, monkeypatch):
    monkeypatch.setattr(window.autoLyricInterface, "is_running", lambda: False)

    first = QCloseEvent()
    window.closeEvent(first)
    assert first.isAccepted() is True
    assert window.app_settings.value("main_geometry") is not None

    # The deferred close after hide() runs closeEvent a second time; it must
    # not overwrite the geometry captured while the window was still visible.
    window.app_settings.setValue("main_geometry", b"sentinel")
    second = QCloseEvent()
    window.closeEvent(second)
    assert bytes(window.app_settings.value("main_geometry")) == b"sentinel"
