import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from gui.fluent_compat import FluentWindow, NavigationItemPosition, setTheme, Theme, FluentIcon

from gui import __version__
from gui.global_settings_view import GlobalSettingsInterface
from gui.i18n import tr, set_language
from gui.model_config_view import ModelConfigInterface
from gui.auto_lyric_view import AutoLyricInterface
from gui.sentence_repair_view import SentenceRepairInterface
from gui.settings_utils import create_app_settings


def _load_app_icon() -> QIcon:
    icon_path = Path(__file__).resolve().parents[1] / "icon.png"
    if not icon_path.is_file():
        return QIcon()
    return QIcon(str(icon_path))


class _FluentMainWindow(FluentWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"Vocal2Midi v{__version__}")
        app_icon = _load_app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.app_settings = create_app_settings()
        self._restore_window_state()

        # Apply the saved language before building any widget so every page
        # constructs in the right language.
        set_language(str(self.app_settings.value("language", "zh")))

        self.globalSettingsInterface = GlobalSettingsInterface(self)
        setTheme(self._load_theme())
        self._enable_mica()

        self.modelConfigInterface = ModelConfigInterface(
            self.globalSettingsInterface.settings,
            self.globalSettingsInterface.project_root,
            self,
        )
        self.autoLyricInterface = AutoLyricInterface(
            self.globalSettingsInterface,
            self.modelConfigInterface,
            self,
        )
        self.sentenceRepairInterface = SentenceRepairInterface(
            self.globalSettingsInterface,
            self.modelConfigInterface,
            self,
        )

        self.initNavigation()
        self.globalSettingsInterface.languageChanged.connect(self._on_language_changed)

    def _on_language_changed(self, language: str):
        set_language(language)
        self._retranslate_navigation()
        self.modelConfigInterface.retranslate_ui()
        self.autoLyricInterface.retranslate_ui()
        self.sentenceRepairInterface.retranslate_ui()
        self.globalSettingsInterface.retranslate_ui()

    def _retranslate_navigation(self):
        nav = self.navigationInterface
        for page, key in (
            (self.autoLyricInterface, "nav_auto"),
            (self.sentenceRepairInterface, "nav_repair"),
            (self.modelConfigInterface, "nav_model"),
            (self.globalSettingsInterface, "nav_settings"),
        ):
            item = nav.widget(page.objectName())
            if item is not None:
                item.setText(tr(key))

    def _restore_window_state(self):
        """Restore the last geometry, or use a default that fits the screen.

        Qt sizes are logical pixels, so the same numbers scale with the DPI
        factor automatically.  The window stays freely resizable: no
        setFixedSize, otherwise maximize and drag-resize are silently
        disabled.
        """
        self.setMinimumSize(900, 620)
        restored = self.restoreGeometry(self.app_settings.value("main_geometry", b""))
        if not restored:
            screen = self.screen() or QApplication.primaryScreen()
            avail = screen.availableGeometry() if screen is not None else None
            width, height = 1260, 900
            if avail is not None:
                width = min(width, int(avail.width() * 0.92))
                height = min(height, int(avail.height() * 0.92))
            self.resize(width, height)
            self._center_on_available_screen()
        else:
            self._clamp_to_available_screen()
        if self.app_settings.value("main_maximized", False, type=bool):
            self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def _center_on_available_screen(self):
        """Center the window in the available geometry (excludes the taskbar)."""
        avail = self._available_geometry()
        if avail is None:
            return
        geo = self.frameGeometry()
        x = avail.x() + max(0, (avail.width() - geo.width()) // 2)
        y = avail.y() + max(0, (avail.height() - geo.height()) // 2)
        self.move(x, y)

    def _clamp_to_available_screen(self):
        """Keep a restored window reachable after a monitor change."""
        avail = self._available_geometry()
        if avail is None:
            return
        frame = self.frameGeometry()
        # resize() takes the client size: subtract the title bar and borders,
        # otherwise the frame stays wider than the screen and the opposite
        # edge is still cut off.
        deco_w = max(0, frame.width() - self.width())
        deco_h = max(0, frame.height() - self.height())
        max_w = max(self.minimumWidth(), avail.width() - deco_w)
        max_h = max(self.minimumHeight(), avail.height() - deco_h)
        width = min(self.width(), max_w)
        height = min(self.height(), max_h)
        if (width, height) != (self.width(), self.height()):
            self.resize(width, height)
            frame = self.frameGeometry()
        if not avail.intersects(frame):
            self._center_on_available_screen()
            return
        x = min(max(frame.x(), avail.left()), avail.right() - frame.width() + 1)
        y = min(max(frame.y(), avail.top()), avail.bottom() - frame.height() + 1)
        if (x, y) != (frame.x(), frame.y()):
            self.move(x, y)

    def _screen_for_geometry(self):
        """Screen the window actually lives on.

        Before show() Qt answers with the primary screen for ``self.screen()``,
        so a geometry restored onto a secondary monitor would be clamped
        against the wrong screen.  Ask which screen contains it instead.
        """
        screen = QApplication.screenAt(self.frameGeometry().center())
        if screen is None:
            screen = self.screen() or QApplication.primaryScreen()
        return screen

    def _available_geometry(self):
        screen = self._screen_for_geometry()
        return screen.availableGeometry() if screen is not None else None

    def _load_theme(self):
        raw = str(self.globalSettingsInterface.settings.value("theme", "light")).strip().lower()
        return {"dark": Theme.DARK, "auto": Theme.AUTO}.get(raw, Theme.LIGHT)

    def _enable_mica(self):
        # Mica only exists on Windows 11.  The compat window treats this as a
        # no-op, and older builds may not expose the API at all.
        try:
            self.setMicaEffectEnabled(True)
        except Exception:
            pass

    def initNavigation(self):
        self.addSubInterface(self.autoLyricInterface, FluentIcon.MUSIC, tr("nav_auto"))
        self.addSubInterface(self.sentenceRepairInterface, FluentIcon.SYNC, tr("nav_repair"))
        self.addSubInterface(self.modelConfigInterface, FluentIcon.DEVELOPER_TOOLS, tr("nav_model"))
        self.addSubInterface(self.globalSettingsInterface, FluentIcon.SETTING, tr("nav_settings"), position=NavigationItemPosition.BOTTOM)

        self.navigationInterface.setMenuButtonVisible(False)
        
        self.navigationInterface.setExpandWidth(50)
        self.navigationInterface.setMinimumExpandWidth(50)
        self.navigationInterface.setMaximumWidth(50)

        self.navigationInterface.setCurrentItem(self.autoLyricInterface.objectName())
        self.stackedWidget.setCurrentWidget(self.autoLyricInterface)

    def _running_pages(self):
        pages = (self.autoLyricInterface, self.sentenceRepairInterface)
        return [page for page in pages if page.is_running()]

    def closeEvent(self, event):
        # Destroying a running QThread crashes at exit ("QThread: Destroyed
        # while thread is still running"); stop the worker first.
        running = self._running_pages()
        if running:
            answer = QMessageBox.question(
                self,
                tr("close_running_title"),
                tr("close_running_body"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            for page in running:
                page.request_stop()
            pending = [page for page in running if not page.wait_for_worker(5000)]
            if pending:
                # Cancellation is cooperative and the worker can sit in an
                # uninterruptible call (ffmpeg decode allows up to 10 minutes),
                # so hide the window and finish closing once the thread
                # settles.  Save now, while the window still holds its real
                # geometry: the deferred close() runs on a hidden window.
                self._persist_window_state()
                self.hide()
                for page in pending:
                    page.on_worker_settled(self.close)
                event.ignore()
                return
        self._persist_window_state()
        event.accept()

    def _persist_window_state(self):
        """Write the session geometry exactly once, only on a real exit."""
        if getattr(self, "_window_state_saved", False):
            return
        self._window_state_saved = True
        self.app_settings.setValue("main_geometry", self.saveGeometry())
        self.app_settings.setValue("main_maximized", self.isMaximized())


def run_app():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app_icon = _load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


# The frameless Fluent window is intended for the Windows desktop.  On macOS
# use the native Qt shell while keeping this module's public MainWindow import
# stable for callers and tests.
if sys.platform == "darwin":
    from gui.macos_main import MacMainWindow

    MainWindow = MacMainWindow
else:
    MainWindow = _FluentMainWindow
