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

        self.initNavigation()
        self.globalSettingsInterface.languageChanged.connect(self._on_language_changed)

    def _on_language_changed(self, language: str):
        set_language(language)
        self._retranslate_navigation()
        self.modelConfigInterface.retranslate_ui()
        self.autoLyricInterface.retranslate_ui()
        self.globalSettingsInterface.retranslate_ui()

    def _retranslate_navigation(self):
        nav = self.navigationInterface
        for page, key in (
            (self.autoLyricInterface, "nav_auto"),
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
        geo = self.frameGeometry()
        if not avail.intersects(geo):
            self.resize(
                min(self.width(), avail.width()),
                min(self.height(), avail.height()),
            )
            self._center_on_available_screen()
            return
        x = min(max(geo.x(), avail.left()), avail.right() - geo.width() + 1)
        y = min(max(geo.y(), avail.top()), avail.bottom() - geo.height() + 1)
        self.move(x, y)

    def _available_geometry(self):
        screen = self.screen() or QApplication.primaryScreen()
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
        self.addSubInterface(self.modelConfigInterface, FluentIcon.DEVELOPER_TOOLS, tr("nav_model"))
        self.addSubInterface(self.globalSettingsInterface, FluentIcon.SETTING, tr("nav_settings"), position=NavigationItemPosition.BOTTOM)

        self.navigationInterface.setMenuButtonVisible(False)
        
        self.navigationInterface.setExpandWidth(50)
        self.navigationInterface.setMinimumExpandWidth(50)
        self.navigationInterface.setMaximumWidth(50)

        self.navigationInterface.setCurrentItem(self.autoLyricInterface.objectName())
        self.stackedWidget.setCurrentWidget(self.autoLyricInterface)

    def closeEvent(self, event):
        self.app_settings.setValue("main_geometry", self.saveGeometry())
        self.app_settings.setValue("main_maximized", self.isMaximized())
        # Destroying a running QThread crashes at exit ("QThread: Destroyed
        # while thread is still running"); stop the worker first.
        if not self.autoLyricInterface.is_running():
            event.accept()
            return
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
        self.autoLyricInterface.request_stop()
        if self.autoLyricInterface.wait_for_worker(5000):
            event.accept()
            return
        # Cancellation is cooperative and the worker can sit in an
        # uninterruptible call (ffmpeg decode allows up to 10 minutes), so hide
        # the window and finish closing once the thread actually settles.
        self.hide()
        self.autoLyricInterface.on_worker_settled(self.close)
        event.ignore()


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
