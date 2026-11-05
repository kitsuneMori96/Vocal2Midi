"""Save-dir follow: the newest added/dropped file decides the folder.

The old rule used ``os.path.commonpath`` over every parent directory, which
walked up to ``D:\\`` for a batch spread over sibling folders, and it ran on
``filesChanged``, so deleting a row also moved the save dir.  Both are
regressions now: additions follow the last added file, removals never move.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def clean_settings():
    from gui.global_settings_view import GlobalSettingsInterface

    ui = GlobalSettingsInterface()
    keys = ("enable_batch_mode", "save_dir", "save_dir_follow")
    for key in keys:
        ui.settings.remove(key)
    yield
    for key in keys:
        ui.settings.remove(key)


@pytest.fixture
def view(qapp):
    from gui.auto_lyric_view import AutoLyricInterface
    from gui.global_settings_view import GlobalSettingsInterface
    from gui.model_config_view import ModelConfigInterface

    settings_ui = GlobalSettingsInterface()
    model_cfg = ModelConfigInterface(settings_ui.settings, settings_ui.project_root)
    widget = AutoLyricInterface(settings_ui, model_cfg)
    yield widget
    widget.close()


def test_follows_folder_of_the_newest_file(view, tmp_path):
    first = tmp_path / "a" / "one.wav"
    second = tmp_path / "b" / "two.wav"
    view.audio_list.add_paths([str(first)])
    assert view.save_dir_edit.text() == str(first.parent)

    view.audio_list.add_paths([str(second)])
    # Not the common parent (tmp_path): the file just added wins.
    assert view.save_dir_edit.text() == str(second.parent)


def test_cross_drive_addition_follows_the_new_file(view):
    view.audio_list.add_paths(["D:/music/a/one.wav"])
    view.audio_list.add_paths(["E:/other/two.wav"])
    assert view.save_dir_edit.text() == os.path.normpath("E:/other")


def test_removing_a_file_never_moves_the_save_dir(view, tmp_path):
    view.audio_list.add_paths([str(tmp_path / "a" / "one.wav")])
    view.audio_list.add_paths([str(tmp_path / "b" / "two.wav")])
    expected = view.save_dir_edit.text()

    view.audio_list.remove_item(view.audio_list.item(1))
    assert view.save_dir_edit.text() == expected

    view.clear_audio_files()
    from gui.settings_utils import default_output_dir

    assert view.save_dir_edit.text() == str(
        default_output_dir(view.global_settings.project_root)
    )
    assert view.follow_dir_switch.isChecked() is True


def test_switch_off_freezes_and_switch_on_refollows(view, tmp_path):
    first = tmp_path / "a" / "one.wav"
    second = tmp_path / "b" / "two.wav"
    view.audio_list.add_paths([str(first)])
    view.follow_dir_switch.setChecked(False)
    frozen = view.save_dir_edit.text()

    view.audio_list.add_paths([str(second)])
    assert view.save_dir_edit.text() == frozen

    view.follow_dir_switch.setChecked(True)
    assert view.save_dir_edit.text() == str(second.parent)


def test_manual_edit_disables_following(view, tmp_path):
    view.audio_list.add_paths([str(tmp_path / "a" / "one.wav")])
    custom = str(tmp_path / "out")
    view.save_dir_edit.setText(custom)

    assert view.follow_dir_switch.isChecked() is False
    view.audio_list.add_paths([str(tmp_path / "b" / "two.wav")])
    assert view.save_dir_edit.text() == custom


def test_picker_opens_in_the_last_used_folder(view, monkeypatch, tmp_path):
    from gui import auto_lyric_view

    folder = tmp_path / "a"
    folder.mkdir()
    view.audio_list.add_paths([str(folder / "one.wav")])
    seen = {}

    def fake_get_open_file_names(parent, title, directory, filters):
        seen["dir"] = directory
        return [], ""

    def fake_get_open_file_name(parent, title, directory, filters):
        seen["dir"] = directory
        return "", ""

    monkeypatch.setattr(auto_lyric_view.QFileDialog, "getOpenFileNames", fake_get_open_file_names)
    monkeypatch.setattr(auto_lyric_view.QFileDialog, "getOpenFileName", fake_get_open_file_name)

    view.add_audio_files()
    assert seen["dir"] == str(folder)


def test_picker_prefers_save_dir_when_following_is_off(view, monkeypatch, tmp_path):
    from gui import auto_lyric_view

    view.audio_list.add_paths([str(tmp_path / "a" / "one.wav")])
    custom = tmp_path / "out"
    custom.mkdir()
    view.save_dir_edit.setText(str(custom))
    assert view.follow_dir_switch.isChecked() is False

    seen = {}

    def fake_get_open_file_names(parent, title, directory, filters):
        seen["dir"] = directory
        return [], ""

    monkeypatch.setattr(auto_lyric_view.QFileDialog, "getOpenFileNames", fake_get_open_file_names)
    view.add_audio_files()
    assert seen["dir"] == str(custom)
