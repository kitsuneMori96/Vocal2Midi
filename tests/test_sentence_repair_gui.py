import os

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def page(qapp):
    from gui.global_settings_view import GlobalSettingsInterface
    from gui.model_config_view import ModelConfigInterface
    from gui.sentence_repair_view import SentenceRepairInterface

    settings_ui = GlobalSettingsInterface()
    settings_ui.settings.remove("alignment_engine")
    model_cfg = ModelConfigInterface(settings_ui.settings, settings_ui.project_root)
    view = SentenceRepairInterface(settings_ui, model_cfg)
    yield view
    settings_ui.settings.remove("alignment_engine")


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def _ustx(bpm, notes, resolution=480):
    return {
        "version": 0.6,
        "resolution": resolution,
        "bpm": bpm,
        "tempos": [{"position": 0, "bpm": bpm}],
        "tracks": [{
            "phonemizer": "OpenUtau.Core.DefaultPhonemizer",
            "renderer_settings": {},
            "track_name": "Track1",
            "track_color": "Blue",
            "mute": False,
            "solo": False,
            "volume": 0,
            "pan": 0,
            "track_expressions": [],
            "voice_color_names": [""],
        }],
        "voice_parts": [{
            "position": 0,
            "duration": 4800,
            "notes": notes,
            "curves": [],
            "masked_curves": [],
        }],
    }


def test_page_builds_with_empty_state(page):
    assert page.objectName() == "sentenceRepairInterface"
    assert page.strategy_combo.count() == 3
    assert page.lang_combo.count() == 4
    assert page.line_combo.count() == 1
    assert page.line_combo.currentData() is None
    assert page.line_combo.itemText(0)  # placeholder, not empty
    assert page.cb_tifa.isChecked() != page.cb_hfa.isChecked()


def test_lyrics_parse_and_window_resolution(page, tmp_path):
    lyrics = _write(tmp_path / "song.lrc", "[00:10.00]鼓動強くなって\n[00:15.00]次の行\n")
    page.lyrics_edit.setText(lyrics)

    assert page.line_combo.count() == 2
    assert page.line_combo.currentData() == 0
    line, window, splice = page._resolve_window()
    assert line.text == "鼓動強くなって"
    assert window == pytest.approx((9.5, 15.3))
    assert splice == pytest.approx((10.0, 15.0))

    page.start_edit.setText("20")
    page.end_edit.setText("24.5")
    _line, window, splice = page._resolve_window()
    assert window == pytest.approx((20.0, 24.5))
    assert splice == pytest.approx((20.0, 24.5))


def test_japanese_lyrics_switch_language(page, tmp_path):
    lyrics = _write(tmp_path / "ja.lrc", "[00:10.00]鼓動強くなって\n[00:15.00]次の行\n")
    page.lang_combo.setCurrentIndex(page.lang_combo.findData("zh"))
    page.lyrics_edit.setText(lyrics)
    assert page.lang_combo.currentData() == "ja"
    assert page.lyric_output_combo.currentData() in ("romaji", "kana")


def test_base_ustx_sets_tempo(page, tmp_path):
    base = tmp_path / "base.ustx"
    base.write_text(yaml.safe_dump(_ustx(71.5, [])), encoding="utf-8")
    page.base_edit.setText(str(base))
    assert page.tempo_spin.value() == pytest.approx(71.5)


def test_yue_unchecks_hfa(page):
    page.cb_hfa.setChecked(True)
    index = page.lang_combo.findData("yue")
    page.lang_combo.setCurrentIndex(index)
    assert page.lang_combo.currentData() == "yue"
    assert page.cb_hfa.isChecked() is False


def test_apply_selected_splices_take(page, tmp_path, monkeypatch):
    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [])
    base = tmp_path / "song.ustx"
    base.write_text(
        yaml.safe_dump(_ustx(120.0, [
            {"position": 0, "duration": 480, "lyric": "a"},
            {"position": 12000, "duration": 480, "lyric": "old"},
            {"position": 15000, "duration": 480, "lyric": "b"},
        ])),
        encoding="utf-8",
    )
    take = tmp_path / "tifa-01.ustx"
    take.write_text(
        yaml.safe_dump(_ustx(120.0, [
            {"position": 10500, "duration": 240, "lyric": "new"},
            {"position": 11000, "duration": 240, "lyric": "new2"},
        ])),
        encoding="utf-8",
    )

    page._job_base = base
    page._job_output_dir = tmp_path
    page._last_manifest = {
        "window_start": 9.5,
        "window_end": 15.3,
        "splice_start": 10.0,
        "splice_end": 15.0,
        "takes": [{"take": "tifa-01", "ustx": take.name}],
    }
    page._take_labels = ["tifa-01"]
    page.results_table.insertRow(0)
    page.results_table.selectRow(0)

    page.apply_selected()

    patched = yaml.safe_load(base.read_text(encoding="utf-8"))
    notes = patched["voice_parts"][0]["notes"]
    lyrics = [note["lyric"] for note in notes]
    assert "old" not in lyrics
    assert lyrics == ["a", "new", "new2", "b"]
    assert notes == sorted(notes, key=lambda note: note["position"])
    backup = yaml.safe_load((tmp_path / "song_backup.ustx").read_text(encoding="utf-8"))
    assert "old" in [note["lyric"] for note in backup["voice_parts"][0]["notes"]]


def test_add_reference_selected_mutes_main_track(page, tmp_path, monkeypatch):
    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [])
    base = tmp_path / "song.ustx"
    base.write_text(
        yaml.safe_dump(_ustx(120.0, [
            {"position": 0, "duration": 480, "lyric": "a"},
            {"position": 12000, "duration": 480, "lyric": "old"},
            {"position": 15000, "duration": 480, "lyric": "b"},
        ])),
        encoding="utf-8",
    )
    take = tmp_path / "tifa-01.ustx"
    take.write_text(
        yaml.safe_dump(_ustx(120.0, [
            {"position": 10500, "duration": 240, "lyric": "new"},
            {"position": 11000, "duration": 240, "lyric": "new2"},
        ])),
        encoding="utf-8",
    )

    page._job_base = base
    page._job_output_dir = tmp_path
    page._last_manifest = {
        "window_start": 9.5,
        "window_end": 15.3,
        "splice_start": 10.0,
        "splice_end": 15.0,
        "takes": [{"take": "tifa-01", "ustx": take.name}],
    }
    page._take_labels = ["tifa-01"]
    page.results_table.insertRow(0)
    page.results_table.selectRow(0)

    page.add_reference_selected()

    patched = yaml.safe_load(base.read_text(encoding="utf-8"))
    assert [note["lyric"] for note in patched["voice_parts"][0]["notes"]] == ["a", "old", "b"]
    assert [note["lyric"] for note in patched["voice_parts"][1]["notes"]] == ["new", "new2"]
    assert patched["tracks"][0]["mute"] is True
    assert patched["tracks"][1]["track_name"] == "V2M参考 tifa-01"
    assert patched["tracks"][1]["mute"] is False


def test_apply_without_selection_warns(page, tmp_path):
    page._last_manifest = None
    page._job_base = tmp_path / "base.ustx"
    page._job_output_dir = tmp_path
    page.apply_selected()
    assert page.worker is None


def test_export_curves_requires_inputs(page, tmp_path):
    page.export_curves()
    assert page.worker is None
    assert page.is_running() is False


def test_populate_results_fills_table(page, tmp_path):
    from inference.io.note_io import NoteInfo
    from inference.pipeline.sentence_gacha import (
        TakeParams,
        TakeResult,
        _note_diagnostics,
    )

    notes = [
        NoteInfo(onset=10.0, offset=10.3, pitch=60, lyric="a"),
        NoteInfo(onset=10.3, offset=10.6, pitch=61, lyric="a"),
    ]
    params = TakeParams("tifa-01", "tifa", 0.2, 0.02, 0.2, 0.0, 8)
    result = TakeResult(
        params=params,
        notes=notes,
        ustx_path=tmp_path / "tifa-01.ustx",
        diagnostics=_note_diagnostics(notes, 9.5, 15.3),
    )

    page._populate_results([result])

    assert page.results_table.rowCount() == 1
    assert page.results_table.item(0, 0).text() == "tifa-01"
    assert page.results_table.item(0, 7).text() == "a a"
    assert page._take_labels == ["tifa-01"]
    assert page.results_table.currentRow() == 0


def test_function_worker_returns_result():
    from gui.fluent_worker import FunctionWorker

    worker = FunctionWorker(lambda session=None, cancel_checker=None: "done")
    worker.start()
    assert worker.wait(5000)
    assert worker.result == "done"
