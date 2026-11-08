import json

import numpy as np
import pytest
import yaml

from inference.API.rmvpe_api import RmvpeResult
from inference.API.ustx_api import save_ustx
from inference.io.note_io import NoteInfo
from inference.pipeline.ustx_curves import (
    CURVE_EXPRESSIONS,
    _clip_runs,
    build_f0_runs,
    build_vuv_runs,
    inject_curves,
    seconds_to_ticks,
)

TEMPOS = [{"position": 0, "bpm": 120}]
RESOLUTION = 480


def _rmvpe(midi, voiced=None, step=0.01, offset=0.0):
    midi = np.asarray(midi, dtype=float)
    mask = np.ones(len(midi), dtype=bool) if voiced is None else np.asarray(voiced, dtype=bool)
    return RmvpeResult(time_step_seconds=step, midi_pitch=midi, voiced_mask=mask, time_offset=offset)


def test_seconds_to_ticks_uses_resolution():
    # 120bpm, 480 ticks/beat -> 960 ticks per second
    assert seconds_to_ticks(0.0, TEMPOS, RESOLUTION) == 0
    assert seconds_to_ticks(1.0, TEMPOS, RESOLUTION) == 960
    assert seconds_to_ticks(1.5, TEMPOS, RESOLUTION) == 1440


def test_seconds_to_ticks_follows_tempo_changes():
    tempos = [{"position": 0, "bpm": 120}, {"position": 960, "bpm": 60}]
    # 1s at 120bpm -> tick 960; next 1s at 60bpm -> +480
    assert seconds_to_ticks(1.0, tempos, RESOLUTION) == 960
    assert seconds_to_ticks(2.0, tempos, RESOLUTION) == 1440


def test_f0_runs_hold_midi_cents_on_the_grid():
    # 10ms frames at 120bpm/480ppqn -> 9.6 ticks; the run is resampled to 5-tick steps.
    rmvpe = _rmvpe([65.0] * 6, step=0.01)
    runs = build_f0_runs(rmvpe, TEMPOS, RESOLUTION)
    assert len(runs) == 1
    assert runs[0]["x"] == 0
    assert set(runs[0]["ys"]) == {6500}
    ticks = [runs[0]["x"] + i * 5 for i in range(len(runs[0]["ys"]))]
    assert ticks == list(range(0, 51, 5))


def test_f0_runs_are_split_by_silence():
    rmvpe = _rmvpe([65.0] * 10, voiced=[True] * 5 + [False] * 5, step=0.01)
    runs = build_f0_runs(rmvpe, TEMPOS, RESOLUTION)
    assert len(runs) == 1
    assert runs[0]["x"] == 0
    assert len(runs[0]["ys"]) == 9


def test_f0_runs_interpolate_between_frames():
    rmvpe = _rmvpe([60.0, 72.0], step=0.02)
    runs = build_f0_runs(rmvpe, TEMPOS, RESOLUTION)
    ys = runs[0]["ys"]
    assert ys[0] == 6000
    assert ys[-1] == 7200
    assert 6000 < ys[len(ys) // 2] < 7200


def test_f0_runs_subtract_part_offset_and_apply_time_offset():
    rmvpe = _rmvpe([60.0], step=0.01, offset=1.0)
    runs = build_f0_runs(rmvpe, TEMPOS, RESOLUTION, part_offset_ticks=480)
    # 1s -> tick 960, minus part offset 480
    assert runs[0]["x"] == 480


def test_vuv_runs_cover_only_voiced_frames():
    rmvpe = _rmvpe([60] * 8, voiced=[True, True, False, False, True, True, True, True], step=0.01)
    runs = build_vuv_runs(rmvpe, TEMPOS, RESOLUTION)
    assert [run["x"] for run in runs] == [0, 40]
    assert all(set(run["ys"]) == {100} for run in runs)
    assert len(runs[0]["ys"]) == 3
    assert len(runs[1]["ys"]) == 6


def test_clip_runs_splits_a_straddling_run():
    runs = [{"x": 0, "ys": list(range(10))}]
    clipped = _clip_runs(runs, 10, 20)
    assert [run["x"] for run in clipped] == [0, 25]
    assert clipped[0]["ys"] == [0, 1]
    assert clipped[1]["ys"] == [5, 6, 7, 8, 9]


def _write_base(tmp_path, position=0):
    path = tmp_path / "base.ustx"
    notes = [NoteInfo(onset=0.0, offset=1.0, pitch=65, lyric="な")]
    save_ustx(notes, path, tempo=120.0, rmvpe_result=_rmvpe([65.0, 65.0]))
    with path.open("r", encoding="utf-8") as f:
        project = yaml.safe_load(f)
    project["voice_parts"][0]["position"] = position
    project["voice_parts"][0]["masked_curves"] = []
    path.write_text(yaml.safe_dump(project, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _load(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_inject_adds_expressions_selectors_and_masked_curves(tmp_path):
    base = _write_base(tmp_path)
    out = tmp_path / "out.ustx"
    summary = inject_curves(
        base, out,
        {"f0": [{"x": 0, "ys": [6500, 6600]}], "vuv": [{"x": 0, "ys": [100, 100]}]},
        window_seconds=(0.0, 2.0),
    )
    project = _load(out)
    part = project["voice_parts"][0]
    abbrs = [curve["abbr"] for curve in part["masked_curves"]]
    assert abbrs == ["f0", "vuv"]
    assert part["masked_curves"][0]["runs"] == [{"x": 0, "ys": [6500, 6600]}]
    assert project["expressions"]["f0"] == CURVE_EXPRESSIONS["f0"]
    assert project["expressions"]["f0"]["type"] == "MaskedCurve"
    assert {"f0", "vuv"} <= set(project["exp_selectors"])
    assert summary["curves"]["f0"]["runs"] == 1
    saved = json.loads(out.with_name("out.curves.json").read_text(encoding="utf-8"))
    assert saved["window_ticks"] == [0, 1920]


def test_inject_preserves_notes_pitd_and_tempo(tmp_path):
    base = _write_base(tmp_path)
    before = _load(base)
    out = tmp_path / "out.ustx"
    inject_curves(base, out, {"f0": [{"x": 0, "ys": [6500]}]}, window_seconds=(0.0, 1.0))
    after = _load(out)
    assert after["voice_parts"][0]["notes"] == before["voice_parts"][0]["notes"]
    assert after["voice_parts"][0]["curves"] == before["voice_parts"][0]["curves"]
    assert after["tempos"] == before["tempos"]


def test_inject_replaces_only_inside_window(tmp_path):
    base = _write_base(tmp_path)
    first = tmp_path / "first.ustx"
    # 300 values span ticks 0..1495, i.e. past the 1s window of the second pass.
    inject_curves(base, first, {"f0": [{"x": 0, "ys": [6500] * 300}]}, window_seconds=(0.0, 2.0))
    second = tmp_path / "second.ustx"
    inject_curves(first, second, {"f0": [{"x": 0, "ys": [6600, 6600]}]}, window_seconds=(0.0, 1.0))
    curve = next(c for c in _load(second)["voice_parts"][0]["masked_curves"] if c["abbr"] == "f0")
    assert [run["x"] for run in curve["runs"]] == [0, 965]
    assert curve["runs"][0]["ys"] == [6600, 6600]
    assert len(curve["runs"][1]["ys"]) == 107


def test_inject_preserve_existing_keeps_old_runs(tmp_path):
    base = _write_base(tmp_path)
    first = tmp_path / "first.ustx"
    inject_curves(base, first, {"f0": [{"x": 0, "ys": [6500]}]}, window_seconds=(0.0, 1.0))
    second = tmp_path / "second.ustx"
    inject_curves(
        first, second, {"f0": [{"x": 480, "ys": [6700]}]},
        window_seconds=(0.0, 1.0), preserve_existing=True,
    )
    curve = next(c for c in _load(second)["voice_parts"][0]["masked_curves"] if c["abbr"] == "f0")
    assert [run["x"] for run in curve["runs"]] == [0, 480]


def test_inject_is_idempotent(tmp_path):
    base = _write_base(tmp_path)
    runs = {"f0": [{"x": 0, "ys": [6500, 6600]}]}
    first = tmp_path / "first.ustx"
    inject_curves(base, first, runs, window_seconds=(0.0, 2.0))
    second = tmp_path / "second.ustx"
    inject_curves(first, second, runs, window_seconds=(0.0, 2.0))
    project = _load(second)
    assert [c["abbr"] for c in project["voice_parts"][0]["masked_curves"]].count("f0") == 1
    curve = next(c for c in project["voice_parts"][0]["masked_curves"] if c["abbr"] == "f0")
    assert curve["runs"] == [{"x": 0, "ys": [6500, 6600]}]


def test_inject_refuses_overwriting_base(tmp_path):
    base = _write_base(tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        inject_curves(base, base, {"f0": [{"x": 0, "ys": [6500]}]})


def test_inject_respects_part_position(tmp_path):
    base = _write_base(tmp_path, position=480)
    out = tmp_path / "out.ustx"
    inject_curves(base, out, {"f0": [{"x": 0, "ys": [6500]}]}, window_seconds=(1.0, 2.0))
    project = _load(out)
    assert project["voice_parts"][0]["position"] == 480
    assert project["voice_parts"][0]["masked_curves"][0]["runs"] == [{"x": 0, "ys": [6500]}]


def test_cli_curve_names_default_and_validation():
    from scripts.sentence_curves import _parse_curves

    assert _parse_curves(None) == ["f0", "vuv"]
    assert _parse_curves("vuv") == ["vuv"]
    with pytest.raises(ValueError, match="unknown curve"):
        _parse_curves("dyn")


def test_cli_rejects_missing_base(tmp_path):
    from scripts.sentence_curves import main

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"")
    assert main([str(audio), "--base", str(tmp_path / "missing.ustx")]) == 2


def test_cli_rejects_overwriting_base(tmp_path):
    from scripts.sentence_curves import main

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"")
    base = _write_base(tmp_path)
    assert main([str(audio), "--base", str(base), "-o", str(base)]) == 2


def test_ticks_to_bar_beat_matches_ruler():
    from inference.pipeline.ustx_curves import ticks_to_bar_beat

    sigs = [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}]
    assert ticks_to_bar_beat(0, sigs, 480) == (1, 1.0)
    assert ticks_to_bar_beat(1920, sigs, 480) == (2, 1.0)
    assert ticks_to_bar_beat(2880, sigs, 480) == (2, 3.0)
    assert ticks_to_bar_beat(1919, sigs, 480)[0] == 1


def test_ticks_to_bar_beat_honours_signature_change():
    from inference.pipeline.ustx_curves import ticks_to_bar_beat

    sigs = [
        {"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4},
        {"bar_position": 4, "beat_per_bar": 3, "beat_unit": 4},
    ]
    assert ticks_to_bar_beat(4 * 1920 + 960, sigs, 480) == (5, 3.0)


def test_bar_beat_label_uses_tempo_map():
    from application.sentence_jobs import bar_beat_label

    header = {
        "project": {"time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}]},
        "tempos": [{"position": 0, "bpm": 120}],
        "resolution": 480,
    }
    assert bar_beat_label(2.0, header) == "2.1"
    assert bar_beat_label(3.0, header) == "2.3"
    assert bar_beat_label(1.0, None) == ""
