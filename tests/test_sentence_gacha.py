import json

import pytest
import yaml

from inference.API.ustx_api import save_ustx
from inference.LyricFA.tools.lyric_matcher import strip_lrc_markup
from inference.io.note_io import NoteInfo
from inference.pipeline.sentence_gacha import (
    _note_diagnostics,
    add_reference_take_into_ustx,
    build_takes,
    clip_lyrics_for_window,
    load_manifest,
    parse_lyric_lines,
    resolve_window,
    select_line,
    splice_take_into_ustx,
)

LRC_SAMPLE = """[ti:月夜の君と空の上]
[ar:風華]
[al:いつか、届く、あの空に。]
[00:07.700]词：KAB.
[00:15.410]曲：高井ウララ
[00:30.839]長い髪を結わいて風を
[00:33.767]感じてみるの
[00:37.245]窓の外は青い空に
"""


def test_lrc_lines_match_strip_lrc_markup():
    lines = parse_lyric_lines(LRC_SAMPLE)
    cleaned = strip_lrc_markup(LRC_SAMPLE)
    assert [line.text for line in lines] == cleaned.splitlines()
    assert lines[0].start == pytest.approx(30.839)
    assert lines[0].end == pytest.approx(33.767)
    assert lines[1].start == pytest.approx(33.767)
    assert lines[1].end == pytest.approx(37.245)
    assert lines[-1].end is None


def test_plain_lyrics_keep_all_lines_without_timestamps():
    lines = parse_lyric_lines("鼓動強くなって\n胸を貫いてゆく")
    assert [line.text for line in lines] == ["鼓動強くなって", "胸を貫いてゆく"]
    assert all(line.start is None and line.end is None for line in lines)


def test_select_line_by_index_and_text():
    lines = parse_lyric_lines(LRC_SAMPLE)
    assert select_line(lines, line_number=2).text == "感じてみるの"
    assert select_line(lines, sentence="感じて").text == "感じてみるの"
    with pytest.raises(ValueError, match="out of range"):
        select_line(lines, line_number=99)
    with pytest.raises(ValueError, match="no lyric line"):
        select_line(lines, sentence="存在しない")


def test_select_line_rejects_ambiguous_text():
    lines = parse_lyric_lines("同じ言葉\n別の行\n同じ言葉")
    with pytest.raises(ValueError, match="matches several lines"):
        select_line(lines, sentence="同じ")


def test_resolve_window_pads_line_and_clamps_to_song():
    lines = parse_lyric_lines(LRC_SAMPLE)
    line = lines[0]
    start, end = resolve_window(
        lines, line=line, start=None, end=None,
        pad_before=0.5, pad_after=0.3, total_duration=100.0,
    )
    assert start == pytest.approx(30.339)
    assert end == pytest.approx(34.067)

    start, end = resolve_window(
        lines, line=line, start=None, end=None,
        pad_before=0.5, pad_after=0.3, total_duration=31.0,
    )
    assert end == pytest.approx(31.0)
    assert end - start > 0


def test_resolve_window_requires_bounds_without_timestamps():
    lines = parse_lyric_lines("鼓動強くなって")
    with pytest.raises(ValueError, match="pass --start and --end"):
        resolve_window(
            lines, line=lines[0], start=None, end=None,
            pad_before=0.5, pad_after=0.3, total_duration=100.0,
        )
    start, end = resolve_window(
        lines, line=lines[0], start=10.0, end=14.0,
        pad_before=0.5, pad_after=0.3, total_duration=100.0,
    )
    assert (start, end) == (10.0, 14.0)


def test_clip_lyrics_cover_overlapping_lines_only():
    lines = parse_lyric_lines(LRC_SAMPLE)
    # The padded window reaches 0.3s into the next line, so its text is kept
    # for alignment while the untouched third line is excluded.
    text = clip_lyrics_for_window(lines, 30.0, 34.0)
    assert text.splitlines() == ["長い髪を結わいて風を", "感じてみるの"]
    text = clip_lyrics_for_window(lines, 30.0, 33.5)
    assert text.splitlines() == ["長い髪を結わいて風を"]


def test_build_takes_random_is_seeded():
    base = {"seg_threshold": 0.2, "seg_radius": 0.02, "est_threshold": 0.2, "t0": 0.0, "nsteps": 8}
    first = build_takes(engines=["tifa", "hfa"], strategy="random", takes=3, seed=7, base=base)
    second = build_takes(engines=["tifa", "hfa"], strategy="random", takes=3, seed=7, base=base)
    assert first == second
    assert len(first) == 6
    assert {take.alignment_engine for take in first} == {"tifa", "hfa"}
    for take in first:
        assert 0.12 <= take.seg_threshold <= 0.30
        assert 0.010 <= take.seg_radius <= 0.050
        assert 0.15 <= take.est_threshold <= 0.30
        assert 0.0 <= take.t0 <= 0.30
        assert take.nsteps in (6, 8, 10, 12)


def test_build_takes_grid_subsamples_and_repeat_keeps_params():
    base = {"seg_threshold": 0.2, "seg_radius": 0.02, "est_threshold": 0.2, "t0": 0.0, "nsteps": 8}
    grid = build_takes(
        engines=["tifa"], strategy="grid", takes=4, seed=0, base=base,
        seg_thresholds=[0.1, 0.2, 0.3], est_thresholds=[0.1, 0.2, 0.3],
    )
    assert len(grid) == 4
    assert len({(t.seg_threshold, t.est_threshold) for t in grid}) == 4

    repeated = build_takes(engines=["tifa"], strategy="grid", takes=2, seed=0, base=base,
                           seg_thresholds=[0.1], est_thresholds=[0.1])
    assert {(t.seg_threshold, t.est_threshold) for t in repeated} == {(0.1, 0.1)}

    reps = build_takes(engines=["hfa"], strategy="repeat", takes=3, seed=0, base=base)
    assert len(reps) == 3
    assert len({(t.seg_threshold, t.seg_radius, t.est_threshold, t.t0, t.nsteps) for t in reps}) == 1
    assert [t.label for t in reps] == ["hfa-r01", "hfa-r02", "hfa-r03"]


def test_diagnostics_flag_short_and_repeated_notes():
    notes = [
        NoteInfo(0.0, 0.20, 65.0, "な"),
        NoteInfo(0.20, 0.25, 67.0, "あ"),
        NoteInfo(0.25, 0.30, 67.0, "あ"),
        NoteInfo(0.30, 0.35, 67.0, "あ"),
        NoteInfo(0.60, 1.00, 63.0, "て"),
    ]
    diag = _note_diagnostics(notes, 0.0, 1.0)
    assert diag["short_notes"] == 3
    assert diag["repeated_lyric_notes"] == 2
    assert diag["internal_gaps"] == 1
    assert diag["lyric_sequence"] == "な あ あ あ て"
    assert diag["flag_score"] > 0


def _write_ustx(path, notes, tempo=120.0):
    save_ustx(notes, path, tempo)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_splice_replaces_only_window_notes(tmp_path):
    base_path = tmp_path / "base.ustx"
    take_path = tmp_path / "take.ustx"
    base = _write_ustx(base_path, [
        NoteInfo(1.0, 1.5, 60.0, "ど"),
        NoteInfo(2.2, 2.6, 62.0, "な"),
        NoteInfo(4.5, 5.0, 64.0, "て"),
    ])
    take = _write_ustx(take_path, [
        NoteInfo(2.1, 2.4, 65.0, "な"),
        NoteInfo(3.0, 3.5, 67.0, "あ"),
    ])
    base_part = base["voice_parts"][0]
    base_part["curves"] = [{"abbr": "pitd", "xs": [0, 480], "ys": [0, 10]}]
    base_path.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")

    out_path = tmp_path / "patched.ustx"
    removed, inserted = splice_take_into_ustx(base_path, take_path, out_path, (2.0, 4.0))
    assert (removed, inserted) == (1, 2)

    patched = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    part = patched["voice_parts"][0]
    assert [note["lyric"] for note in part["notes"]] == ["ど", "な", "あ", "て"]
    assert part["curves"] == [{"abbr": "pitd", "xs": [0, 480], "ys": [0, 10]}]
    onsets = [note["position"] / patched["resolution"] * 60.0 / patched["bpm"] for note in part["notes"]]
    assert onsets == pytest.approx([1.0, 2.1, 3.0, 4.5], abs=0.005)


def test_reference_take_mutes_main_track_and_keeps_notes(tmp_path):
    base_path = tmp_path / "base.ustx"
    take_path = tmp_path / "take.ustx"
    _write_ustx(base_path, [
        NoteInfo(1.0, 1.5, 60.0, "ど"),
        NoteInfo(2.2, 2.6, 62.0, "な"),
        NoteInfo(4.5, 5.0, 64.0, "て"),
    ])
    _write_ustx(take_path, [
        NoteInfo(2.1, 2.4, 65.0, "な"),
        NoteInfo(3.0, 3.5, 67.0, "あ"),
        NoteInfo(6.0, 6.5, 69.0, "outside"),
    ])

    out_path = tmp_path / "reference.ustx"
    inserted = add_reference_take_into_ustx(base_path, take_path, out_path, (2.0, 4.0), "V2M参考 tifa-01")
    assert inserted == 2

    result = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert [note["lyric"] for note in result["voice_parts"][0]["notes"]] == ["ど", "な", "て"]
    assert len(result["tracks"]) == 2
    assert result["tracks"][0]["mute"] is True
    reference_track = result["tracks"][1]
    assert reference_track["track_name"] == "V2M参考 tifa-01"
    assert reference_track["mute"] is False
    assert reference_track["phonemizer"] == result["tracks"][0]["phonemizer"]
    reference_part = result["voice_parts"][1]
    assert reference_part["track_no"] == 1
    assert [note["lyric"] for note in reference_part["notes"]] == ["な", "あ"]


def test_splice_trims_notes_at_window_edges(tmp_path):
    base_path = tmp_path / "base.ustx"
    take_path = tmp_path / "take.ustx"
    _write_ustx(base_path, [NoteInfo(1.9, 2.5, 60.0, "ど")])
    _write_ustx(take_path, [NoteInfo(1.7, 2.3, 62.0, "な")])
    out_path = tmp_path / "patched.ustx"
    removed, inserted = splice_take_into_ustx(base_path, take_path, out_path, (2.0, 4.0))
    assert (removed, inserted) == (1, 1)
    patched = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    note = patched["voice_parts"][0]["notes"][0]
    assert note["position"] / patched["resolution"] * 60.0 / patched["bpm"] == pytest.approx(2.0, abs=0.005)
    assert note["duration"] / patched["resolution"] * 60.0 / patched["bpm"] == pytest.approx(0.3, abs=0.005)


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "gacha_manifest.json"
    path.write_text(json.dumps({"window_start": 1.0, "window_end": 2.0, "takes": []}), encoding="utf-8")
    assert load_manifest(path)["window_start"] == 1.0
