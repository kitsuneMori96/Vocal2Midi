import numpy as np
import pytest

import inference.API.game_api as game_api
from inference.API.game_api import (
    _continuation_lyric,
    collapse_vibrato_tails,
    extract_pitches_and_align,
    extract_pitches_only,
    extract_vowel_boundaries,
)
from inference.io.note_io import NoteInfo
from inference.HubertFA.tools.align_word import Phoneme, Word


class _FailingGameModel:
    timestep = 0.01

    def infer_batch(self, **kwargs):
        raise RuntimeError("GAME exploded")


class _CapturingGameModel:
    timestep = 0.01

    def __init__(self):
        self.calls = []

    def infer_batch(self, **kwargs):
        self.calls.append(kwargs)
        return [
            (
                np.array([0.5], dtype=np.float32),
                np.array([1], dtype=np.int64),
                np.array([60.0], dtype=np.float32),
            )
        ]


def _make_word(start, end, text, phones):
    word = Word(start, end, text)
    word.phonemes = [Phoneme(ph_start, ph_end, ph_text) for ph_start, ph_end, ph_text in phones]
    return word


def test_no_lyrics_game_inference_error_is_not_swallowed():
    chunks = [{"waveform": np.zeros(32, dtype=np.float32), "offset": 0.0}]

    with pytest.raises(RuntimeError, match="GAME exploded"):
        extract_pitches_only(
            chunks,
            sr=16000,
            game_model=_FailingGameModel(),
            ts=[0.0],
            seg_threshold=0.2,
            seg_radius=0.02,
            est_threshold=0.2,
            batch_size=1,
        )


def test_game_language_is_forced_empty_semantics():
    chunks = [{"waveform": np.zeros(32, dtype=np.float32), "offset": 0.0}]
    model = _CapturingGameModel()

    notes = extract_pitches_only(
        chunks,
        sr=16000,
        game_model=model,
        ts=[0.0],
        seg_threshold=0.2,
        seg_radius=0.02,
        est_threshold=0.2,
        batch_size=1,
        language="ja",
    )

    assert len(notes) == 1
    assert len(model.calls) == 1
    assert model.calls[0]["language"] == 0


def test_extract_vowel_boundaries_ja_uses_first_singable_nucleus():
    words = [
        _make_word(0.0, 0.25, "ka", [(0.0, 0.08, "k"), (0.08, 0.25, "a")]),
        _make_word(0.25, 0.50, "ni", [(0.25, 0.33, "n"), (0.33, 0.50, "i")]),
    ]

    word_durs, word_vuvs, lyrics, _vowels = extract_vowel_boundaries(words, ["か", "に"], language="ja")

    assert word_durs == pytest.approx([0.08, 0.25, 0.17])
    assert word_vuvs == [0, 1, 1]
    assert lyrics == ["", "か", "に"]


def test_extract_vowel_boundaries_zh_does_not_use_coda_as_vowel_start():
    words = [
        _make_word(0.0, 0.30, "ang", [(0.0, 0.22, "a"), (0.22, 0.30, "ng")]),
        _make_word(0.30, 0.60, "ni", [(0.30, 0.36, "n"), (0.36, 0.60, "i")]),
    ]

    word_durs, word_vuvs, lyrics, _vowels = extract_vowel_boundaries(words, ["昂", "你"], language="zh")

    assert word_durs == pytest.approx([0.36, 0.24])
    assert word_vuvs == [1, 1]
    assert lyrics == ["昂", "你"]


def test_extract_vowel_boundaries_en_uses_arpabet_vowels():
    words = [
        _make_word(0.0, 0.40, "fall", [(0.0, 0.05, "f"), (0.05, 0.30, "ao"), (0.30, 0.40, "l")]),
        _make_word(0.40, 0.70, "fly", [(0.40, 0.45, "f"), (0.45, 0.50, "l"), (0.50, 0.70, "ay")]),
    ]

    word_durs, word_vuvs, lyrics, _vowels = extract_vowel_boundaries(
        words, ["fall", "fly"], language="en"
    )

    # zh/ja end anchoring: each note starts at its vowel and extends to the
    # next word's vowel, so word-initial consonants are absorbed into notes
    # and the phrase is contiguous; only the leading "f" is a vuvs=0 gap.
    assert word_durs == pytest.approx([0.05, 0.45, 0.20])
    assert word_vuvs == [0, 1, 1]
    assert lyrics == ["", "fall", "fly"]


def test_extract_vowel_boundaries_en_onsets_absorbed_notes_contiguous():
    words = [
        _make_word(0.0, 0.40, "my", [(0.0, 0.10, "m"), (0.10, 0.40, "ay")]),
        _make_word(
            0.40, 0.85, "stop",
            [(0.40, 0.46, "s"), (0.46, 0.52, "t"), (0.52, 0.75, "aa"), (0.75, 0.85, "p")],
        ),
    ]

    word_durs, word_vuvs, lyrics, _vowels = extract_vowel_boundaries(
        words, ["my", "stop"], language="en"
    )

    # "stop"'s unvoiced "s t" onset is absorbed into the preceding "my" note
    # (my spans 0.10-0.52) so the two notes butt together with no hole; only
    # the first word's own onset "m" is a gap (nothing precedes it).
    assert word_durs == pytest.approx([0.10, 0.42, 0.33])
    assert word_vuvs == [0, 1, 1]
    assert lyrics == ["", "my", "stop"]


def test_extract_vowel_boundaries_en_pause_breaks_absorption():
    words = [
        _make_word(0.0, 0.40, "my", [(0.0, 0.10, "m"), (0.10, 0.40, "ay")]),
        _make_word(0.40, 0.90, "SP", []),
        _make_word(
            0.90, 1.35, "stop",
            [(0.90, 0.96, "s"), (0.96, 1.02, "t"), (1.02, 1.25, "aa"), (1.25, 1.35, "p")],
        ),
    ]

    word_durs, word_vuvs, lyrics, _vowels = extract_vowel_boundaries(
        words, ["my", "stop"], language="en"
    )

    # A real pause (SP) breaks the chain like in zh/ja: "my" ends at its word
    # end, the SP is a vuvs=0 gap, and "stop"'s onset after the pause becomes
    # a gap too (there is no sung note before it to absorb into).
    assert word_durs == pytest.approx([0.10, 0.30, 0.50, 0.12, 0.33])
    assert word_vuvs == [0, 1, 0, 0, 1]
    assert lyrics == ["", "my", "", "", "stop"]


def test_extract_vowel_boundaries_en_multisyllable_starts_at_vowels():
    words = [
        _make_word(
            0.0, 0.86, "possible",
            [
                (0.00, 0.06, "p"), (0.06, 0.22, "aa"), (0.22, 0.30, "s"),
                (0.30, 0.46, "ax"), (0.46, 0.52, "b"), (0.52, 0.74, "ax"),
                (0.74, 0.86, "l"),
            ],
        ),
    ]

    word_durs, word_vuvs, lyrics, _vowels = extract_vowel_boundaries(
        words, ["possible"], language="en"
    )

    # three vowel nuclei -> three chunks; the word-initial p becomes an
    # unvoiced gap entry while inter-syllable onsets (s, b) stay inside the
    # preceding note span, exactly like the zh/ja semantics
    assert word_durs == pytest.approx([0.06, 0.24, 0.22, 0.34])
    assert word_vuvs == [0, 1, 1, 1]
    assert lyrics == ["", "possible", "+", "+"]


def test_en_singable_phones_reject_consonant_v_names():
    from inference.API.game_api import _is_singable_phone

    assert _is_singable_phone("en/ao", "en")
    assert _is_singable_phone("ay", "en")
    assert not _is_singable_phone("en/v", "en")
    assert not _is_singable_phone("en/_r", "en")


def _note(onset, dur, pitch, lyric):
    return NoteInfo(onset=onset, offset=onset + dur, pitch=pitch, lyric=lyric)


def test_continuation_lyric_reuses_previous_vowel():
    assert _continuation_lyric("お", "ja", "kana") == "お"
    assert _continuation_lyric("の", "ja", "kana") == "お"
    assert _continuation_lyric("o", "ja", "romaji") == "o"
    assert _continuation_lyric("ni", "zh", "pinyin") == "i"
    assert _continuation_lyric("fall", "en", "word") is None
    assert _continuation_lyric("-", "ja", "kana") is None


class _DummyGameModel:
    timestep = 0.01


def _run_with_fake_game(monkeypatch, words, chars, durations, presence, scores):
    chunk = {"waveform": np.zeros(16000 * 4, dtype=np.float32), "offset": 0.0}
    monkeypatch.setattr(
        game_api,
        "_run_game_inference_batch",
        lambda **kwargs: [(
            np.asarray(durations, dtype=np.float32),
            np.asarray(presence, dtype=np.int64),
            np.asarray(scores, dtype=np.float32),
        )],
    )
    notes, _ = extract_pitches_and_align(
        [chunk],
        sr=16000,
        pred_dict={"chunk_0": (None, None, words)},
        chars_dict={"chunk_0": chars},
        game_model=_DummyGameModel(),
        ts=[0.0],
        seg_threshold=0.2,
        seg_radius=0.02,
        est_threshold=0.2,
        batch_size=1,
        language="ja",
        lyric_output_mode="kana",
    )
    return notes


def test_note_in_unvoiced_word_continues_previous_vowel(monkeypatch):
    words = [
        _make_word(0.0, 0.30, "SP", [(0.0, 0.30, "SP")]),
        _make_word(0.30, 0.70, "a", [(0.30, 0.70, "a")]),
        _make_word(0.70, 0.90, "SP", [(0.70, 0.90, "SP")]),
        _make_word(0.90, 1.30, "a", [(0.90, 1.30, "a")]),
    ]

    notes = _run_with_fake_game(
        monkeypatch, words, ["な", "が"],
        durations=[0.30, 0.40, 0.20, 0.40],
        presence=[1, 1, 1, 1],
        scores=[60, 62, 64, 65],
    )

    # The leading orphan is dropped; the in-phrase unvoiced note sings あ
    # instead of exporting a bare '+' continuation.
    assert [n.lyric for n in notes] == ["な", "あ", "が"]
    assert [round(n.onset, 3) for n in notes] == [0.30, 0.70, 0.90]
    assert [round(n.offset, 3) for n in notes] == [0.70, 0.90, 1.30]


def test_unvoiced_word_note_after_phrase_gap_is_dropped(monkeypatch):
    words = [
        _make_word(0.0, 0.30, "SP", [(0.0, 0.30, "SP")]),
        _make_word(0.30, 0.70, "a", [(0.30, 0.70, "a")]),
        _make_word(0.70, 2.00, "SP", [(0.70, 2.00, "SP")]),
        _make_word(2.00, 2.40, "a", [(2.00, 2.40, "a")]),
    ]

    notes = _run_with_fake_game(
        monkeypatch, words, ["な", "が"],
        durations=[0.30, 0.40, 1.10, 0.20, 0.40],
        presence=[1, 1, 0, 1, 1],
        scores=[60, 62, 0, 64, 65],
    )

    assert [n.lyric for n in notes] == ["な", "が"]
    assert notes[-1].onset == pytest.approx(2.00)


# Real GAME output for 月夜の君と空の上 110.17-111.22s: the sustained 「イ」 tail
# of 「に」 wobbles and the model shredded it into six short fragments.
def test_collapse_vibrato_tails_absorbs_measured_phrase_final_vibrato():
    notes = [_note(*args) for args in [
        (110.1694, 0.291, 60.0, "ni"),
        (110.4602, 0.089, 58.0, "イ"),
        (110.5496, 0.120, 61.0, "イ"),
        (110.6697, 0.111, 59.0, "イ"),
        (110.7804, 0.120, 62.0, "イ"),
        (110.9005, 0.109, 57.0, "イ"),
        (111.0099, 0.211, 62.0, "イ"),
    ]]

    logs = collapse_vibrato_tails(notes)

    assert len(notes) == 1
    assert notes[0].lyric == "ni"
    assert notes[0].pitch == pytest.approx(60.0)
    assert notes[0].offset == pytest.approx(111.2207, abs=1e-3)
    assert any("absorbed" in line for line in logs)


def test_collapse_vibrato_tails_keeps_mid_phrase_fragments():
    # 107.86s: the 「エ」 tail is also shredded, but another word follows
    # immediately, so absorbing would swallow a real syllable.
    notes = [
        _note(107.0500, 0.810, 60.0, "ge"),
        _note(107.8602, 0.080, 59.0, "エ"),
        _note(107.9402, 0.170, 60.0, "エ"),
        _note(108.1097, 0.151, 60.0, "エ"),
        _note(108.2604, 0.129, 61.0, "no"),
    ]
    before = [(n.onset, n.offset, n.pitch, n.lyric) for n in notes]

    logs = collapse_vibrato_tails(notes)

    assert [(n.onset, n.offset, n.pitch, n.lyric) for n in notes] == before
    assert logs == []


def test_collapse_vibrato_tails_keeps_stepping_pitches_at_phrase_end():
    # Same lyric and phrase-final, but the pitch steps upward instead of
    # alternating around one centre: that is a real melisma, not vibrato.
    notes = [
        _note(20.000, 0.300, 60.0, "ni"),
        _note(20.300, 0.120, 61.0, "イ"),
        _note(20.420, 0.120, 63.0, "イ"),
        _note(20.540, 0.120, 65.0, "イ"),
    ]

    logs = collapse_vibrato_tails(notes)

    assert len(notes) == 4
    assert any("kept" in line for line in logs)


def test_collapse_vibrato_tails_requires_matching_previous_vowel():
    notes = [
        _note(30.000, 0.300, 60.0, "ka"),
        _note(30.300, 0.120, 59.0, "イ"),
        _note(30.420, 0.120, 61.0, "イ"),
        _note(30.540, 0.120, 59.0, "イ"),
    ]

    assert collapse_vibrato_tails(notes) == []
    assert len(notes) == 4


def test_collapse_vibrato_tails_requires_vowel_only_kana_lyric():
    notes = [
        _note(40.000, 0.300, 64.0, "yo"),
        _note(40.300, 0.200, 60.0, "ko"),
        _note(40.500, 0.200, 59.0, "ko"),
        _note(40.700, 0.200, 61.0, "ko"),
    ]

    assert collapse_vibrato_tails(notes) == []
    assert len(notes) == 4


def test_collapse_vibrato_tails_requires_short_fragments():
    notes = [
        _note(50.000, 0.300, 60.0, "ni"),
        _note(50.300, 0.400, 59.0, "イ"),
        _note(50.700, 0.120, 61.0, "イ"),
        _note(50.820, 0.120, 59.0, "イ"),
    ]

    assert collapse_vibrato_tails(notes) == []
    assert len(notes) == 4


def test_collapse_vibrato_tails_keeps_whole_song_notes_list_identity():
    notes = [_note(0.0, 0.5, 60.0, "ni"), _note(0.5, 0.2, 59.0, "イ"),
             _note(0.7, 0.2, 61.0, "イ"), _note(0.9, 0.2, 59.0, "イ")]
    kept = notes

    collapse_vibrato_tails(notes)

    assert notes is kept
    assert len(notes) == 1


def test_load_game_model_reuses_cached_session(monkeypatch, tmp_path):
    """The GAME model is cached per (path, device) and reused across runs:
    per-song DirectML session teardown access-violates onnxruntime on the
    next song, so the loader must not rebuild the session."""
    from inference.API import game_api

    constructions = []

    class _FakeGameModel:
        def __init__(self, path, requested_device=None):
            constructions.append((str(path), requested_device))
            self.provider_name = "CPUExecutionProvider"

    monkeypatch.setattr(game_api, "GameOnnxModel", _FakeGameModel)
    game_api._GAME_MODEL_CACHE.clear()
    try:
        first = game_api.load_game_model(str(tmp_path), device="cpu")
        second = game_api.load_game_model(str(tmp_path), device="cpu")
        assert first is second
        assert len(constructions) == 1

        other = game_api.load_game_model(str(tmp_path / "other"), device="dml")
        assert other is not first
        assert len(constructions) == 2
    finally:
        game_api._GAME_MODEL_CACHE.clear()


def test_extract_vowel_boundaries_cantonese_jyutping_not_overridden_by_hanzi():
    words = [
        _make_word(0.0, 0.3, "冷", [(0.0, 0.1, "l"), (0.1, 0.3, "aang")]),
        _make_word(0.3, 0.6, "风", [(0.3, 0.4, "f"), (0.4, 0.6, "ung")]),
    ]
    _, _, lyrics, _ = extract_vowel_boundaries(words, ["laang", "fung"], language="yue")
    assert lyrics == ["", "laang", "fung"]


def test_extract_vowel_boundaries_mandarin_pinyin_not_overridden_by_hanzi():
    words = [
        _make_word(0.0, 0.3, "冷", [(0.0, 0.1, "l"), (0.1, 0.3, "eng")]),
        _make_word(0.3, 0.6, "风", [(0.3, 0.4, "f"), (0.4, 0.6, "eng")]),
    ]
    _, _, lyrics, _ = extract_vowel_boundaries(words, ["leng", "feng"], language="zh")
    assert lyrics == ["", "leng", "feng"]


