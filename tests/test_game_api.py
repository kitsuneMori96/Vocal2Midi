import numpy as np
import pytest

from inference.API.game_api import extract_pitches_only, extract_vowel_boundaries
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

    word_durs, word_vuvs, lyrics, vowels = extract_vowel_boundaries(words, ["か", "に"], language="ja")

    assert word_durs == pytest.approx([0.08, 0.25, 0.17])
    assert word_vuvs == [0, 1, 1]
    assert lyrics == ["", "か", "に"]
    assert vowels == [None, "a", "i"]


def test_extract_vowel_boundaries_zh_does_not_use_coda_as_vowel_start():
    words = [
        _make_word(0.0, 0.30, "ang", [(0.0, 0.22, "a"), (0.22, 0.30, "ng")]),
        _make_word(0.30, 0.60, "ni", [(0.30, 0.36, "n"), (0.36, 0.60, "i")]),
    ]

    word_durs, word_vuvs, lyrics, vowels = extract_vowel_boundaries(words, ["昂", "你"], language="zh")

    assert word_durs == pytest.approx([0.36, 0.24])
    assert word_vuvs == [1, 1]
    assert lyrics == ["昂", "你"]
    assert vowels == ["a", "i"]


def test_extract_vowel_boundaries_en_uses_arpabet_vowels():
    words = [
        _make_word(0.0, 0.40, "fall", [(0.0, 0.05, "f"), (0.05, 0.30, "ao"), (0.30, 0.40, "l")]),
        _make_word(0.40, 0.70, "fly", [(0.40, 0.45, "f"), (0.45, 0.50, "l"), (0.50, 0.70, "ay")]),
    ]

    word_durs, word_vuvs, lyrics, vowels = extract_vowel_boundaries(
        words, ["fall", "fly"], language="en"
    )

    # English emits one syllable chunk per word (fall -> f+ao+l is one chunk,
    # fly -> f+l+ay likewise), each carrying the word as its lyric.
    assert word_durs == pytest.approx([0.40, 0.30])
    assert word_vuvs == [1, 1]
    assert lyrics == ["fall", "fly"]
    assert vowels == [None, None]


def test_en_singable_phones_reject_consonant_v_names():
    from inference.API.game_api import _is_singable_phone

    assert _is_singable_phone("en/ao", "en")
    assert _is_singable_phone("ay", "en")
    assert not _is_singable_phone("en/v", "en")
    assert not _is_singable_phone("en/_r", "en")


def test_tail_vowel_kana_mode_maps_to_katakana_row():
    words = [
        _make_word(0.0, 0.25, "ka", [(0.0, 0.08, "k"), (0.08, 0.25, "a")]),
    ]

    _, _, _, vowels = extract_vowel_boundaries(
        words, ["か"], language="ja", lyric_output_mode="kana"
    )

    assert vowels == [None, "ア"]


def test_melisma_tail_note_gets_nucleus_vowel_not_dash():
    import pathlib

    from inference.API.game_api import extract_pitches_and_align

    class _TwoNoteModel:
        timestep = 0.01

        def infer_batch(self, **kwargs):
            return [
                (
                    np.array([0.5, 0.5], dtype=np.float32),
                    np.array([1, 1], dtype=np.int64),
                    np.array([60.0, 62.0], dtype=np.float32),
                )
            ]

    word = _make_word(0.0, 1.0, "ni", [(0.0, 0.10, "n"), (0.10, 1.0, "i")])
    pred_dict = {"chunk_0": (pathlib.Path("chunk_0.wav"), 1.0, [word])}
    chunks = [{"waveform": np.zeros(16000, dtype=np.float32), "offset": 0.0}]

    all_notes, processed = extract_pitches_and_align(
        chunks,
        16000,
        pred_dict,
        {"chunk_0": ["你"]},
        _TwoNoteModel(),
        [0.0],
        0.2,
        0.02,
        0.2,
        language="zh",
    )

    assert processed == {0}
    assert [n.lyric for n in all_notes] == ["你", "i"]
    assert "-" not in [n.lyric for n in all_notes]


def test_melisma_tail_note_keeps_dash_for_english():
    import pathlib

    from inference.API.game_api import extract_pitches_and_align

    class _TwoNoteModel:
        timestep = 0.01

        def infer_batch(self, **kwargs):
            return [
                (
                    np.array([0.5, 0.5], dtype=np.float32),
                    np.array([1, 1], dtype=np.int64),
                    np.array([60.0, 62.0], dtype=np.float32),
                )
            ]

    word = _make_word(
        0.0, 1.0, "fly", [(0.0, 0.05, "f"), (0.05, 0.10, "l"), (0.10, 1.0, "ay")]
    )
    pred_dict = {"chunk_0": (pathlib.Path("chunk_0.wav"), 1.0, [word])}
    chunks = [{"waveform": np.zeros(16000, dtype=np.float32), "offset": 0.0}]

    all_notes, _ = extract_pitches_and_align(
        chunks,
        16000,
        pred_dict,
        {"chunk_0": ["fly"]},
        _TwoNoteModel(),
        [0.0],
        0.2,
        0.02,
        0.2,
        language="en",
    )

    assert [n.lyric for n in all_notes] == ["fly", "-"]
