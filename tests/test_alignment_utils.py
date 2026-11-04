import numpy as np

from inference.game.alignment_utils import align_notes_to_words


def test_note_inside_unvoiced_word_is_kept():
    words = [0.2, 0.4, 0.2]
    vuv = [1, 0, 1]
    notes = ["C4", "E4", "G4"]
    durs = [0.2, 0.2, 0.2]

    seq, dur, slur = align_notes_to_words(words, vuv, notes, durs, apply_word_uv=True)

    assert seq == ["C4", "E4", "G4", "rest"]
    assert np.allclose(dur, [0.2, 0.2, 0.2, 0.2])
    assert slur == [0, 0, 1, 0]
    assert abs(sum(dur) - sum(words)) < 1e-6


def test_unvoiced_word_without_notes_still_rests():
    words = [0.2, 0.3, 0.2]
    vuv = [1, 0, 1]
    notes = ["C4", "rest", "G4"]
    durs = [0.2, 0.3, 0.2]

    seq, dur, slur = align_notes_to_words(words, vuv, notes, durs, apply_word_uv=True)

    assert seq == ["C4", "rest", "G4"]
    assert np.allclose(dur, [0.2, 0.3, 0.2])
    assert slur == [0, 0, 0]


def test_one_lyric_slot_per_word():
    # The lyric loop consumes a word for every slur==0 unit; splitting a rest
    # around an interior note must not add or remove slots.
    words = [0.2, 0.4, 0.2]
    vuv = [1, 0, 1]
    notes = ["C4", "E4", "G4"]
    durs = [0.2, 0.2, 0.2]

    _, dur, slur = align_notes_to_words(words, vuv, notes, durs, apply_word_uv=True)

    assert sum(1 for flag in slur if flag == 0) == len(words)
    assert slur[0] == 0


def test_voiced_words_unchanged():
    words = [0.2, 0.2]
    vuv = [1, 1]
    notes = ["C4", "E4"]
    durs = [0.2, 0.2]

    seq, dur, slur = align_notes_to_words(words, vuv, notes, durs, apply_word_uv=True)

    assert seq == ["C4", "E4"]
    assert np.allclose(dur, [0.2, 0.2])
    assert slur == [0, 0]
