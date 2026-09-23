import pathlib
import sys
import traceback

import librosa
import numpy as np

# Add repo path if needed
ROOT_DIR = pathlib.Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.io.note_io import NoteInfo, pad_1d_arrays
from inference.game.alignment_utils import align_notes_to_words
from inference.game.onnx_runtime import GameOnnxModel


_SINGABLE_JA_PHONEMES = {"a", "i", "u", "e", "o"}
_SINGABLE_ZH_FALLBACK_VOWELS = set("aeiouv")
# ARPABET vowel inventory used by the English CMU dict (ds_cmudict-07b.txt).
_SINGABLE_EN_PHONEMES = {
    "aa", "ae", "ah", "ao", "aw", "ax", "ay",
    "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw",
}
_NON_SINGABLE_WORD_TOKENS = {"SP", "AP", "EP", "br", "sil", "pau"}


def _normalize_ts(ts) -> list[float]:
    if ts is None:
        return []
    if hasattr(ts, "tolist"):
        ts = ts.tolist()
    return [float(t) for t in ts]


def _resolve_game_language_id(game_model: GameOnnxModel, language: str | None = None) -> int:
    # The GAME model config carries a language id map (en/ja/yue/zh), but it is
    # intentionally NOT used: GAME runs language-neutral.
    del game_model, language
    return 0


def _normalize_phone_text(phone_text: str) -> str:
    return str(phone_text or "").split("/")[-1].strip()


def _is_singable_phone(phone_text: str, language: str | None) -> bool:
    phone_raw = _normalize_phone_text(phone_text)
    phone = phone_raw.lower()
    lang = (language or "").lower()

    if not phone or phone == "sp":
        return False
    if lang == "ja":
        return phone in _SINGABLE_JA_PHONEMES or phone_raw == "N"
    if lang == "en":
        # ARPABET: consonants like "v" must not be treated as vowels (the zh
        # fallback would match the letter "v" inside the phone name).
        return phone in _SINGABLE_EN_PHONEMES

    if phone in {"n", "ng", "m"}:
        return False
    return any(ch in _SINGABLE_ZH_FALLBACK_VOWELS for ch in phone)


def _find_word_nucleus_start(word, language: str | None) -> float | None:
    phonemes = getattr(word, "phonemes", None) or []
    for phoneme in phonemes:
        if _is_singable_phone(getattr(phoneme, "text", ""), language):
            return float(phoneme.start)
    return None


# Romaji vowel -> katakana row for kana lyric mode tail auto-fill.
_ROMAJI_VOWEL_TO_KANA = {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ", "N": "ン"}


# Romaji mora -> hiragana, borrowed from lfa_api (lazy import: JaG2p pulls
# pyopenjtalk, which game_api must not require at module import time).
_ROMAJI_MORA_KANA_CACHE: dict | None = None


def _get_romaji_mora_kana() -> dict:
    global _ROMAJI_MORA_KANA_CACHE
    if _ROMAJI_MORA_KANA_CACHE is None:
        try:
            from inference.API.lfa_api import _ROMAJI_TO_KANA_MORA as mapping

            _ROMAJI_MORA_KANA_CACHE = dict(mapping)
        except Exception:
            _ROMAJI_MORA_KANA_CACHE = {}
    return _ROMAJI_MORA_KANA_CACHE


def _word_matches_char(word_text: str, char: str, language: str | None) -> bool:
    """Whether an HFA word plausibly produced a lyric char (anti-cascade).

    Direct text equality covers hanzi/pinyin/kana-word text; for Japanese
    mora words in romaji ("ka") the lfa mora map is consulted so "ka"
    matches "か" (and vowel katakana "ア" via the tail-fill map).
    """
    wt, ch = str(word_text or ""), str(char or "")
    if not wt or not ch:
        return False
    if ch == wt or ch.lower() == wt.lower():
        return True
    if (language or "").lower() == "ja":
        kana = _get_romaji_mora_kana().get(wt.lower())
        if kana is not None and (kana == ch or kana.lower() == ch.lower()):
            return True
        if _ROMAJI_VOWEL_TO_KANA.get(wt.lower()) == ch:
            return True
    return False


def _looks_like_insertion(word, next_word, char: str, language: str | None) -> bool:
    """One-step lookahead: current word matches nothing, next word matches.

    An inserted HFA word (breath fragment, split mora) must not consume a
    lyric slot, or every later lyric in the chunk shifts by one.
    """
    if next_word is None or next_word.text in _NON_SINGABLE_WORD_TOKENS:
        return False
    if _find_word_nucleus_start(next_word, language) is None:
        return False
    return not _word_matches_char(word.text, char, language) and _word_matches_char(
        next_word.text, char, language
    )


def _word_nucleus_vowel(word, language: str | None, lyric_output_mode: str | None = None) -> str | None:
    """Nucleus vowel display text for melisma tail auto-fill, or None.

    Tail notes used to get '-'; with the nucleus vowel they sing the right
    sound in OpenUtau without manual filling. English returns None (ARPABET
    -> orthography mapping is unreliable; '+' markers already carry it).
    """
    lang = (language or "").lower()
    if lang == "en":
        return None
    nucleus = None
    for phoneme in getattr(word, "phonemes", None) or []:
        if _is_singable_phone(getattr(phoneme, "text", ""), language):
            nucleus = _normalize_phone_text(getattr(phoneme, "text", ""))
            break
    if not nucleus:
        return None
    if lang == "ja" and (lyric_output_mode or "").lower() == "kana":
        return _ROMAJI_VOWEL_TO_KANA.get(nucleus, nucleus)
    return nucleus


# Stops begin a new syllable chunk; other inter-vowel consonants (fricatives,
# nasals, liquids) close the previous chunk as its coda.
_EN_STOPS = {"p", "b", "t", "d", "k", "g", "dx", "jh", "ch"}


def _english_syllable_chunks(word):
    """Split one aligned English word into per-syllable time chunks.

    Each vowel nucleus owns one chunk. The boundary before a nucleus falls
    immediately before the last stop consonant (p/b/t/d/k/g/dx) of the
    consonant run that precedes it, so the stop begins the next chunk while
    fricatives and sonorants close the previous one as coda
    (impossible -> [ih m][p aa s ax][b ax l] = im/poss/ible). A run without a
    stop creates no boundary, which keeps a word-final "ax"+sonorant
    (syllabic consonant, e.g. -le/-en) inside the last chunk automatically.

    Returns [(start, end), ...] or None when the word has no singable vowel.
    """
    phones = getattr(word, "phonemes", None) or []
    if not phones:
        return None

    vowel_idx = [i for i, ph in enumerate(phones) if _is_singable_phone(getattr(ph, "text", ""), "en")]
    if not vowel_idx:
        return None

    bounds = [float(phones[0].start)]
    for k in range(1, len(vowel_idx)):
        run = phones[vowel_idx[k - 1] + 1 : vowel_idx[k]]
        stop_start = None
        for ph in run:
            # Lowercase: _is_singable_phone is case-insensitive, so the stop
            # check must be too (uppercase ARPABET would hide every stop).
            if _normalize_phone_text(ph.text).lower() in _EN_STOPS:
                stop_start = float(ph.start)
        if stop_start is not None:
            bounds.append(stop_start)
        else:
            bounds.append(float(phones[vowel_idx[k]].start))
    bounds.append(float(word.end))

    # A chunk made of a single lone vowel (no onset, no coda — e.g. the "i" of
    # -ible squeezed between two stops) has no note of its own; fold it into
    # the following chunk so it does not become a phantom syllable.
    merged = []
    spans = list(zip(bounds, bounds[1:]))
    i = 0
    while i < len(spans):
        start, end = spans[i]
        span_phones = [ph for ph in phones if start <= ph.start < end]
        if (
            len(span_phones) == 1
            and _is_singable_phone(getattr(span_phones[0], "text", ""), "en")
            and i + 1 < len(spans)
        ):
            merged.append((start, spans[i + 1][1]))
            i += 2
            continue
        merged.append((start, end))
        i += 1
    return merged


def _extract_vowel_boundaries_english(result_word, original_chars: list[str], lyric_output_mode: str | None = None):
    """Per-syllable chunk boundaries for English.

    Each chunk becomes one align unit for GAME; chunk 0 carries the whole word
    as its lyric and every later chunk carries "+" (the syllable-position
    marker). Melisma (转音, pitch-transition) notes inside a chunk fall back to the sustain symbol
    assigned by the caller. Vowels are always None (keep '-'; ARPABET does
    not map reliably to orthography).
    """
    word_durs = []
    word_vuvs = []
    lyrics = []
    vowels: list[str | None] = []

    char_idx = 0
    last_end = 0.0

    for word in result_word:
        if word.text in _NON_SINGABLE_WORD_TOKENS:
            if word.end > last_end:
                word_durs.append(word.end - last_end)
                word_vuvs.append(0)
                lyrics.append("")
                vowels.append(None)
                last_end = word.end
            continue

        lyric = word.text
        if char_idx < len(original_chars):
            while char_idx < len(original_chars) and original_chars[char_idx].lower() != word.text.lower():
                char_idx += 1
            if char_idx < len(original_chars):
                lyric = original_chars[char_idx]
                char_idx += 1

        chunks = _english_syllable_chunks(word)
        if chunks is None:
            if word.end > last_end:
                word_durs.append(word.end - last_end)
                word_vuvs.append(0)
                lyrics.append("")
                vowels.append(None)
                last_end = word.end
            continue

        if chunks[0][0] > last_end + 0.005:
            word_durs.append(chunks[0][0] - last_end)
            word_vuvs.append(0)
            lyrics.append("")
            vowels.append(None)
        elif chunks[0][0] < last_end:
            chunks[0] = (last_end, max(chunks[0][1], last_end + 0.001))

        for k, (chunk_start, chunk_end) in enumerate(chunks):
            if chunk_end <= chunk_start:
                continue
            word_durs.append(chunk_end - chunk_start)
            word_vuvs.append(1)
            lyrics.append(lyric if k == 0 else "+")
            vowels.append(None)
        last_end = chunks[-1][1]

    return word_durs, word_vuvs, lyrics, vowels


def load_game_model(model_dir: str, device=None):
    """
    Loads the GAME ONNX model suite.
    """
    print(f"Loading GAME ONNX model from '{model_dir}'...")
    try:
        model = GameOnnxModel(pathlib.Path(model_dir), requested_device=device)
    except Exception as e:
        raise RuntimeError(
            f"Error loading GAME ONNX model: {e}\n"
            "Please ensure the GAME ONNX model directory and its contents are correct."
        )

    print(f"GAME ONNX model loaded successfully with provider: {model.provider_name}.")
    return model


def extract_vowel_boundaries(
    result_word,
    original_chars: list[str],
    language: str | None = None,
    lyric_output_mode: str | None = None,
):
    if (language or "").lower() == "en":
        return _extract_vowel_boundaries_english(result_word, original_chars, lyric_output_mode)

    word_durs = []
    word_vuvs = []
    lyrics = []
    vowels: list[str | None] = []

    char_idx = 0
    last_end = 0.0

    ignore_tokens = _NON_SINGABLE_WORD_TOKENS
    is_romaji = len(original_chars) > 0 and all(c.isascii() or c == "" for c in original_chars)

    for i, word in enumerate(result_word):
        if word.text in ignore_tokens:
            if word.end > last_end:
                word_durs.append(word.end - last_end)
                word_vuvs.append(0)
                lyrics.append("")
                vowels.append(None)
                last_end = word.end
            continue

        vowel_start = _find_word_nucleus_start(word, language)
        if vowel_start is None:
            if word.end > last_end:
                word_durs.append(word.end - last_end)
                word_vuvs.append(0)
                lyrics.append("")
                vowels.append(None)
                last_end = word.end
            continue

        if (
            char_idx < len(original_chars)
            and _looks_like_insertion(
                word,
                result_word[i + 1] if i + 1 < len(result_word) else None,
                original_chars[char_idx],
                language,
            )
        ):
            # Inserted fragment: emit an unvoiced gap without consuming the char.
            if word.end > last_end:
                word_durs.append(word.end - last_end)
                word_vuvs.append(0)
                lyrics.append("")
                vowels.append(None)
                last_end = word.end
            continue

        if vowel_start > last_end + 0.005:
            word_durs.append(vowel_start - last_end)
            word_vuvs.append(0)
            lyrics.append("")
            vowels.append(None)
        elif vowel_start < last_end:
            vowel_start = last_end

        next_vowel_start = word.end
        if i + 1 < len(result_word):
            next_w = result_word[i + 1]
            if next_w.text not in ignore_tokens:
                next_nucleus_start = _find_word_nucleus_start(next_w, language)
                if next_nucleus_start is not None:
                    next_vowel_start = next_nucleus_start

        note_end = next_vowel_start
        if i + 1 < len(result_word) and result_word[i + 1].text in ignore_tokens:
            note_end = word.end

        dur = note_end - vowel_start
        if dur < 0:
            dur = 0.0
        if dur <= 0:
            # A zero-length voiced entry would be skipped by the note/word
            # aligner while the caller still consumes a lyric slot, shifting
            # every later lyric in the chunk by one. Drop it instead.
            continue

        if is_romaji:
            # Bounded seek: skip-ahead over deleted chars is preserved, but an
            # inserted word matching nothing ahead must not run char_idx off
            # the end (old code then emitted raw HFA text for the whole tail).
            seek = char_idx
            while seek < len(original_chars) and original_chars[seek].lower() != word.text.lower():
                seek += 1
            if seek >= len(original_chars):
                if word.end > last_end:
                    word_durs.append(word.end - last_end)
                    word_vuvs.append(0)
                    lyrics.append("")
                    vowels.append(None)
                    last_end = word.end
                continue

        word_durs.append(dur)
        word_vuvs.append(1)
        vowels.append(_word_nucleus_vowel(word, language, lyric_output_mode))

        if is_romaji:
            while char_idx < len(original_chars) and original_chars[char_idx].lower() != word.text.lower():
                char_idx += 1
            if char_idx < len(original_chars):
                lyrics.append(original_chars[char_idx])
                char_idx += 1
            else:
                lyrics.append(word.text)
        else:
            if char_idx < len(original_chars):
                lyrics.append(original_chars[char_idx])
                char_idx += 1
            else:
                lyrics.append(word.text)

        last_end = note_end

    return word_durs, word_vuvs, lyrics, vowels


def _run_game_inference_batch(
    *,
    game_model: GameOnnxModel,
    waveforms_np: list[np.ndarray],
    waveform_durations_np: list[float],
    known_durations_np: list[np.ndarray] | None,
    seg_threshold: float,
    seg_radius: float,
    est_threshold: float,
    ts,
    language: str | None = None,
):
    padded_wavs = pad_1d_arrays(waveforms_np).astype(np.float32, copy=False)
    padded_kd = None
    if known_durations_np is not None:
        padded_kd = pad_1d_arrays(known_durations_np, pad_value=0.0).astype(np.float32, copy=False)

    boundary_radius = int(round(seg_radius / game_model.timestep))
    return game_model.infer_batch(
        waveforms=padded_wavs,
        durations=np.asarray(waveform_durations_np, dtype=np.float32),
        known_durations=padded_kd,
        boundary_threshold=float(seg_threshold),
        boundary_radius=boundary_radius,
        score_threshold=float(est_threshold),
        language=_resolve_game_language_id(game_model, language),
        ts=_normalize_ts(ts),
    )


def extract_pitches_and_align(
    chunks,
    sr,
    pred_dict,
    chars_dict,
    game_model,
    ts,
    seg_threshold,
    seg_radius,
    est_threshold,
    batch_size=4,
    cancel_checker=None,
    language=None,
    lyric_output_mode=None,
):
    """
    Extract pitches using the GAME ONNX runtime and align them to lyrics.
    """
    # Melisma (转音, pitch-transition) tail notes get the unit's nucleus vowel
    # (zh/ja) so OpenUtau sings them without manual filling; English and
    # units without a nucleus keep '-' (see _word_nucleus_vowel).
    sustain_lyric = "-"
    print("[Hybrid Pipeline] Extracting pitches with GAME ONNX...")

    all_notes = []
    batch_infos = []
    processed_chunk_indices = set()

    for chunk_idx, chunk in enumerate(chunks):
        if cancel_checker and cancel_checker():
            raise InterruptedError("GAME task cancelled")
        stem = f"chunk_{chunk_idx}"
        if stem not in pred_dict:
            print(f"[Warning] {stem}: missing HFA prediction; skipping lyric-aligned GAME for this chunk.")
            continue

        _, _, result_word = pred_dict[stem]
        if not result_word:
            print(f"[Warning] {stem}: empty HFA word result; skipping lyric-aligned GAME for this chunk.")
            continue

        word_durs, word_vuvs, lyrics, vowels = extract_vowel_boundaries(
            result_word,
            chars_dict.get(stem, []),
            language=language,
            lyric_output_mode=lyric_output_mode,
        )
        if not word_durs:
            print(f"[Warning] {stem}: no usable word durations; skipping lyric-aligned GAME for this chunk.")
            continue

        batch_infos.append(
            {
                "chunk_idx": chunk_idx,
                "waveform": chunk["waveform"],
                "waveform_duration": len(chunk["waveform"]) / sr,
                "word_durs": word_durs,
                "known_durations": np.asarray(word_durs, dtype=np.float32),
                "offset": chunk["offset"],
                "word_vuvs": word_vuvs,
                "lyrics": lyrics,
                "vowels": vowels,
            }
        )

    for i in range(0, len(batch_infos), batch_size):
        if cancel_checker and cancel_checker():
            raise InterruptedError("GAME task cancelled")
        batch = batch_infos[i : i + batch_size]

        waveforms_np = [info["waveform"] for info in batch]
        waveform_durations_np = [info["waveform_duration"] for info in batch]
        known_durations_np = [info["known_durations"] for info in batch]

        try:
            batch_results = _run_game_inference_batch(
                game_model=game_model,
                waveforms_np=waveforms_np,
                waveform_durations_np=waveform_durations_np,
                known_durations_np=known_durations_np,
                seg_threshold=seg_threshold,
                seg_radius=seg_radius,
                est_threshold=est_threshold,
                ts=ts,
                language=language,
            )
        except Exception:
            print("Error during GAME ONNX inference batch:")
            traceback.print_exc()
            raise

        for result, info in zip(batch_results, batch):
            durations, presence, scores = result

            note_dur = durations[durations > 0].tolist()
            valid_presence = presence[durations > 0]
            valid_scores = scores[durations > 0]
            if not note_dur:
                print(f"[Warning] GAME returned no note durations for chunk at {info['offset']:.2f}s; skipping.")
                continue

            note_seq = [
                librosa.midi_to_note(float(m), unicode=False, cents=True) if v else "rest"
                for m, v in zip(valid_scores, valid_presence)
            ]

            a_note_seq, a_note_dur, a_note_slur = align_notes_to_words(
                info["word_durs"],
                info["word_vuvs"],
                note_seq,
                note_dur,
                apply_word_uv=True,
                assign_by_onset=(language or "").lower() == "en",
            )

            lyric_idx = 0
            current_onset = info["offset"]
            pending_lyric = ""
            current_vowel = ""
            unit_vowels = info.get("vowels") or []

            for n_seq, n_dur, n_slur in zip(a_note_seq, a_note_dur, a_note_slur):
                if n_slur == 0:
                    if lyric_idx < len(info["lyrics"]):
                        word_lyric = info["lyrics"][lyric_idx]
                        if info["word_vuvs"][lyric_idx] == 1:
                            pending_lyric = word_lyric
                            current_vowel = (
                                unit_vowels[lyric_idx] or ""
                                if lyric_idx < len(unit_vowels)
                                else ""
                            )
                        else:
                            pending_lyric = ""
                            current_vowel = ""
                        lyric_idx += 1

                if n_seq != "rest":
                    pitch = librosa.note_to_midi(n_seq, round_midi=False)
                    if pending_lyric:
                        lyric_to_assign = pending_lyric
                        pending_lyric = ""
                        is_continuation = False
                    else:
                        lyric_to_assign = current_vowel or sustain_lyric
                        is_continuation = True

                    is_contiguous = len(all_notes) > 0 and abs(all_notes[-1].offset - current_onset) < 0.01
                    can_merge = (
                        is_continuation
                        and is_contiguous
                        and abs(all_notes[-1].pitch - pitch) < 0.1
                        and all_notes[-1].lyric == lyric_to_assign
                    )

                    if can_merge:
                        all_notes[-1].offset += n_dur
                    else:
                        all_notes.append(
                            NoteInfo(
                                onset=current_onset,
                                offset=current_onset + n_dur,
                                pitch=pitch,
                                lyric=lyric_to_assign,
                            )
                        )

                current_onset += n_dur
            processed_chunk_indices.add(info["chunk_idx"])

    return all_notes, processed_chunk_indices


def extract_pitches_only(
    chunks,
    sr,
    game_model,
    ts,
    seg_threshold,
    seg_radius,
    est_threshold,
    batch_size=4,
    cancel_checker=None,
    language=None,
):
    """
    Extract pitches using the GAME ONNX runtime without lyric alignment.
    """
    print("[Hybrid Pipeline] Extracting pitches with GAME ONNX (no-lyrics mode)...")

    all_notes = []
    batch_infos = []
    for chunk in chunks:
        if cancel_checker and cancel_checker():
            raise InterruptedError("GAME task cancelled")
        batch_infos.append(
            {
                "waveform": chunk["waveform"],
                "waveform_duration": len(chunk["waveform"]) / sr,
                "offset": chunk["offset"],
            }
        )

    for i in range(0, len(batch_infos), batch_size):
        if cancel_checker and cancel_checker():
            raise InterruptedError("GAME task cancelled")
        batch = batch_infos[i : i + batch_size]

        waveforms_np = [info["waveform"] for info in batch]
        waveform_durations_np = [info["waveform_duration"] for info in batch]
        known_durations_np = [np.zeros(1, dtype=np.float32) for _ in batch]

        try:
            batch_results = _run_game_inference_batch(
                game_model=game_model,
                waveforms_np=waveforms_np,
                waveform_durations_np=waveform_durations_np,
                known_durations_np=known_durations_np,
                seg_threshold=seg_threshold,
                seg_radius=seg_radius,
                est_threshold=est_threshold,
                ts=ts,
                language=language,
            )
        except Exception:
            print("Error during GAME ONNX inference batch (no-lyrics):")
            traceback.print_exc()
            raise

        for result, info in zip(batch_results, batch):
            durations, presence, scores = result
            valid = durations > 0
            note_dur = durations[valid].tolist()
            note_presence = presence[valid].tolist()
            note_scores = scores[valid].tolist()
            if not note_dur:
                print(f"[Warning] GAME returned no note durations for chunk at {info['offset']:.2f}s; skipping.")
                continue

            current_onset = info["offset"]
            for n_dur, n_presence, n_score in zip(note_dur, note_presence, note_scores):
                if n_presence:
                    all_notes.append(
                        NoteInfo(
                            onset=current_onset,
                            offset=current_onset + n_dur,
                            pitch=float(n_score),
                            lyric="",
                        )
                    )
                current_onset += n_dur

    return all_notes
