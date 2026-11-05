import pathlib
import sys
import traceback

import librosa
import numpy as np

ROOT_DIR = pathlib.Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.io.note_io import NoteInfo, pad_1d_arrays
from inference.game.alignment_utils import align_notes_to_words
from inference.game.onnx_runtime import GameOnnxModel
import logging

logger = logging.getLogger(__name__)


_SINGABLE_JA_PHONEMES = {"a", "i", "u", "e", "o"}
_SINGABLE_ZH_FALLBACK_VOWELS = set("aeiouv")
# ARPABET vowels in the English CMU dict (ds_cmudict-07b.txt).
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
    if (language or "").lower() == "yue":
        # Syllabic nasals in Cantonese (m, ng) act as nucleus when no other vowel is present
        for phoneme in phonemes:
            p_text = _normalize_phone_text(getattr(phoneme, "text", "")).lower()
            if p_text in {"m", "ng"}:
                return float(phoneme.start)
    return None


# Romaji vowel -> kana for kana lyric mode tail auto-fill; hiragana so tails
# match the kana-mode lyrics produced by the ASR/TiFA display path.
_ROMAJI_VOWEL_TO_KANA = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お", "N": "ん"}


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


def _english_syllable_chunks(word):
    """Split one aligned English word into per-syllable time chunks.

    Each chunk starts at its vowel nucleus while an inter-syllable onset stays
    in the preceding chunk, so word-initial consonants are sung inside the
    previous note. Returns [(start, end), ...] or None when no singable vowel.
    """
    phones = getattr(word, "phonemes", None) or []
    if not phones:
        return None

    vowel_idx = [i for i, ph in enumerate(phones) if _is_singable_phone(getattr(ph, "text", ""), "en")]
    if not vowel_idx:
        return None

    bounds = [float(phones[vowel_idx[0]].start)]
    for k in range(1, len(vowel_idx)):
        bounds.append(float(phones[vowel_idx[k]].start))
    bounds.append(float(word.end))
    return list(zip(bounds, bounds[1:]))


def _extract_vowel_boundaries_english(result_word, original_chars: list[str], lyric_output_mode: str | None = None):
    """Per-syllable chunk boundaries for English.

    Each chunk becomes one align unit for GAME; chunk 0 carries the whole word
    as its lyric and every later chunk carries "+" (the syllable-position
    marker). Melisma notes inside a chunk use the caller's sustain symbol.
    Vowels are always None (keep '-'; ARPABET does not map reliably to
    orthography).
    """
    word_durs = []
    word_vuvs = []
    lyrics = []
    vowels: list = []

    char_idx = 0
    last_end = 0.0

    for i, word in enumerate(result_word):
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

        # The final chunk extends to the next singable word's vowel nucleus, so
        # that word's onset consonants are sung inside this note (no vuvs=0 hole
        # per word boundary); after SP/AP the caller's gap entry applies.
        note_end = float(word.end)
        if i + 1 < len(result_word) and result_word[i + 1].text not in _NON_SINGABLE_WORD_TOKENS:
            next_nucleus = _find_word_nucleus_start(result_word[i + 1], "en")
            if next_nucleus is not None:
                note_end = max(note_end, float(next_nucleus))
        chunks[-1] = (chunks[-1][0], note_end)

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


_GAME_MODEL_CACHE: dict[tuple[str, str], GameOnnxModel] = {}


def load_game_model(model_dir: str, device=None, use_cache: bool = True):
    """Load the GAME ONNX model suite.

    Cached in-process: recreating the DirectML session per song corrupts
    onnxruntime's DML state and access-violates the process on the next song.
    """
    resolved = str(pathlib.Path(model_dir))
    key = (resolved, str(device or "").strip().lower() or "auto")
    if use_cache:
        cached = _GAME_MODEL_CACHE.get(key)
        if cached is not None:
            logger.info(f"Reusing cached GAME model from '{resolved}'.")
            return cached

    logger.info(f"Loading GAME ONNX model from '{model_dir}'...")
    try:
        model = GameOnnxModel(pathlib.Path(model_dir), requested_device=device)
    except Exception as e:
        raise RuntimeError(
            f"Error loading GAME ONNX model: {e}\n"
            "Please ensure the GAME ONNX model directory and its contents are correct."
        )

    logger.info(f"GAME ONNX model loaded successfully with provider: {model.provider_name}.")
    if use_cache:
        _GAME_MODEL_CACHE[key] = model
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
    vowels: list = []

    char_idx = 0
    last_end = 0.0

    ignore_tokens = _NON_SINGABLE_WORD_TOKENS
    is_romaji = (
        (language or "").lower() == "ja"
        and len(original_chars) > 0
        and all(c.isascii() or c == "" for c in original_chars)
        and all(w.text.isascii() for w in result_word if w.text and w.text not in ignore_tokens)
    )

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
            # A consonant-only token (ja 'cl' = っ) has a display char but no
            # note of its own. Not consuming that char hands っ to the next
            # voiced word and shifts every later lyric in the chunk by one.
            if char_idx < len(original_chars):
                char_idx += 1
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
    try:
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
    except UnicodeDecodeError as exc:
        # onnxruntime formats the DirectML failure message with the system
        # ANSI code page; decoding it as UTF-8 fails and hides the real error,
        # which on this path is almost always a driver timeout from low VRAM.
        raise RuntimeError(
            f"GAME ONNX inference failed on {game_model.provider_name} with an "
            "undecodable driver message (the system code page is not UTF-8). "
            "This usually means the GPU ran out of memory or its driver timed "
            f"out; retry with batch size 1 (requested {len(waveforms_np)}, "
            f"padded to {padded_wavs.shape[1] / game_model.samplerate:.1f}s each)."
        ) from exc


# Melisma tails are filled with the unit's nucleus vowel as kana
# (_word_nucleus_vowel), so a run of notes carrying the same vowel-only kana can
# only be one sung mora. GAME splits a sustained vowel whose pitch wobbles into
# short alternating-pitch fragments, which the same-pitch merge in the chunk
# loop cannot join.
_VOWEL_ONLY_KANA = frozenset("あいうえおアイウエオ")
_KANA_ROWS = {
    "a": "あかさたなはまやらわがざだばぱぁゃ",
    "i": "いきしちにひみりゐぎじぢびぴぃ",
    "u": "うくすつぬふむゆるゔぐずづぶぷぅゅ",
    "e": "えけせてねへめれゑげぜでべぺぇ",
    "o": "おこそとのほもよろをごぞどぼぽをぉょ",
}
_KANA_VOWEL = {kana: vowel for vowel, kana in _KANA_ROWS.items() for kana in kana}
_KANA_VOWEL["ん"] = "N"

VIBRATO_MIN_FRAGMENTS = 3
VIBRATO_MAX_FRAGMENT_SEC = 0.25
VIBRATO_MIN_SPAN_SEC = 0.30
VIBRATO_JOIN_SEC = 0.005
VIBRATO_PHRASE_GAP_SEC = 0.60
# A melisma tail continues the previous note without a pause. A lyric-less
# fragment that starts later than this after the previous note sits on the
# next syllable's onset, so repeating the old vowel there prepends it to the
# next lyric instead of continuing the previous one.
CONTINUATION_MAX_GAP_SEC = 0.05
# GAME can split one sustained vowel into pieces whose pitch estimates differ
# by a few cents; merging only exact matches left an a-a-a stutter.
CONTINUATION_MERGE_PITCH_SEMITONES = 0.5
# A lyric-less fragment that ends on the next note and already carries the
# next syllable's pitch is that syllable's onset leaking backwards, not a
# continuation of the previous vowel.
NEXT_ONSET_ANTICIPATION_SEC = 0.02
VIBRATO_MAX_CENTRE_OFFSET = 1.0
VIBRATO_MIN_REVERSALS = 2


def _continuation_lyric(prev_lyric: str, language: str | None, lyric_output_mode: str | None) -> str | None:
    """Singable lyric for a note continuing the previous syllable, or None.

    A note GAME places inside a word HFA marked unvoiced has no lyric of its
    own; repeating the previous syllable's vowel keeps it singable instead of
    exporting a bare '+' continuation.
    """
    lang = (language or "").lower()
    if lang == "en":
        return None
    vowel = _lyric_vowel(prev_lyric)
    if not vowel:
        return None
    if (lyric_output_mode or "").lower() == "kana":
        return _ROMAJI_VOWEL_TO_KANA.get(vowel, vowel)
    return "n" if vowel == "N" else vowel.lower()



def _lyric_vowel(lyric: str) -> str | None:
    """Nucleus vowel of a ja lyric written as kana, katakana or romaji."""
    text = str(lyric or "").strip()
    if not text:
        return None
    ch = text[-1]
    if 0x30A1 <= ord(ch) <= 0x30F6:
        ch = chr(ord(ch) - 0x60)
    if ch in _KANA_VOWEL:
        return _KANA_VOWEL[ch]
    for c in reversed(text.lower()):
        if c in "aiueo":
            return c
        if c == "n":
            return "N"
    return None


def _direction_reversals(pitches: list[float], reference: float) -> int:
    """How often the fragment pitch crosses the reference pitch.

    A vibrato bounces across the note it belongs to several times; a melisma
    that steps to another pitch keeps one direction or stops.
    """
    signs = []
    for pitch in pitches:
        delta = pitch - reference
        if abs(delta) < 0.25:
            continue
        signs.append(1 if delta > 0 else -1)
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def collapse_vibrato_tails(notes: list[NoteInfo]) -> list[str]:
    """Absorb phrase-final vibrato fragments into the note they belong to.

    Every gate must hold or the notes are left untouched: the run carries a
    vowel-only kana lyric matching the previous note's vowel, it is the last
    phrase of the song, its fragments are short and contiguous, and their
    pitch alternates around the previous note's pitch instead of stepping to a
    new one. Absorbing only lengthens the previous note, so no pitch is
    invented; the exported pitch curve keeps the vibrato.
    """
    ordered = sorted(notes, key=lambda n: (n.onset, n.offset))
    logs: list[str] = []
    absorbed: list[NoteInfo] = []
    total = len(ordered)

    i = 0
    while i < total:
        j = i + 1
        while (
            j < total
            and ordered[j].lyric == ordered[i].lyric
            and ordered[j].onset - ordered[j - 1].offset <= VIBRATO_JOIN_SEC
        ):
            j += 1
        start, i = i, j
        run = ordered[start:j]

        if len(run) < VIBRATO_MIN_FRAGMENTS or run[0].lyric not in _VOWEL_ONLY_KANA:
            continue
        if any(n.offset - n.onset > VIBRATO_MAX_FRAGMENT_SEC for n in run):
            continue
        if run[-1].offset - run[0].onset < VIBRATO_MIN_SPAN_SEC:
            continue
        prev = ordered[start - 1] if start > 0 else None
        if prev is None or run[0].onset - prev.offset > VIBRATO_JOIN_SEC:
            continue
        if _lyric_vowel(prev.lyric) != _lyric_vowel(run[0].lyric):
            continue
        if j < total and ordered[j].onset - run[-1].offset < VIBRATO_PHRASE_GAP_SEC:
            continue

        pitches = [n.pitch for n in run]
        centre = float(np.median(pitches))
        reversals = _direction_reversals(pitches, prev.pitch)
        if abs(centre - prev.pitch) > VIBRATO_MAX_CENTRE_OFFSET or reversals < VIBRATO_MIN_REVERSALS:
            logs.append(
                f"[GAME] vibrato tail kept @ {run[0].onset:.2f}-{run[-1].offset:.2f}s: "
                f"{len(run)} fragments '{run[0].lyric}' centre {centre:.1f} vs prev "
                f"{prev.pitch:.1f} ({prev.lyric}), reversals {reversals}"
            )
            continue

        prev.offset = run[-1].offset
        absorbed.extend(run)
        logs.append(
            f"[GAME] vibrato tail absorbed @ {run[0].onset:.2f}-{run[-1].offset:.2f}s: "
            f"{len(run)} fragments '{run[0].lyric}' -> 1 note '{prev.lyric}' "
            f"(centre {centre:.1f}, prev {prev.pitch:.1f}, reversals {reversals})"
        )

    if absorbed:
        dropped = {id(n) for n in absorbed}
        notes[:] = [n for n in notes if id(n) not in dropped]
    return logs


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
    """Extract pitches using the GAME ONNX runtime and align them to lyrics."""
    # Melisma (pitch-transition) tail notes get the unit's nucleus vowel
    # (zh/ja) so OpenUtau sings them without manual filling; English and
    # units without a nucleus keep '-' (see _word_nucleus_vowel).
    sustain_lyric = "-"
    logger.info("[Hybrid Pipeline] Extracting pitches with GAME ONNX...")

    all_notes = []
    batch_infos = []
    processed_chunk_indices = set()

    for chunk_idx, chunk in enumerate(chunks):
        if cancel_checker and cancel_checker():
            raise InterruptedError("GAME task cancelled")
        stem = f"chunk_{chunk_idx}"
        if stem not in pred_dict:
            logger.warning(f"[Warning] {stem}: missing HFA prediction; skipping lyric-aligned GAME for this chunk.")
            continue

        _, _, result_word = pred_dict[stem]
        if not result_word:
            logger.warning(f"[Warning] {stem}: empty HFA word result; skipping lyric-aligned GAME for this chunk.")
            continue

        word_durs, word_vuvs, lyrics, vowels = extract_vowel_boundaries(
            result_word,
            chars_dict.get(stem, []),
            language=language,
            lyric_output_mode=lyric_output_mode,
        )
        if not word_durs:
            logger.warning(f"[Warning] {stem}: no usable word durations; skipping lyric-aligned GAME for this chunk.")
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
            logger.error("Error during GAME ONNX inference batch:", exc_info=True)
            raise

        for result, info in zip(batch_results, batch):
            durations, presence, scores = result

            note_dur = durations[durations > 0].tolist()
            valid_presence = presence[durations > 0]
            valid_scores = scores[durations > 0]
            if not note_dur:
                logger.warning(f"[Warning] GAME returned no note durations for chunk at {info['offset']:.2f}s; skipping.")
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
                assign_by_onset=False,
            )

            lyric_idx = 0
            current_onset = info["offset"]
            pending_lyric = ""
            current_vowel = ""
            unit_unvoiced = False
            unit_vowels = info.get("vowels") or []

            note_onsets = np.cumsum([0.0] + list(a_note_dur))[:-1] + info["offset"]
            next_notes = [None] * len(a_note_seq)
            upcoming = None
            for i in range(len(a_note_seq) - 1, -1, -1):
                next_notes[i] = upcoming
                if a_note_seq[i] != "rest":
                    upcoming = (librosa.note_to_midi(a_note_seq[i], round_midi=False), note_onsets[i])

            for i, (n_seq, n_dur, n_slur) in enumerate(zip(a_note_seq, a_note_dur, a_note_slur)):
                if n_slur == 0:
                    if lyric_idx < len(info["lyrics"]):
                        word_lyric = info["lyrics"][lyric_idx]
                        unit_unvoiced = info["word_vuvs"][lyric_idx] == 0
                        if not unit_unvoiced:
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
                        is_continuation = True
                        prev_note = all_notes[-1] if all_notes else None
                        if (
                            prev_note is None
                            or current_onset - prev_note.offset > CONTINUATION_MAX_GAP_SEC
                        ):
                            # No syllable and no touch with the previous note:
                            # a GAME false positive on the next syllable's
                            # onset. Singing the old vowel here would paste
                            # the previous syllable's tail in front of the
                            # next lyric, so drop the fragment.
                            current_onset += n_dur
                            continue
                        if unit_unvoiced:
                            nxt = next_notes[i]
                            if (
                                nxt is not None
                                and nxt[1] - (current_onset + n_dur) <= NEXT_ONSET_ANTICIPATION_SEC
                                and abs(pitch - nxt[0]) < abs(pitch - prev_note.pitch)
                            ):
                                # The fragment sits on a consonant-only unit and
                                # already carries the next syllable's pitch: it
                                # is that syllable leaking backwards, not a
                                # continuation. Keeping it sang a low filler
                                # vowel through the sokuon closure.
                                current_onset += n_dur
                                continue
                        lyric_to_assign = current_vowel or (
                            _continuation_lyric(prev_note.lyric, language, lyric_output_mode)
                            or sustain_lyric
                        )

                    is_contiguous = len(all_notes) > 0 and abs(all_notes[-1].offset - current_onset) < 0.01
                    can_merge = (
                        is_continuation
                        and is_contiguous
                        and abs(all_notes[-1].pitch - pitch) < CONTINUATION_MERGE_PITCH_SEMITONES
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
    """Extract pitches using the GAME ONNX runtime without lyric alignment."""
    logger.info("[Hybrid Pipeline] Extracting pitches with GAME ONNX (no-lyrics mode)...")

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
            logger.error("Error during GAME ONNX inference batch (no-lyrics):", exc_info=True)
            raise

        for result, info in zip(batch_results, batch):
            durations, presence, scores = result
            valid = durations > 0
            note_dur = durations[valid].tolist()
            note_presence = presence[valid].tolist()
            note_scores = scores[valid].tolist()
            if not note_dur:
                logger.warning(f"[Warning] GAME returned no note durations for chunk at {info['offset']:.2f}s; skipping.")
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
