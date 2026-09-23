"""Velocity (dynamics) curve extraction for Vocal2Midi.

Locked plan: phoneme-guided HMM smoothing
  extract_rms_obs (librosa) -> build_emission_probs (self) ->
  build_transition_matrix (self) -> librosa.sequence.viterbi (reuse) ->
  state_to_velocity + pchip/savgol (scipy) -> {note_velocities, dyn curve}

Interface notes (verified against repo):
- NoteInfo(onset, offset, pitch, lyric="") : seconds, seconds, midi, str
- pred_dict[stem] = (wav_path: Path, wav_length: float, words: WordList)
  Word(start, end, text, phonemes); Phoneme(start, end, text); all SECONDS,
  chunk-LOCAL time. Global time = chunks[idx]["offset"] + local.
- chunks[i] = {"offset": float(sec), "waveform": np.ndarray}
- USTX: UCurveInterval=5 ticks, _to_ticks(s,tempo)=round(s*tempo*8),
  edge_trim=min(0.025, dur*0.15).
- librosa.sequence.viterbi(prob, transition): prob=(n_states, n_frames)
  raw probs in [0,1]; internally takes log. Never pre-log; clip eps=1e-10.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

FRAME_LENGTH = 2048
HOP_LENGTH = 512
EMISSION_EPS = 1e-10

# 6 dynamics states pp/p/mp/mf/f/ff; Gaussian membership centers + velocity map.
N_VELOCITY_STATES = 6
STATE_CENTERS = np.array([0.08, 0.25, 0.42, 0.58, 0.75, 0.92], dtype=np.float64)
STATE_SIGMA = 0.12
STATE_VELOCITIES = np.array([15, 40, 60, 80, 100, 118], dtype=np.int32)
# Unvoiced/consonant frames are clamped to this observation value so the
# emission favors low states without hard-zeroing any column.
UNVOICED_OBS = 0.1


@dataclass
class VelocityObs:
    """Frame-level loudness observation (1a output)."""

    norm_obs: np.ndarray  # [T] float32 in [0,1], p10-p95 normalized
    db: np.ndarray  # [T] float32 dB (<=0, ref=max)
    rms: np.ndarray  # [T] float32 linear RMS
    frame_times: np.ndarray  # [T] float64 seconds, = idx*hop/sr (NOT 10ms assumption)
    sample_rate: int
    hop_length: int


def extract_rms_obs(
    waveform: np.ndarray,
    sr: int,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> VelocityObs:
    """Compute per-frame normalized loudness observation.

    Steps: mono-mix -> librosa.feature.rms -> amplitude_to_dB(ref=max)
    -> per-utterance p10-p95 normalize to [0,1].
    Degenerate (constant/silent) input -> 0.5 mid to avoid div-by-zero.
    """
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    if wav.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return VelocityObs(
            norm_obs=empty,
            db=empty,
            rms=empty,
            frame_times=np.zeros((0,), dtype=np.float64),
            sample_rate=int(sr),
            hop_length=int(hop_length),
        )

    rms = librosa.feature.rms(y=wav, frame_length=int(frame_length), hop_length=int(hop_length))[0].astype(
        np.float32
    )
    # ref=max -> db <= 0; top_db=None keeps full range for p10-p95 below.
    db = librosa.amplitude_to_db(rms, ref=np.max).astype(np.float32)
    # Silence frames (rms==0) give -inf; replace with floor before percentile.
    finite = db[np.isfinite(db)]
    if finite.size == 0:
        norm = np.full_like(rms, 0.5, dtype=np.float32)
    else:
        floor = float(finite.min())
        db_safe = np.where(np.isfinite(db), db, floor).astype(np.float32)
        p10 = float(np.percentile(db_safe, 10))
        p95 = float(np.percentile(db_safe, 95))
        span = p95 - p10
        if span < 1e-6:
            norm = np.full_like(db_safe, 0.5, dtype=np.float32)
        else:
            norm = np.clip((db_safe - p10) / span, 0.0, 1.0).astype(np.float32)
        db = db_safe

    frame_times = (np.arange(rms.shape[0], dtype=np.float64) * float(hop_length) / float(sr)).astype(np.float64)
    return VelocityObs(
        norm_obs=norm,
        db=db,
        rms=rms,
        frame_times=frame_times,
        sample_rate=int(sr),
        hop_length=int(hop_length),
    )


def build_emission_probs(
    norm_obs: np.ndarray,
    voiced_mask: np.ndarray | None = None,
    eps: float = EMISSION_EPS,
) -> np.ndarray:
    """Map frame observations to per-state emission probs (6, T).

    voiced frames keep norm_obs; unvoiced (consonant/SP/AP) frames are
    clamped to UNVOICED_OBS=0.1 so low states dominate there. Gaussian
    membership -> column-normalize -> clip(eps,1). Raw probs (no log):
    librosa.sequence.viterbi takes log internally.
    """
    obs = np.asarray(norm_obs, dtype=np.float64).reshape(-1)
    n_frames = int(obs.shape[0])
    if n_frames == 0:
        return np.zeros((N_VELOCITY_STATES, 0), dtype=np.float64)

    if voiced_mask is None:
        voiced = np.ones(n_frames, dtype=bool)
    else:
        voiced = np.asarray(voiced_mask).reshape(-1).astype(bool)
        if voiced.shape[0] != n_frames:
            raise ValueError(f"voiced_mask len {voiced.shape[0]} != obs len {n_frames}")

    effective = np.where(voiced, np.clip(obs, 0.0, 1.0), UNVOICED_OBS).astype(np.float64)
    diff = (effective[None, :] - STATE_CENTERS[:, None]) / STATE_SIGMA
    prob = np.exp(-0.5 * diff * diff)
    col_sum = prob.sum(axis=0, keepdims=True)
    col_sum[col_sum <= 0] = 1.0
    prob = prob / col_sum
    # Final clip guarantees the eps floor for librosa's internal log;
    # column sums stay within 1..1+6*eps which viterbi tolerates.
    prob = np.clip(prob, eps, 1.0)
    return prob.astype(np.float64)


# ── 1c: phoneme mask + global transition ─────────────────────────────
# NOTE: librosa.sequence.viterbi(prob, transition) only accepts ONE
# time-invariant (6,6) transition matrix, so per-frame rules ("zero the
# upper triangle on vowel->consonant frames") CANNOT be a global matrix:
# they would forbid all upward moves everywhere and risk all-zero rows
# (-> log -> nan, pitfall #1). Phoneme forcing therefore lives in the
# emission (1b clamp to 0.1); the transition stays global-smooth; note
# boundaries are handled by per-segment decode in 1d.

_SINGABLE_JA = {"a", "i", "u", "e", "o"}
_SINGABLE_ZH_VOWELS = set("aeiouv")
_SINGABLE_EN = {
    "aa", "ae", "ah", "ao", "aw", "ax", "ay",
    "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw",
}
_NON_SINGABLE_WORDS = {"SP", "AP", "EP", "br", "sil", "pau"}


def _normalize_phone_text(phone_text: str) -> str:
    return str(phone_text or "").split("/")[-1].strip()


def is_singable_phone(phone_text: str, language: str | None = None) -> bool:
    """Local copy of game_api._is_singable_phone (avoids onnxruntime import)."""
    raw = _normalize_phone_text(phone_text)
    phone = raw.lower()
    lang = (language or "").lower()
    if not phone or phone == "sp":
        return False
    if lang == "ja":
        return phone in _SINGABLE_JA or raw == "N"
    if lang == "en":
        return phone in _SINGABLE_EN
    if phone in {"n", "ng", "m"}:
        return False
    return any(ch in _SINGABLE_ZH_VOWELS for ch in phone)


def build_voiced_mask(
    frame_times: np.ndarray,
    pred_dict: dict | None = None,
    chunks: list | None = None,
    language: str | None = "zh",
) -> np.ndarray:
    """Per-frame voiced mask from HFA phonemes (global seconds).

    frame_times MUST be idx*hop/sr (from extract_rms_obs), phoneme
    start/end are chunk-LOCAL seconds so global = chunk_offset + local.
    No pred_dict/chunks -> all voiced (pure-energy fallback).
    """
    times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    n = int(times.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if not pred_dict or not chunks:
        return np.ones(n, dtype=bool)

    unvoiced: list[tuple[float, float]] = []
    for idx, chunk in enumerate(chunks):
        offset = float((chunk or {}).get("offset", 0.0))
        pred = pred_dict.get(f"chunk_{idx}")
        if pred is None or len(pred) < 3:
            continue
        words = pred[2] or []
        for word in words:
            wtext = getattr(word, "text", "")
            wstart = float(getattr(word, "start", 0.0)) + offset
            wend = float(getattr(word, "end", 0.0)) + offset
            if wtext in _NON_SINGABLE_WORDS:
                unvoiced.append((wstart, wend))
                continue
            for phoneme in getattr(word, "phonemes", None) or []:
                if not is_singable_phone(getattr(phoneme, "text", ""), language):
                    unvoiced.append((float(phoneme.start) + offset, float(phoneme.end) + offset))

    if not unvoiced:
        return np.ones(n, dtype=bool)
    mask = np.ones(n, dtype=bool)
    for start, end in unvoiced:
        if end <= start:
            continue
        mask[(times >= start) & (times < end)] = False
    return mask


def build_transition_matrix() -> np.ndarray:
    """Global (6,6) smooth transition, rows sum to 1.

    Raw weights by state distance: d0=0.85 (strong self-loop), d1=0.06,
    d2=0.008, d3=0.004, d4=0.002, d5=0.01 (-> [0,5]≈0.01 after norm).
    """
    raw_by_dist = {0: 0.85, 1: 0.06, 2: 0.008, 3: 0.004, 4: 0.002, 5: 0.01}
    mat = np.zeros((N_VELOCITY_STATES, N_VELOCITY_STATES), dtype=np.float64)
    for i in range(N_VELOCITY_STATES):
        for j in range(N_VELOCITY_STATES):
            mat[i, j] = raw_by_dist[abs(i - j)]
    row_sum = mat.sum(axis=1, keepdims=True)
    assert bool((row_sum > 0).all()), "transition has an all-zero row"
    mat = mat / row_sum
    return mat


def note_boundary_frames(notes: list, frame_times: np.ndarray) -> list[int]:
    """Frame indices of note onsets (excluding the first note).

    notes are GLOBAL seconds; frame_times are idx*hop/sr. Used by 1d to
    split viterbi decode into per-note segments (= boundary relaxation
    without needing a time-varying transition).
    """
    times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    if times.size == 0 or not notes:
        return []
    ordered = sorted(notes, key=lambda n: float(getattr(n, "onset", 0.0)))
    bounds: list[int] = []
    for note in ordered[1:]:
        onset = float(getattr(note, "onset", 0.0))
        idx = int(np.searchsorted(times, onset, side="left"))
        idx = max(1, min(idx, int(times.size) - 1))
        if not bounds or idx != bounds[-1]:
            bounds.append(idx)
    return bounds


# ── 1d: viterbi decode + smoothing + per-note velocities ──────────────

def decode_states(
    prob: np.ndarray,
    transition: np.ndarray | None = None,
    segment_bounds: list[int] | None = None,
) -> np.ndarray:
    """Viterbi decode via librosa (prob=(6,T) raw probs, NOT log).

    segment_bounds: frame indices (from note_boundary_frames) where a new
    note starts; decode is split per segment and concatenated, which is
    the boundary-relaxation substitute for a time-varying transition.
    Returns states [T] int in [0,5]. nan input -> raises (fail fast).
    """
    import librosa.sequence

    prob = np.asarray(prob, dtype=np.float64)
    if prob.shape[0] != N_VELOCITY_STATES:
        raise ValueError(f"prob must be (6,T), got {prob.shape}")
    n_frames = int(prob.shape[1])
    if n_frames == 0:
        return np.zeros((0,), dtype=np.int32)
    if bool(np.isnan(prob).any()):
        raise ValueError("prob contains nan (check eps floor / transition rows)")
    trans = build_transition_matrix() if transition is None else np.asarray(transition, dtype=np.float64)
    if trans.shape != (N_VELOCITY_STATES, N_VELOCITY_STATES):
        raise ValueError(f"transition must be (6,6), got {trans.shape}")

    if not segment_bounds:
        states = librosa.sequence.viterbi(prob, trans)
        return np.asarray(states, dtype=np.int32).reshape(-1)

    cuts = sorted({int(b) for b in segment_bounds if 0 < int(b) < n_frames})
    edges = [0, *cuts, n_frames]
    parts: list[np.ndarray] = []
    for a, b in zip(edges, edges[1:]):
        seg = prob[:, a:b]
        if seg.shape[1] == 0:
            continue
        parts.append(np.asarray(librosa.sequence.viterbi(seg, trans), dtype=np.int32).reshape(-1))
    if not parts:
        return np.zeros((n_frames,), dtype=np.int32)
    return np.concatenate(parts, axis=0).astype(np.int32)


def states_to_frame_velocity(states: np.ndarray) -> np.ndarray:
    """Map decoded states [T] to MIDI velocity centers (float [T])."""
    states = np.asarray(states).reshape(-1).astype(np.int32)
    if states.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if bool((states < 0).any()) or bool((states >= N_VELOCITY_STATES).any()):
        raise ValueError("states out of range [0,5]")
    return STATE_VELOCITIES[states].astype(np.float64)


def smooth_velocity_curve(vel: np.ndarray, window: int = 11) -> np.ndarray:
    """savgol smooth frame velocities; clip to [1,127]; never nan."""
    from scipy.signal import savgol_filter

    v = np.asarray(vel, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return v
    if v.size < 5:
        return np.clip(v, 1.0, 127.0)
    w = max(5, int(window) | 1)  # odd >= 5
    w = min(w, int(v.size // 2 * 2 + 1))
    if w < 5:
        return np.clip(v, 1.0, 127.0)
    poly = 2 if w > 3 else 1
    try:
        out = savgol_filter(v, window_length=w, polyorder=poly, mode="interp")
    except Exception:
        out = v
    out = np.asarray(out, dtype=np.float64)
    out[~np.isfinite(out)] = np.clip(v[~np.isfinite(out)], 1.0, 127.0)
    return np.clip(out, 1.0, 127.0)


def velocity_to_dyn(vel: float | np.ndarray) -> np.ndarray | int:
    """Map MIDI velocity to USTX dyn: (vel-64)*3 clipped to [-240,120]."""
    arr = np.asarray(vel, dtype=np.float64)
    dyn = np.clip(np.round((arr - 64.0) * 3.0), -240, 120).astype(np.int32)
    if dyn.ndim == 0:
        return int(dyn)
    return dyn


def note_velocities(
    notes: list,
    frame_times: np.ndarray,
    frame_vel: np.ndarray,
    fallback: int = 100,
) -> list[int]:
    """Per-note velocity = median frame velocity in trimmed note span.

    Trim mirrors ustx _build_pitd_curve: edge_trim=min(0.025, dur*0.15).
    Frames strictly inside [onset+trim, offset-trim); empty span or no
    frames -> fallback. Returns ints in [1,127], note order preserved.
    """
    times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    vel = np.asarray(frame_vel, dtype=np.float64).reshape(-1)
    if times.shape[0] != vel.shape[0]:
        raise ValueError(f"frame_times {times.shape[0]} != frame_vel {vel.shape[0]}")
    out: list[int] = []
    for note in notes or []:
        onset = float(getattr(note, "onset", 0.0))
        offset = float(getattr(note, "offset", 0.0))
        dur = max(0.0, offset - onset)
        trim = min(0.025, dur * 0.15)
        lo, hi = onset + trim, offset - trim
        if hi <= lo or times.size == 0:
            out.append(int(fallback))
            continue
        sel = vel[(times >= lo) & (times < hi)]
        if sel.size == 0:
            out.append(int(fallback))
            continue
        out.append(int(np.clip(round(float(np.median(sel))), 1, 127)))
    return out
