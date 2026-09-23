"""Tests for the phoneme-guided HMM velocity (dynamics) pipeline.

Cases 1-3 cover the core chain; case 4 (short-note dense transition,
appended LAST - do not reorder) guards against over-smoothing turning
step jumps into pseudo-crescendo slopes.
"""

import unittest

import numpy as np

from inference.API.velocity_api import (
    build_emission_probs,
    build_transition_matrix,
    build_voiced_mask,
    decode_states,
    extract_rms_obs,
    extract_velocity,
    note_boundary_frames,
    note_velocities,
    smooth_velocity_curve,
    states_to_frame_velocity,
)

SR = 44100


def _tone(seconds: float, amp=1.0, freq: float = 440.0) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_extract_rms_obs_range_and_grid():
    y = _tone(2.0)
    obs = extract_rms_obs(y, SR)
    assert obs.norm_obs.size > 0
    assert not bool(np.isnan(obs.norm_obs).any())
    assert float(obs.norm_obs.min()) >= 0.0 and float(obs.norm_obs.max()) <= 1.0
    # Frame grid must be idx*hop/sr, never an assumed 10ms.
    step = float(obs.frame_times[1] - obs.frame_times[0])
    assert abs(step - 512 / SR) < 1e-9


def test_emission_transition_shapes():
    y = _tone(1.0)
    obs = extract_rms_obs(y, SR)
    prob = build_emission_probs(obs.norm_obs, np.ones(len(obs.norm_obs), dtype=bool))
    assert prob.shape == (6, len(obs.norm_obs))
    assert not bool(np.isnan(prob).any())
    assert float(prob.min()) >= 1e-10
    np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-6)

    trans = build_transition_matrix()
    assert trans.shape == (6, 6)
    np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-12)
    assert abs(float(trans[0, 5]) - 0.01) < 0.003
    for i in range(6):
        assert trans[i, i] == trans[i].max()


def test_crescendo_monotonic_and_smoothing():
    from types import SimpleNamespace

    t = np.arange(SR * 3) / SR
    y = (_tone(3.0) * np.linspace(0.1, 1.0, len(t))).astype(np.float32)
    obs = extract_rms_obs(y, SR)
    prob = build_emission_probs(obs.norm_obs)
    states = decode_states(prob, build_transition_matrix())
    assert states[: len(states) // 2].mean() < states[len(states) // 2 :].mean()

    raw = states_to_frame_velocity(states)
    smooth = smooth_velocity_curve(raw)
    assert not bool(np.isnan(smooth).any())
    # savgol turns steps into ramps: max-step must drop, trend must hold.
    assert float(np.abs(np.diff(smooth)).max()) <= float(np.abs(np.diff(raw)).max())
    assert float(np.corrcoef(raw, smooth)[0, 1]) > 0.95

    notes = [
        SimpleNamespace(onset=0.0, offset=1.0),
        SimpleNamespace(onset=1.0, offset=2.0),
        SimpleNamespace(onset=2.0, offset=3.0),
    ]
    vel = note_velocities(notes, obs.frame_times, smooth)
    assert vel[0] <= vel[1] <= vel[2]
    assert all(1 <= v <= 127 for v in vel)


def test_short_note_dense_transition_width():
    """Dense 16ths must keep flat interiors and narrow boundary jumps.

    8 x 16th notes at BPM=120 (125ms each), alternating soft/loud.
    Where adjacent-note dyn differs by >20, the transition width
    (ticks strictly between the two plateaus -> seconds) must stay
    within 30ms plus a +-5ms ticks-to-seconds rounding tolerance.
    """
    from types import SimpleNamespace

    from inference.API.ustx_api import _build_dyn_curve

    bpm, tempo = 120, 120.0
    note_dur = 60 / bpm / 4  # 0.125s
    parts = [_tone(note_dur, amp=0.3 if k % 2 == 0 else 1.0) for k in range(8)]
    y = np.concatenate(parts)
    notes = [
        SimpleNamespace(onset=k * note_dur, offset=(k + 1) * note_dur, pitch=60.0, lyric="a")
        for k in range(8)
    ]

    res = extract_velocity(y, SR, notes)
    assert len(res.note_velocities) == 8
    xs, ys = _build_dyn_curve(notes, res.dyn_xs, res.dyn_ys, tempo)
    assert len(xs) > 0

    ticks_per_sec = tempo * 8.0
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys, dtype=float)
    for k in range(7):
        # Plateaus = median of each note's interior curve points.
        left = ys_arr[(xs_arr >= notes[k].onset * ticks_per_sec) & (xs_arr < notes[k].offset * ticks_per_sec)]
        right = ys_arr[
            (xs_arr >= notes[k + 1].onset * ticks_per_sec) & (xs_arr < notes[k + 1].offset * ticks_per_sec)
        ]
        if left.size == 0 or right.size == 0:
            continue
        lo_med, hi_med = float(np.median(left)), float(np.median(right))
        if abs(hi_med - lo_med) <= 20:
            continue
        lo, hi = (lo_med, hi_med) if lo_med < hi_med else (hi_med, lo_med)
        span = hi - lo
        # Boundary window +-15 ticks (~15.6ms each side at 120bpm).
        tb = notes[k + 1].onset * ticks_per_sec
        in_win = (xs_arr >= tb - 15) & (xs_arr <= tb + 15)
        mid = in_win & (ys_arr > lo + 0.1 * span) & (ys_arr < hi - 0.1 * span)
        width_sec = float(np.count_nonzero(mid)) * 5 / ticks_per_sec
        # Tolerance: 30ms budget + 5ms ticks->seconds rounding slack.
        unittest.TestCase().assertLessEqual(width_sec, 0.030 + 0.005)
