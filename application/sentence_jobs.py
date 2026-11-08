"""Long-running sentence-repair work shared by the GUI and the CLI scripts.

Keeping it here means the reference-curve export runs through exactly one code
path whether it is triggered from the command line or from the desktop UI.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
MIN_CLIP_SEC = 0.2


def read_ustx_header(path) -> dict:
    """Tempo map, resolution and project body of an existing USTX export."""
    with Path(path).open("r", encoding="utf-8") as handle:
        project = yaml.safe_load(handle) or {}
    tempos = project.get("tempos") or [{"position": 0, "bpm": project.get("bpm") or 120}]
    bpm = float(project.get("bpm") or tempos[0].get("bpm") or 120)
    return {
        "project": project,
        "tempos": tempos,
        "resolution": int(project.get("resolution") or 480),
        "bpm": bpm,
    }


def bar_beat_label(seconds: float, header: dict) -> str:
    """OpenUtau-ruler style position (bar.beat) for a song time, or '' without a header."""
    from inference.pipeline.ustx_curves import seconds_to_ticks, ticks_to_bar_beat

    if not header:
        return ""
    ticks = seconds_to_ticks(seconds, header["tempos"], header["resolution"])
    bar, beat = ticks_to_bar_beat(ticks, header["project"].get("time_signatures"), header["resolution"])
    return f"{bar}.{int(beat)}"


def _patch_ustx(base_path, out_path, transform, *, save_timeout: float = 10.0):
    """Run a USTX transform, syncing with a running OpenUtau when in place.

    Writing to the base file itself first asks OpenUtau to save and waits for
    its ack (OpenUtauSaveError when it cannot), then writes atomically and
    keeps a rolling backup at <stem>_backup.ustx. Any other output path leaves
    the base untouched and needs no handshake.
    """
    from application.openutau_bridge import request_save

    base_path = Path(base_path)
    out_path = Path(out_path)
    if out_path.resolve() != base_path.resolve():
        return transform(base_path, out_path)
    request_save(base_path, timeout=save_timeout)
    tmp = base_path.with_name(base_path.stem + ".v2m-tmp.ustx")
    try:
        result = transform(base_path, tmp)
        shutil.copy2(base_path, base_path.with_name(base_path.stem + "_backup.ustx"))
        os.replace(tmp, base_path)
    finally:
        tmp.unlink(missing_ok=True)
    return result


def splice_take_with_handshake(
    base_path, take_path, out_path, window: tuple[float, float], *, save_timeout: float = 10.0
) -> tuple[int, int]:
    """Replace base notes in the window with the take, syncing when in place."""
    from inference.pipeline.sentence_gacha import splice_take_into_ustx

    return _patch_ustx(
        base_path, out_path,
        lambda base, out: splice_take_into_ustx(base, take_path, out, window),
        save_timeout=save_timeout,
    )


def add_reference_take_with_handshake(
    base_path, take_path, out_path, window: tuple[float, float], track_name: str,
    *, save_timeout: float = 10.0,
) -> int:
    """Add the take as a muted reference track, syncing when written in place."""
    from inference.pipeline.sentence_gacha import add_reference_take_into_ustx

    return _patch_ustx(
        base_path, out_path,
        lambda base, out: add_reference_take_into_ustx(base, take_path, out, window, track_name),
        save_timeout=save_timeout,
    )


def export_reference_curves(
    *,
    audio_path,
    base_path,
    out_path,
    window: tuple[float, float] | None,
    curves=("f0", "vuv"),
    device: str = "cpu",
    rmvpe_model_path: str = "",
    track: int = 0,
    replace_window: bool = True,
    preloaded_audio: tuple | None = None,
    cancel_checker=None,
) -> dict:
    """Write f0/vuv masked curves for one span into a copy of ``base_path``.

    ``window`` is ``None`` to process the whole audio. Returns the
    ``inject_curves`` summary plus the resolved output path and window.
    """
    from inference.io.audio_io import load_audio
    from inference.pipeline.auto_lyric_hybrid import _extract_pitch_curve
    from inference.pipeline.ustx_curves import build_f0_runs, build_vuv_runs, inject_curves

    audio_path = Path(audio_path)
    base_path = Path(base_path)
    out_path = Path(out_path)
    if out_path.resolve() == base_path.resolve():
        raise ValueError("refusing to overwrite the base USTX; choose another output path")
    if preloaded_audio is not None:
        waveform, sr = preloaded_audio
    else:
        waveform, sr = load_audio(str(audio_path), SAMPLE_RATE)
    total = len(waveform) / float(sr)
    if window is None:
        start, end = 0.0, total
    else:
        start = max(0.0, min(float(window[0]), total))
        end = max(start, min(float(window[1]), total))
    clip = waveform[int(round(start * sr)):int(round(end * sr))]
    if clip.size < int(MIN_CLIP_SEC * sr):
        raise ValueError(f"clip is too short: {start:.3f}-{end:.3f}s")

    rmvpe = _extract_pitch_curve(
        clip, sr,
        rmvpe_model_path=rmvpe_model_path,
        output_format_set=["ustx"],
        output_pitch_curve=True,
        device=device,
        cancel_checker=cancel_checker,
    )
    if rmvpe is None:
        raise RuntimeError("RMVPE did not produce a pitch curve")
    rmvpe = replace(rmvpe, time_offset=start)

    header = read_ustx_header(base_path)
    part = (header["project"].get("voice_parts") or [{}])[0]
    part_offset = int(part.get("position") or 0)
    runs_by_abbr = {}
    if "f0" in curves:
        runs_by_abbr["f0"] = build_f0_runs(
            rmvpe, header["tempos"], header["resolution"], part_offset
        )
    if "vuv" in curves:
        runs_by_abbr["vuv"] = build_vuv_runs(
            rmvpe, header["tempos"], header["resolution"], part_offset
        )

    summary = inject_curves(
        base_path, out_path, runs_by_abbr,
        window_seconds=(start, end) if replace_window else None,
        track=track,
        preserve_existing=not replace_window,
    )
    summary["output"] = str(out_path)
    summary["window"] = (start, end)
    return summary
