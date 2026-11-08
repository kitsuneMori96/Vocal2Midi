"""Write pitch and voicing reference curves into an OpenUtau USTX.

OpenUtau's built-in pitch curve (``pitd``) only exists inside detected notes, so
it cannot help when the note segmentation itself is wrong. This module adds two
project-level *masked* curves instead, drawn by the curve panel with real gaps
where the voice is silent:

* ``f0``  - absolute pitch in cents (MIDI note number x100), so 6500 is 65.00.
* ``vuv`` - flat 100 while the frame is voiced; gaps mark silence.

MaskedCurve descriptors are always offered in OpenUtau's expression selector
regardless of the renderer, which is why they are used instead of plain curves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from inference.API.rmvpe_api import RmvpeResult
from inference.API.ustx_api import UCurveInterval

logger = logging.getLogger(__name__)

F0_ABBR = "f0"
VUV_ABBR = "vuv"
F0_SCALE = 100.0
VUV_VOICED = 100

CURVE_EXPRESSIONS = {
    F0_ABBR: {
        "name": "pitch (midi x100)",
        "abbr": F0_ABBR,
        "type": "MaskedCurve",
        "min": 2400,
        "max": 10800,
        "default_value": 6000,
        "is_flag": False,
        "flag": "",
    },
    VUV_ABBR: {
        "name": "voiced (gaps = silence)",
        "abbr": VUV_ABBR,
        "type": "MaskedCurve",
        "min": 0,
        "max": 200,
        "default_value": 0,
        "is_flag": False,
        "flag": "",
    },
}


def seconds_to_ticks(seconds: float, tempos: list[dict[str, Any]], resolution: int) -> int:
    """Convert song time to ticks honouring every tempo change."""
    if seconds <= 0:
        return 0
    ordered = sorted(
        ({"position": float(t.get("position", 0)), "bpm": float(t.get("bpm") or 120)} for t in tempos),
        key=lambda t: t["position"],
    ) or [{"position": 0.0, "bpm": 120.0}]
    if ordered[0]["position"] > 0:
        ordered.insert(0, {"position": 0.0, "bpm": ordered[0]["bpm"]})

    elapsed = 0.0
    for i, tempo in enumerate(ordered):
        ticks_per_sec = tempo["bpm"] * resolution / 60.0
        next_position = ordered[i + 1]["position"] if i + 1 < len(ordered) else None
        available = (next_position - tempo["position"]) / ticks_per_sec if next_position is not None else None
        if available is None or seconds <= elapsed + available:
            return int(round(tempo["position"] + (seconds - elapsed) * ticks_per_sec))
        elapsed += available
    last = ordered[-1]
    return int(round(last["position"] + seconds * last["bpm"] * resolution / 60.0))


def ticks_to_bar_beat(
    ticks: float, time_signatures: list[dict[str, Any]] | None, resolution: int
) -> tuple[int, float]:
    """1-based bar number and 1-based beat within the bar, matching the OpenUtau ruler."""
    sigs = sorted(
        (
            {
                "bar": int(ts.get("bar_position", 0) or 0),
                "num": int(ts.get("beat_per_bar", 4) or 4),
                "den": int(ts.get("beat_unit", 4) or 4),
            }
            for ts in (time_signatures or [])
        ),
        key=lambda s: s["bar"],
    )
    if not sigs or sigs[0]["bar"] > 0:
        sigs.insert(0, {"bar": 0, "num": 4, "den": 4})
    tick = max(0, int(ticks))
    acc = 0
    for i, sig in enumerate(sigs):
        ticks_per_beat = resolution * 4 / sig["den"]
        ticks_per_bar = ticks_per_beat * sig["num"]
        next_bar = sigs[i + 1]["bar"] if i + 1 < len(sigs) else None
        span = (next_bar - sig["bar"]) * ticks_per_bar if next_bar is not None else None
        if span is None or tick < acc + span:
            within = tick - acc
            return sig["bar"] + int(within // ticks_per_bar) + 1, (within % ticks_per_bar) / ticks_per_beat + 1
        acc += span
    last = sigs[-1]
    return last["bar"] + 1, 1.0


def _quantize(tick: float) -> int:
    return int(round(tick / UCurveInterval) * UCurveInterval)


def _frame_ticks(
    rmvpe: RmvpeResult, tempos: list[dict[str, Any]], resolution: int, part_offset: int
) -> list[float]:
    offset = float(getattr(rmvpe, "time_offset", 0.0) or 0.0)
    return [
        seconds_to_ticks(i * rmvpe.time_step_seconds + offset, tempos, resolution) - part_offset
        for i in range(len(rmvpe.midi_pitch))
    ]


def _voiced_frames(rmvpe: RmvpeResult) -> np.ndarray:
    count = len(rmvpe.midi_pitch)
    mask = getattr(rmvpe, "voiced_mask", None)
    if mask is None:
        return np.isfinite(rmvpe.midi_pitch)
    mask = np.asarray(mask, dtype=bool)
    if mask.size < count:
        mask = np.concatenate([mask, np.zeros(count - mask.size, dtype=bool)])
    return mask[:count]


def _spans(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (first, last) index pairs of consecutive True frames."""
    spans: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(flags) - 1))
    return spans


def _run_values(ticks: list[float], values: np.ndarray, first: int, last: int) -> tuple[int, list[int]] | None:
    """Resample frames ``first..last`` onto the 5-tick grid as one masked run."""
    span_ticks = ticks[first:last + 1]
    start_x = _quantize(span_ticks[0])
    end_x = _quantize(span_ticks[-1])
    if end_x < start_x:
        return None
    grid = list(range(start_x, end_x + 1, UCurveInterval))
    if len(span_ticks) == 1:
        ys = [int(round(float(values[first]))) for _ in grid]
        return start_x, ys
    interpolated = np.interp(grid, span_ticks, values[first:last + 1])
    return start_x, [int(round(float(v))) for v in interpolated]


def build_f0_runs(
    rmvpe: RmvpeResult,
    tempos: list[dict[str, Any]],
    resolution: int,
    part_offset_ticks: int = 0,
) -> list[dict[str, Any]]:
    """Absolute MIDI pitch (x100) as masked runs; unvoiced stretches stay empty."""
    midi = np.asarray(rmvpe.midi_pitch, dtype=float)
    if midi.size == 0:
        return []
    voiced = _voiced_frames(rmvpe)
    ticks = _frame_ticks(rmvpe, tempos, resolution, part_offset_ticks)
    values = np.clip(midi * F0_SCALE, 0, 12700)
    runs: list[dict[str, Any]] = []
    next_free_x = None
    for first, last in _spans(voiced & np.isfinite(midi)):
        result = _run_values(ticks, values, first, last)
        if result is None:
            continue
        start_x, ys = result
        if next_free_x is not None and start_x < next_free_x:
            # A gap shorter than one grid step: keep the runs ordered and apart.
            drop = (next_free_x - start_x + UCurveInterval - 1) // UCurveInterval
            if drop >= len(ys):
                continue
            ys = ys[drop:]
            start_x += drop * UCurveInterval
        runs.append({"x": start_x, "ys": ys})
        next_free_x = start_x + len(ys) * UCurveInterval
    return runs


def build_vuv_runs(
    rmvpe: RmvpeResult,
    tempos: list[dict[str, Any]],
    resolution: int,
    part_offset_ticks: int = 0,
) -> list[dict[str, Any]]:
    """Flat voiced runs; silence is simply the absence of a run."""
    if len(rmvpe.midi_pitch) == 0:
        return []
    voiced = _voiced_frames(rmvpe)
    ticks = _frame_ticks(rmvpe, tempos, resolution, part_offset_ticks)
    runs: list[dict[str, Any]] = []
    for first, last in _spans(voiced):
        start_x = _quantize(ticks[first])
        end_x = _quantize(ticks[last])
        if end_x < start_x:
            continue
        count = (end_x - start_x) // UCurveInterval + 1
        runs.append({"x": start_x, "ys": [VUV_VOICED] * count})
    return runs


def _select_part(project: dict[str, Any], track: int) -> dict[str, Any]:
    parts = project.get("voice_parts") or []
    if not parts:
        raise ValueError("USTX has no voice_parts")
    matching = [part for part in parts if int(part.get("track_no", 0)) == track]
    if not matching:
        raise ValueError(f"USTX has no voice part on track {track}")
    return max(matching, key=lambda part: len(part.get("notes") or []))


def _clip_runs(runs: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    """Drop the values inside ``[start, end]``, splitting runs that straddle it."""
    result: list[dict[str, Any]] = []
    for run in runs:
        x0 = int(run["x"])
        ys = list(run.get("ys") or [])
        if not ys:
            continue
        run_end = x0 + (len(ys) - 1) * UCurveInterval
        if run_end < start or x0 > end:
            result.append({"x": x0, "ys": ys})
            continue
        current_x: int | None = None
        current: list[int] = []
        for i, y in enumerate(ys):
            x = x0 + i * UCurveInterval
            if start <= x <= end:
                if current:
                    result.append({"x": current_x, "ys": current})
                    current, current_x = [], None
                continue
            if current and current_x is not None and x == current_x + len(current) * UCurveInterval:
                current.append(y)
            else:
                if current:
                    result.append({"x": current_x, "ys": current})
                current_x, current = x, [y]
        if current:
            result.append({"x": current_x, "ys": current})
    return result


def _register_expressions(project: dict[str, Any], abbrs: list[str]) -> None:
    expressions = project.setdefault("expressions", {})
    selectors = project.setdefault("exp_selectors", [])
    for abbr in abbrs:
        expressions.setdefault(abbr, dict(CURVE_EXPRESSIONS[abbr]))
        if abbr not in selectors:
            selectors.append(abbr)


def _merge_runs(
    existing: list[dict[str, Any]] | None,
    new_runs: list[dict[str, Any]],
    window_ticks: tuple[int, int] | None,
    preserve_existing: bool,
) -> list[dict[str, Any]]:
    if existing is None:
        return [dict(run) for run in new_runs]
    if preserve_existing or window_ticks is None:
        kept = [dict(run) for run in existing] if preserve_existing else []
    else:
        kept = _clip_runs(existing, window_ticks[0], window_ticks[1])
    return sorted(kept + [dict(run) for run in new_runs], key=lambda run: int(run["x"]))


def inject_curves(
    base_path: Path,
    out_path: Path,
    runs_by_abbr: dict[str, list[dict[str, Any]]],
    *,
    window_seconds: tuple[float, float] | None = None,
    track: int = 0,
    preserve_existing: bool = False,
) -> dict[str, Any]:
    """Copy ``base_path`` to ``out_path`` with masked curves written on its part.

    ``window_seconds`` limits the region that gets replaced when the base file
    already holds a curve with the same abbr; values outside it are preserved.
    New curves are always written in full. ``preserve_existing`` keeps the old
    runs too instead of replacing them inside the window.
    """
    base_path = Path(base_path)
    out_path = Path(out_path)
    if base_path.resolve() == out_path.resolve():
        raise ValueError("refusing to overwrite the base USTX; choose another output path")
    with base_path.open("r", encoding="utf-8") as f:
        project = yaml.safe_load(f) or {}
    part = _select_part(project, track)
    resolution = int(project.get("resolution") or 480)
    tempos = project.get("tempos") or [{"position": 0, "bpm": float(project.get("bpm") or 120)}]
    part_offset = int(part.get("position") or 0)

    window_ticks = None
    if window_seconds is not None:
        start = seconds_to_ticks(window_seconds[0], tempos, resolution) - part_offset
        end = seconds_to_ticks(window_seconds[1], tempos, resolution) - part_offset
        window_ticks = (min(start, end), max(start, end))

    existing = {entry.get("abbr"): entry for entry in (part.get("masked_curves") or [])}
    _register_expressions(project, list(runs_by_abbr))
    part_curves = [entry for entry in (part.get("masked_curves") or []) if entry.get("abbr") not in runs_by_abbr]
    summary: dict[str, Any] = {
        "base": str(base_path),
        "output": str(out_path),
        "resolution": resolution,
        "part_position": part_offset,
        "window_ticks": list(window_ticks) if window_ticks else None,
        "curves": {},
    }
    max_x = 0
    for abbr, runs in runs_by_abbr.items():
        if not runs:
            logger.warning(f"[Curves] {abbr}: no runs, skipped")
            continue
        previous = existing.get(abbr)
        merged = _merge_runs(previous.get("runs") if previous else None, runs, window_ticks, preserve_existing)
        part_curves.append({"abbr": abbr, "runs": merged})
        values = [y for run in merged for y in run["ys"]]
        first_x = min(int(run["x"]) for run in merged)
        last_x = max(int(run["x"]) + (len(run["ys"]) - 1) * UCurveInterval for run in merged)
        max_x = max(max_x, last_x)
        summary["curves"][abbr] = {
            "runs": len(merged),
            "values": len(values),
            "x_range": [first_x, last_x],
            "y_range": [min(values), max(values)],
        }
    part["masked_curves"] = part_curves
    part["duration"] = max(int(part.get("duration") or 0), max_x)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(project, f, allow_unicode=True, sort_keys=False)
    summary_path = out_path.with_name(f"{out_path.stem}.curves.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"[Curves] wrote {summary_path}: "
        + ", ".join(f"{abbr}={info['runs']}run/{info['values']}val" for abbr, info in summary["curves"].items())
    )
    return summary
