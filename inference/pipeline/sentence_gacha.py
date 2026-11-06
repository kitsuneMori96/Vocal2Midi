"""Sentence-level "gacha": rerun one lyric line with different decoding params.

The ASR/HFA stage depends only on the clip and its lyrics, so it runs once per
alignment engine; every take after that is GAME-only. Candidates keep the
song's absolute timeline so the best one can be spliced back into the
full-song USTX.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from inference.API.lfa_api import create_lyric_matcher
from inference.API.ustx_api import save_ustx
from inference.LyricFA.tools.lyric_matcher import (
    _LRC_CREDIT_RE,
    _LRC_METADATA_RE,
    _LRC_TIMESTAMP_RE,
)
from inference.io.audio_io import load_audio
from inference.io.note_io import NoteInfo
from inference.pipeline.auto_lyric_hybrid import (
    _extract_pitch_curve,
    _prepare_context,
    _run_game_stage,
    _slice_chunks,
)
from inference.pipeline.lyric_alignment import _run_lyric_alignment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
MIN_CLIP_SEC = 3.0
SHORT_NOTE_SEC = 0.06
RUN_FLAG_LENGTH = 3
MAX_REPEAT_RUN = 2

RANDOM_RANGES = {
    "seg_threshold": (0.12, 0.30),
    "seg_radius": (0.010, 0.050),
    "est_threshold": (0.15, 0.30),
    "t0": (0.0, 0.30),
}
RANDOM_NSTEPS = (6, 8, 10, 12)
GRID_DEFAULTS = {
    "seg_threshold": (0.15, 0.20, 0.25),
    "seg_radius": (0.01, 0.02, 0.035),
    "est_threshold": (0.15, 0.20, 0.25),
    "t0": (0.0, 0.2),
    "nsteps": (8,),
}


@dataclass(frozen=True)
class LyricLine:
    index: int
    text: str
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class TakeParams:
    label: str
    alignment_engine: str
    seg_threshold: float
    seg_radius: float
    est_threshold: float
    t0: float
    nsteps: int

    def to_dict(self) -> dict:
        return {
            "take": self.label,
            "alignment_engine": self.alignment_engine,
            "seg_threshold": self.seg_threshold,
            "seg_radius": self.seg_radius,
            "est_threshold": self.est_threshold,
            "t0": self.t0,
            "nsteps": self.nsteps,
        }


@dataclass
class TakeResult:
    params: TakeParams
    notes: list[NoteInfo]
    ustx_path: Path
    diagnostics: dict


@dataclass
class GachaConfig:
    audio_path: str
    output_dir: Path
    output_key: str
    language: str
    lyric_output_mode: str
    device: str
    game_model_dir: str
    hfa_model_dir: str
    asr_model_path: str
    tifa_model_path: str
    tempo: float
    batch_size: int = 4
    asr_batch_size: int = 4
    slicing_method: str = "smart"
    slice_min_sec: float = 5.0
    slice_max_sec: float = 10.0
    pitch_curve: bool = False
    rmvpe_model_path: str = ""
    phoneme_asr_model_path: str = ""
    pinyin_asr_model_path: str = ""
    chinese_asr_engine: str = "qwen"
    japanese_asr_engine: str = "romaji"
    japanese_g2p_engine: str = "kashi-g2p-onnx"
    kashi_g2p_model_path: str = ""
    asr_session: object = None


def _parse_timestamp(stamp: str) -> float:
    minutes, seconds = stamp[1:-1].split(":", 1)
    return int(minutes) * 60.0 + float(seconds.replace(":", "."))


def _iter_sung_lines(raw_text: str):
    """Yield (text, timestamps) with the same filters as strip_lrc_markup."""
    raw_text = raw_text.lstrip("\ufeff")
    if not _LRC_TIMESTAMP_RE.search(raw_text):
        for line in raw_text.splitlines():
            text = line.strip()
            if text:
                yield text, ()
        return

    metadata = {}
    for line in raw_text.splitlines():
        match = _LRC_METADATA_RE.match(line)
        if match:
            metadata[match.group(1).lower()] = match.group(2).strip()
    header = {
        text for text in (
            metadata.get("ti", ""),
            metadata.get("ar", ""),
            metadata.get("al", ""),
            f"{metadata.get('ti', '')} - {metadata.get('ar', '')}".strip(" -"),
        ) if text
    }

    for line in raw_text.splitlines():
        if _LRC_METADATA_RE.match(line):
            continue
        stamps = tuple(_LRC_TIMESTAMP_RE.findall(line))
        text = _LRC_TIMESTAMP_RE.sub("", line).strip()
        if not text or text in header or _LRC_CREDIT_RE.match(text):
            continue
        yield text, stamps


def parse_lyric_lines(raw_text: str) -> list[LyricLine]:
    """Sung lyric lines with LRC start/end times (None for plain text)."""
    lines: list[LyricLine] = []
    for text, stamps in _iter_sung_lines(raw_text):
        if not stamps:
            lines.append(LyricLine(len(lines), text))
            continue
        for stamp in stamps:
            lines.append(LyricLine(len(lines), text, _parse_timestamp(stamp)))
    filled: list[LyricLine] = []
    for i, line in enumerate(lines):
        if line.start is None:
            filled.append(line)
            continue
        next_start = next((l.start for l in lines[i + 1:] if l.start is not None), None)
        filled.append(replace(line, end=next_start))
    return filled


def select_line(lines: list[LyricLine], line_number: int | None = None, sentence: str | None = None) -> LyricLine:
    if line_number is not None:
        if not 1 <= line_number <= len(lines):
            raise ValueError(f"line {line_number} out of range (1..{len(lines)})")
        return lines[line_number - 1]
    if not sentence:
        raise ValueError("give --line or --sentence to pick the target line")
    exact = [line for line in lines if line.text == sentence]
    if len(exact) == 1:
        return exact[0]
    partial = [line for line in lines if sentence in line.text]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"no lyric line contains {sentence!r}")
    listing = ", ".join(f"{line.index + 1}:{line.text}" for line in partial)
    raise ValueError(f"{sentence!r} matches several lines: {listing}; use --line")


def resolve_window(
    lines: list[LyricLine],
    *,
    line: LyricLine | None,
    start: float | None,
    end: float | None,
    pad_before: float,
    pad_after: float,
    total_duration: float,
) -> tuple[float, float]:
    if start is None or end is None:
        if line is None or line.start is None:
            raise ValueError("cannot derive the window from the lyrics; pass --start and --end")
        line_start = line.start
        line_end = line.end if line.end is not None else line_start + 8.0
        start = line_start - pad_before if start is None else start
        end = line_end + pad_after if end is None else end
    start = max(0.0, float(start))
    end = min(float(total_duration), float(end))
    if end - start < MIN_CLIP_SEC:
        end = min(float(total_duration), start + MIN_CLIP_SEC)
    if end <= start:
        raise ValueError(f"invalid window {start:.3f}-{end:.3f}s")
    return start, end


def clip_lyrics_for_window(
    lines: list[LyricLine], start: float, end: float, fallback: str = ""
) -> str:
    selected = [
        line for line in lines
        if line.start is not None
        and line.start < end
        and (line.end if line.end is not None else line.start + 8.0) > start
    ]
    if selected:
        return "\n".join(line.text for line in selected)
    if fallback:
        return fallback
    return "\n".join(line.text for line in lines)


def _round_sig(value: float, digits: int = 3) -> float:
    return float(f"{value:.{digits}f}")


def build_takes(
    *,
    engines: list[str],
    strategy: str,
    takes: int,
    seed: int,
    base: dict,
    seg_thresholds: list[float] | None = None,
    seg_radii: list[float] | None = None,
    est_thresholds: list[float] | None = None,
    t0_values: list[float] | None = None,
    nsteps_values: list[int] | None = None,
) -> list[TakeParams]:
    if takes <= 0:
        raise ValueError("--takes must be positive")
    if not engines:
        raise ValueError("at least one alignment engine is required")

    def make(idx: int, engine: str, values: dict, prefix: str = "") -> TakeParams:
        return TakeParams(
            label=f"{engine}-{prefix}{idx + 1:02d}",
            alignment_engine=engine,
            seg_threshold=values["seg_threshold"],
            seg_radius=values["seg_radius"],
            est_threshold=values["est_threshold"],
            t0=values["t0"],
            nsteps=values["nsteps"],
        )

    if strategy == "repeat":
        values = {
            "seg_threshold": base["seg_threshold"],
            "seg_radius": base["seg_radius"],
            "est_threshold": base["est_threshold"],
            "t0": base["t0"],
            "nsteps": base["nsteps"],
        }
        return [make(i, engine, values, "r") for engine in engines for i in range(takes)]

    if strategy == "grid":
        axes = {
            "seg_threshold": seg_thresholds or GRID_DEFAULTS["seg_threshold"],
            "seg_radius": seg_radii or GRID_DEFAULTS["seg_radius"],
            "est_threshold": est_thresholds or GRID_DEFAULTS["est_threshold"],
            "t0": t0_values or GRID_DEFAULTS["t0"],
            "nsteps": nsteps_values or GRID_DEFAULTS["nsteps"],
        }
        combos = [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]
        if len(combos) > takes:
            if takes == 1:
                combos = [combos[0]]
            else:
                indices = sorted({
                    round(i * (len(combos) - 1) / (takes - 1)) for i in range(takes)
                })
                combos = [combos[i] for i in indices]
        return [make(i, engine, combo) for engine in engines for i, combo in enumerate(combos)]

    if strategy == "random":
        rng = random.Random(seed)
        samples = []
        for _ in range(takes):
            values = {
                name: _round_sig(rng.uniform(lo, hi), 3)
                for name, (lo, hi) in RANDOM_RANGES.items()
            }
            values["nsteps"] = rng.choice(RANDOM_NSTEPS)
            samples.append(values)
        return [make(i, engine, sample) for engine in engines for i, sample in enumerate(samples)]

    raise ValueError(f"unknown strategy {strategy!r}")


def _shifted_rmvpe(result, offset: float):
    if result is None:
        return None
    return replace(result, time_offset=offset) if hasattr(result, "__dataclass_fields__") else result


def _note_diagnostics(notes: list[NoteInfo], start: float, end: float) -> dict:
    inside = [n for n in notes if n.onset < end and n.offset > start]
    durations = [n.offset - n.onset for n in inside]
    short = [n for n in inside if n.offset - n.onset < SHORT_NOTE_SEC]
    longest_run = 0
    run = 0
    previous = None
    for note in inside:
        if previous is not None and note.lyric == previous:
            run += 1
        else:
            run = 1
        longest_run = max(longest_run, run)
        previous = note.lyric
    gaps = 0
    for left, right in zip(inside, inside[1:]):
        if right.onset - left.offset > 0.05:
            gaps += 1
    repeats = sum(n - 1 for n in _run_lengths(inside) if n >= RUN_FLAG_LENGTH)
    # Gaps are normal around sokuon and rests, so only fragments and
    # same-lyric runs feed the score.
    flag_score = len(short) * 2 + repeats * 3
    return {
        "note_count": len(inside),
        "min_duration": round(min(durations), 4) if durations else 0.0,
        "short_notes": len(short),
        "max_same_lyric_run": longest_run,
        "repeated_lyric_notes": repeats,
        "internal_gaps": gaps,
        "flag_score": flag_score,
        "lyric_sequence": " ".join(n.lyric for n in inside),
    }


def _run_lengths(notes: list[NoteInfo]) -> list[int]:
    lengths: list[int] = []
    previous = None
    for note in notes:
        if previous is not None and note.lyric == previous:
            lengths[-1] += 1
        else:
            lengths.append(1)
        previous = note.lyric
    return lengths


def run_gacha(
    config: GachaConfig,
    takes: list[TakeParams],
    window: tuple[float, float],
    *,
    lyric_window_text: str,
    clip_lines_text: str,
    line: LyricLine | None,
    splice_range: tuple[float, float] | None = None,
    cancel_checker=None,
    preloaded_audio: tuple | None = None,
) -> list[TakeResult]:
    start, end = window
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if preloaded_audio is not None:
        waveform, sr = preloaded_audio
    else:
        waveform, sr = load_audio(config.audio_path, SAMPLE_RATE)
    total = len(waveform) / float(sr)
    start = max(0.0, min(start, total))
    end = max(start, min(end, total))
    clip = waveform[int(round(start * sr)):int(round(end * sr))]
    if clip.size < int(0.5 * sr):
        raise ValueError(f"clip is too short: {start:.3f}-{end:.3f}s")
    logger.info(
        f"[Gacha] clip {start:.3f}-{end:.3f}s ({len(clip) / sr:.2f}s), "
        f"{len(takes)} take(s), lyrics={clip_lines_text!r}"
    )

    output_formats = ["ustx"]
    ctx = _prepare_context(
        config.audio_path, config.output_key, config.output_dir, output_formats,
        config.language, config.chinese_asr_engine, config.lyric_output_mode,
        config.tempo, config.batch_size, config.asr_batch_size,
        config.slice_min_sec, config.slice_max_sec, config.device,
    )
    rmvpe_result = _extract_pitch_curve(
        clip, sr,
        rmvpe_model_path=config.rmvpe_model_path,
        output_format_set=ctx.output_format_set,
        output_pitch_curve=config.pitch_curve,
        device=ctx.device,
        cancel_checker=cancel_checker,
    )
    rmvpe_result = _shifted_rmvpe(rmvpe_result, start)

    # Deliberately slice without the RMVPE mask: chunking must not depend on
    # whether --pitch-curve was requested, or takes would not be comparable.
    chunks = _slice_chunks(
        clip, sr,
        slicing_method=config.slicing_method,
        slice_min_sec=config.slice_min_sec,
        slice_max_sec=config.slice_max_sec,
        rmvpe_result=None,
        cancel_checker=cancel_checker,
    )
    matcher = create_lyric_matcher(
        ctx.language,
        clip_lines_text,
        engine=config.japanese_g2p_engine,
        model_dir=config.kashi_g2p_model_path,
    )

    outcomes = {}
    for engine in sorted({take.alignment_engine for take in takes}):
        logger.info(f"[Gacha] aligning clip once with {engine} ...")
        outcomes[engine] = _run_lyric_alignment(
            chunks, sr, ctx, matcher,
            asr_model_path=config.asr_model_path,
            hfa_model_dir=config.hfa_model_dir,
            tifa_model_path=config.tifa_model_path,
            alignment_engine=engine,
            phoneme_asr_model_path=config.phoneme_asr_model_path,
            pinyin_asr_model_path=config.pinyin_asr_model_path,
            japanese_asr_engine=config.japanese_asr_engine,
            japanese_g2p_engine=config.japanese_g2p_engine,
            kashi_g2p_model_path=config.kashi_g2p_model_path,
            asr_batch_size=config.asr_batch_size,
            asr_session=config.asr_session,
            cancel_checker=cancel_checker,
        )

    results: list[TakeResult] = []
    for index, take in enumerate(takes, start=1):
        if cancel_checker and cancel_checker():
            raise InterruptedError("Task was cancelled")
        ts = _t0_nstep_to_ts(take.t0, take.nsteps)
        logger.info(
            f"[Gacha] take {index}/{len(takes)} {take.label}: "
            f"seg={take.seg_threshold} r={take.seg_radius} est={take.est_threshold} "
            f"t0={take.t0} nsteps={take.nsteps}"
        )
        notes = _run_game_stage(
            chunks, sr, ctx, outcomes[take.alignment_engine],
            ts=ts,
            game_model_dir=config.game_model_dir,
            batch_size=config.batch_size,
            seg_threshold=take.seg_threshold,
            seg_radius=take.seg_radius,
            est_threshold=take.est_threshold,
            cancel_checker=cancel_checker,
        )
        for note in notes:
            note.onset += start
            note.offset += start
        ustx_path = config.output_dir / f"{config.output_key}_{take.label}.ustx"
        save_ustx(notes, ustx_path, tempo=float(config.tempo), rmvpe_result=rmvpe_result)
        results.append(
            TakeResult(
                params=take,
                notes=notes,
                ustx_path=ustx_path,
                diagnostics=_note_diagnostics(notes, start, end),
            )
        )

    if splice_range is None:
        splice_range = (start, end)
    splice_range = (
        max(start, min(float(splice_range[0]), end)),
        max(start, min(float(splice_range[1]), end)),
    )
    if splice_range[1] <= splice_range[0]:
        splice_range = (start, end)
    _write_reports(config, results, (start, end), line=line, lyric_text=lyric_window_text,
                   clip_lines_text=clip_lines_text, splice_range=splice_range)
    return results


def _t0_nstep_to_ts(t0: float, nsteps: int) -> list[float]:
    if nsteps <= 0:
        return [t0]
    step = (1 - t0) / nsteps
    return [t0 + i * step for i in range(nsteps)]


def _write_reports(
    config: GachaConfig,
    results: list[TakeResult],
    window: tuple[float, float],
    *,
    line: LyricLine | None,
    lyric_text: str,
    clip_lines_text: str,
    splice_range: tuple[float, float] | None = None,
) -> None:
    start, end = window
    splice_start, splice_end = splice_range if splice_range else (start, end)
    manifest = {
        "audio": config.audio_path,
        "output_key": config.output_key,
        "target_line": lyric_text,
        "line_index": (line.index if line is not None else None),
        "line_start": (line.start if line is not None else None),
        "line_end": (line.end if line is not None else None),
        "window_start": start,
        "window_end": end,
        "splice_start": splice_start,
        "splice_end": splice_end,
        "clip_lyrics": clip_lines_text,
        "tempo": config.tempo,
        "takes": [
            {
                **result.params.to_dict(),
                "ustx": result.ustx_path.name,
                "diagnostics": result.diagnostics,
                "notes": [
                    {
                        "onset": round(note.onset, 4),
                        "offset": round(note.offset, 4),
                        "pitch": round(float(note.pitch), 3),
                        "lyric": note.lyric,
                    }
                    for note in result.notes
                ],
            }
            for result in results
        ],
    }
    manifest_path = config.output_dir / f"{config.output_key}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ordered = sorted(
        results,
        key=lambda result: (
            result.diagnostics["flag_score"],
            result.diagnostics["note_count"],
            result.params.label,
        ),
    )
    lines = [
        f"# 句子抽卡 {Path(config.audio_path).name}",
        f"# 目标行 {lyric_text!r}  片段 {start:.3f}-{end:.3f}s  拼接区 {splice_start:.3f}-{splice_end:.3f}s  "
        f"tempo {config.tempo}",
        "# 按问题标记数排序（仅供参考，最终以试听为准）",
        "take\talign\tseg_thr\tseg_r\test_thr\tt0\tnsteps\tnotes\tmin_dur\tshort\trepeat\tgaps\tlyrics",
    ]
    for result in ordered:
        params = result.params
        diag = result.diagnostics
        lines.append(
            f"{params.label}\t{params.alignment_engine}\t{params.seg_threshold}\t"
            f"{params.seg_radius}\t{params.est_threshold}\t{params.t0}\t{params.nsteps}\t"
            f"{diag['note_count']}\t{diag['min_duration']:.3f}\t{diag['short_notes']}\t"
            f"{diag['repeated_lyric_notes']}\t{diag['internal_gaps']}\t{diag['lyric_sequence']}"
        )
    (config.output_dir / f"{config.output_key}_compare.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _ustx_note_span(note: dict, resolution: int, bpm: float) -> tuple[float, float]:
    start = note["position"] / resolution * 60.0 / bpm
    duration = note["duration"] / resolution * 60.0 / bpm
    return start, start + duration


def _seconds_to_ticks(seconds: float, resolution: int, bpm: float) -> int:
    return int(round(seconds * bpm / 60.0 * resolution))


def splice_take_into_ustx(
    base_ustx: Path,
    take_ustx: Path,
    out_ustx: Path,
    window: tuple[float, float],
) -> tuple[int, int]:
    """Replace base notes inside the window with the take's notes.

    Returns (removed, inserted). Notes straddling a window edge are replaced;
    the base file's curves and expressions are left untouched.
    """
    base = yaml.safe_load(base_ustx.read_text(encoding="utf-8"))
    take = yaml.safe_load(take_ustx.read_text(encoding="utf-8"))
    base_res = int(base["resolution"])
    base_bpm = float(base["bpm"])
    take_res = int(take["resolution"])
    take_bpm = float(take["bpm"])
    start, end = window

    base_part = base["voice_parts"][0]
    take_part = take["voice_parts"][0]
    kept = []
    removed = 0
    for note in base_part["notes"]:
        note_start, note_end = _ustx_note_span(note, base_res, base_bpm)
        if note_end <= start or note_start >= end:
            kept.append(note)
        else:
            removed += 1

    inserted = 0
    for note in take_part["notes"]:
        note_start, note_end = _ustx_note_span(note, take_res, take_bpm)
        if note_end <= start or note_start >= end:
            continue
        clipped_start = max(note_start, start)
        clipped_end = min(note_end, end)
        if clipped_end - clipped_start <= 0.001:
            continue
        clone = dict(note)
        position = _seconds_to_ticks(clipped_start, base_res, base_bpm)
        clone["position"] = position
        clone["duration"] = max(10, _seconds_to_ticks(clipped_end, base_res, base_bpm) - position)
        kept.append(clone)
        inserted += 1

    kept.sort(key=lambda note: note["position"])
    base_part["notes"] = kept
    base_part["duration"] = max(
        480, max((note["position"] + note["duration"] for note in kept), default=0)
    )
    out_ustx.parent.mkdir(parents=True, exist_ok=True)
    out_ustx.write_text(
        yaml.safe_dump(base, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return removed, inserted


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
