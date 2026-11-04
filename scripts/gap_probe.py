"""Probe intra-phrase note gaps and measure the evidence the repair detector needs.

For every gap between two consecutive notes in a generated USTX, compare the
short-time level of each supplied audio track and the dry track's pitch
continuity, then classify the gap. This is the P0 measurement step of
docs/low-confidence-note-repair.md: it quantifies how many "missing notes" are
actually stripped-but-still-pitched rather than genuine rests.

Usage:
    python/python.exe scripts/gap_probe.py \
        --ustx "song.ustx" --dry "vocal_dry.flac" \
        --track vocals="vocal.flac" --track mix="full.flac" \
        --json gaps.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from inference.API.rmvpe_api import RmvpeTranscriber  # noqa: E402
from inference.io.audio_io import load_audio  # noqa: E402


class _UstxLoader(yaml.SafeLoader):
    """YAML 1.2 style loader: OpenUtau writes lyric `no` unquoted.

    PyYAML follows the YAML 1.1 bool resolver, so a 「の」 mora read back from an
    OpenUtau-saved project becomes Python False and looks like a missing lyric.
    """


# Class attribute is shared with SafeLoader, so replace it instead of mutating.
_UstxLoader.yaml_implicit_resolvers = {
    key: [item for item in resolvers if item[0] != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}

TICKS_PER_BEAT = 480
SR = 44100
FRAME_SEC = 0.025
HOP_SEC = 0.010
MIN_GAP_SEC = 0.06
MAX_GAP_SEC = 1.5
FLANK_SEC = 0.10
CORE_MARGIN_SEC = 0.02
MIN_CORE_SEC = 0.05
# Detector thresholds from the design doc.
STRIPPED_LEVEL_DROP_DB = 8.0
SILENT_DB = -60.0
# Gates set from the measurement run on 月夜の君と空の上: gap interiors whose
# vocal-stem voiced fraction is <=0.07 are true rests, the next cluster starts at
# 0.13. Dry F0 agrees with the stem within ~35 cents when it still holds the note.
REFERENCE_VOICED_FRACTION = 0.10
STRIPPED_VOICED_FRACTION = 0.5
AGREED_CENTS = 35.0


def _tick_to_sec(ticks, bpm: float) -> float:
    return float(ticks) * 60.0 / (float(bpm) * TICKS_PER_BEAT)


def load_notes(ustx_path: pathlib.Path) -> tuple[list[dict], float]:
    doc = yaml.load(ustx_path.read_text(encoding="utf-8"), Loader=_UstxLoader)
    bpm = float(doc.get("bpm") or 120.0)
    notes: list[dict] = []
    for part in doc.get("voice_parts") or []:
        offset = _tick_to_sec(part.get("position", 0), bpm)
        for note in part.get("notes") or []:
            notes.append(
                {
                    "start": offset + _tick_to_sec(note.get("position", 0), bpm),
                    "duration": _tick_to_sec(note.get("duration", 0), bpm),
                    "tone": int(note.get("tone", 0) or 0),
                    "lyric": str(note.get("lyric", "")),
                }
            )
    notes.sort(key=lambda item: item["start"])
    return notes, bpm


def rms_envelope(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(FRAME_SEC * SR))
    hop = max(1, int(HOP_SEC * SR))
    count = 0 if waveform.size < frame else 1 + (waveform.size - frame) // hop
    if count <= 0:
        return np.zeros((0,)), np.zeros((0,))
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    frames = waveform[index]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    db = 20.0 * np.log10(rms + 1e-9)
    times = (np.arange(count) * hop + frame / 2.0) / SR
    return times, db


def _r(value, digits: int = 1):
    return None if value is None else round(float(value), digits)


def _window_stats(times, db, start: float, end: float) -> dict:
    mask = (times >= start) & (times < max(end, start + HOP_SEC))
    if not mask.any():
        return {"median_db": None, "max_db": None}
    values = db[mask]
    return {"median_db": float(np.median(values)), "max_db": float(np.max(values))}


def _window_indices(step: float, start: float, end: float, size: int) -> tuple[int, int]:
    # The probe window can reach before t=0; negative indices would wrap.
    lo = max(0, int(start / step))
    hi = min(size, int(np.ceil(end / step)))
    return lo, hi


def _pitch_stats(entry, step: float, start: float, end: float) -> dict:
    pitch, voiced = entry.midi_pitch, entry.voiced_mask
    lo, hi = _window_indices(step, start, end, voiced.size)
    window = voiced[lo:hi]
    if window.size == 0:
        return {"voiced_fraction": 0.0, "median_jump_cents": None, "pitch_span_semitones": None}
    fraction = float(np.mean(window))
    midi = np.where(window, pitch[lo:hi], np.nan)
    jumps = np.abs(np.diff(midi)) * 100.0
    jumps = jumps[np.isfinite(jumps)]
    span = None
    if np.any(window):
        voiced_midi = midi[np.isfinite(midi)]
        if voiced_midi.size:
            span = float(voiced_midi.max() - voiced_midi.min())
    return {
        "voiced_fraction": fraction,
        "median_jump_cents": float(np.median(jumps)) if jumps.size else None,
        "pitch_span_semitones": span,
    }


def _pitch_compare(dry, ref, step: float, start: float, end: float) -> dict:
    """How far the dry F0 sits from the reference stem where both are voiced.

    Small means the dry still holds the note's pitch and a gain-only repair is
    enough; large or absent means the dry pitch is interpolation over silence.
    """
    lo, hi = _window_indices(step, start, end, dry.voiced_mask.size)
    hi = min(hi, ref.voiced_mask.size)
    dry_v, ref_v = dry.voiced_mask[lo:hi], ref.voiced_mask[lo:hi]
    if dry_v.size == 0:
        return {"agree_fraction": 0.0, "median_agree_cents": None}
    both = dry_v & ref_v
    if not both.any():
        return {"agree_fraction": 0.0, "median_agree_cents": None}
    aligned = slice(lo, hi)
    cents = np.abs(dry.midi_pitch[aligned][both] - ref.midi_pitch[aligned][both]) * 100.0
    return {
        "agree_fraction": float(np.mean(both)),
        "median_agree_cents": float(np.median(cents)),
    }


def classify(gap: dict) -> str:
    core = gap["core"]
    dry = core["tracks"]["dry"]
    flanks = gap["flank_db"]
    ref_label = gap["reference_track"]
    reference = None if ref_label is None else core["tracks"][ref_label]["median_db"]
    pitch = core["pitch"]
    ref_pitch = (core["by_label"] or {}).get(ref_label)
    if dry["median_db"] is not None and dry["median_db"] < SILENT_DB and (
        reference is None or reference < SILENT_DB
    ):
        return "genuine_rest"
    if reference is None or dry["median_db"] is None:
        return "unknown"
    drop = reference - dry["median_db"]
    if drop < STRIPPED_LEVEL_DROP_DB:
        if flanks is not None and flanks - dry["median_db"] >= STRIPPED_LEVEL_DROP_DB:
            return "local_dip"
        return "levelled"
    if ref_pitch is None:
        return "no_pitch_ref"
    if ref_pitch["voiced_fraction"] < REFERENCE_VOICED_FRACTION:
        return "genuine_rest"
    pitch_kept = (
        ref_pitch.get("median_agree_cents") is not None
        and ref_pitch["median_agree_cents"] <= AGREED_CENTS
        and ref_pitch.get("agree_fraction", 0.0) >= REFERENCE_VOICED_FRACTION
        and pitch["voiced_fraction"] >= STRIPPED_VOICED_FRACTION
    )
    return "stripped_note" if pitch_kept else "low_gain"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ustx", required=True)
    parser.add_argument("--dry", required=True, help="Track the pipeline actually consumed.")
    parser.add_argument("--track", action="append", default=[], help="label=path extra tracks.")
    parser.add_argument(
        "--reference",
        default="",
        help="Vocal stem used as the loudness/pitch reference. Default: first --track.",
    )
    parser.add_argument("--rmvpe", default="models/RMVPE")
    parser.add_argument(
        "--pitch-track",
        action="append",
        default=[],
        help="Extra track labels to run RMVPE on (in addition to dry and --reference).",
    )
    parser.add_argument("--device", default="dml")
    parser.add_argument("--json", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    notes, bpm = load_notes(pathlib.Path(args.ustx))
    if not notes:
        print("no notes parsed")
        return 1

    tracks = {"dry": args.dry}
    for item in args.track:
        label, _, path = item.partition("=")
        tracks[label] = path

    print(f"notes={len(notes)} bpm={bpm}")
    envelopes = {}
    for label, path in tracks.items():
        waveform, sr = load_audio(path, SR)
        if sr != SR:
            raise SystemExit(f"{label}: unexpected sample rate {sr}")
        envelopes[label] = rms_envelope(np.asarray(waveform, dtype=np.float32))
        print(f"track {label}: {len(waveform) / SR:.1f}s")

    rmvpe = RmvpeTranscriber(args.rmvpe, device=args.device)
    reference = args.reference or None
    if reference is None:
        reference = "vocals" if "vocals" in tracks else next(
            (label for label in tracks if label != "dry"), None
        )
    if reference is not None and reference not in tracks:
        raise SystemExit(f"reference track {reference!r} not loaded")
    pitches = {}
    wanted = {"dry"} | set(args.pitch_track)
    if reference is not None:
        wanted.add(reference)
    for label in [lab for lab in tracks if lab in wanted]:
        wave, wave_sr = load_audio(tracks[label], SR)
        entry = rmvpe.infer(np.asarray(wave, dtype=np.float32), wave_sr)
        pitches[label] = entry
        print(f"rmvpe {label}: frames={entry.midi_pitch.size} step={entry.time_step_seconds:.4f}s")
    step = pitches["dry"].time_step_seconds

    def _stats(win: tuple[float, float]) -> dict:
        out = {
            "tracks": {
                label: _window_stats(times, db, *win) for label, (times, db) in envelopes.items()
            },
            "pitch": _pitch_stats(pitches["dry"], step, *win),
            "by_label": {},
        }
        for label, entry in pitches.items():
            if label == "dry":
                continue
            item = _pitch_stats(entry, step, *win)
            item.update(_pitch_compare(pitches["dry"], entry, step, *win))
            out["by_label"][label] = item
        return out

    gaps = []
    for index in range(len(notes) - 1):
        left, right = notes[index], notes[index + 1]
        if left["tone"] <= 0 or right["tone"] <= 0:
            continue
        if left["lyric"] in {"", "-", "r", "R", "SP"} or right["lyric"] in {"", "-", "r", "R", "SP"}:
            continue
        start = left["start"] + left["duration"]
        end = right["start"]
        gap_sec = end - start
        if not (MIN_GAP_SEC <= gap_sec <= MAX_GAP_SEC):
            continue
        window = (start - FLANK_SEC, end + FLANK_SEC)
        core_lo, core_hi = start + CORE_MARGIN_SEC, end - CORE_MARGIN_SEC
        # For a 0.1 s gap the flank window is almost entirely the two neighbouring
        # notes, so it cannot tell a rest from a stripped mora. Keep the interior
        # separate and only fall back to it when the gap is too short to have one.
        core = (core_lo, core_hi) if core_hi - core_lo >= MIN_CORE_SEC else window

        gap = {
            "index": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "gap_sec": round(gap_sec, 3),
            "core_sec": round(core_hi - core_lo, 3),
            "has_core": core is not window,
            "lyric_before": left["lyric"],
            "lyric_after": right["lyric"],
            "tracks": _stats(window)["tracks"],
            "flank_db": float(
                np.median(
                    [
                        value
                        for value in (
                            _window_stats(*envelopes["dry"], max(start - FLANK_SEC, 0.0), start)["median_db"],
                            _window_stats(*envelopes["dry"], end, end + FLANK_SEC)["median_db"],
                        )
                        if value is not None
                    ]
                )
                or None
            ),
            "reference_track": reference,
            "core": _stats(core),
        }
        gap["kind"] = classify(gap)
        ref_db = gap["core"]["tracks"][reference]["median_db"] if reference else None
        dry_db = gap["core"]["tracks"]["dry"]["median_db"]
        gap["drop_db"] = (
            None if ref_db is None or dry_db is None else round(ref_db - dry_db, 1)
        )
        gaps.append(gap)

    counts: dict[str, int] = {}
    for gap in gaps:
        counts[gap["kind"]] = counts.get(gap["kind"], 0) + 1
    print(f"gaps={len(gaps)} kinds={json.dumps(counts)}")
    # Show the repair candidates first; chronological order buries them in rests.
    priority = ["stripped_note", "low_gain", "local_dip", "no_pitch_ref", "levelled", "genuine_rest"]
    ordered = sorted(
        gaps,
        key=lambda g: (
            priority.index(g["kind"]) if g["kind"] in priority else len(priority),
            -(g["drop_db"] if g["drop_db"] is not None else -999.0),
        ),
    )
    for gap in ordered[: args.limit or len(ordered)]:
        core = gap["core"]
        print(
            json.dumps(
                {
                    "t": f"{gap['start']:.2f}-{gap['end']:.2f}",
                    "gap": gap["gap_sec"],
                    "core": gap["core_sec"],
                    "lyr": f"{gap['lyric_before']}>{gap['lyric_after']}",
                    "kind": gap["kind"],
                    "dry_db": _r(core["tracks"]["dry"]["median_db"]),
                    "ref_db": _r((core["tracks"].get(gap["reference_track"]) or {}).get("median_db")),
                    "drop": gap["drop_db"],
                    "flank_db": _r(gap["flank_db"]),
                    "others": {
                        label: _r(stats["median_db"])
                        for label, stats in core["tracks"].items()
                        if label not in {"dry", gap["reference_track"]}
                    },
                    "dry_voiced": _r(core["pitch"]["voiced_fraction"], 2),
                    "stem": {
                        label: {
                            "voiced": _r(item["voiced_fraction"], 2),
                            "cents": _r(item.get("median_agree_cents")),
                        }
                        for label, item in core["by_label"].items()
                    },
                },
                ensure_ascii=True,
            )
        )

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"bpm": bpm, "gaps": gaps}, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
