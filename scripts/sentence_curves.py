"""Write pitch and voicing reference curves into an OpenUtau USTX.

Hand-drawing notes is easier than trusting a decoder on hard passages, but you
still need to see pitch and where the voice actually sounds. This writes both
as project *masked* curves on top of an existing USTX export:

    python scripts/sentence_curves.py song.flac --base song.ustx \
        --lyrics-file song.lrc --line 10 -o song_curves.ustx \
        --device dml

Open it in OpenUtau and pick "pitch (midi x100)" (6500 = note 65.00) or
"voiced (gaps = silence)" in the expression selector; gaps mean silence.
``--start/--end`` targets an arbitrary span instead of an LRC line; with no
target the whole song is processed.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import replace
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.auto_lyric_cli import _load_settings_defaults  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_curves(text: str | None) -> list[str]:
    if not text:
        return ["f0", "vuv"]
    names = [part.strip().lower() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in ("f0", "vuv")]
    if unknown:
        raise ValueError(f"unknown curve {', '.join(unknown)} (valid: f0, vuv)")
    return names


def build_argparser(defaults: dict) -> argparse.ArgumentParser:
    device_utils = importlib.import_module("inference.device_utils")
    parser = argparse.ArgumentParser(
        prog="sentence_curves",
        description="Add absolute pitch (f0) and voiced (vuv) curves to an OpenUtau USTX.",
    )
    parser.add_argument("audio", type=Path, help="Vocal audio file")
    parser.add_argument("--base", type=Path, required=True, help="USTX export to add curves to")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output USTX (default: <base>_curves.ustx)")

    target = parser.add_argument_group("target")
    target.add_argument("--lyrics-file", type=Path, default=None, help="UTF-8 lyrics file (.lrc or plain)")
    target.add_argument("--line", type=int, default=None, help="1-based lyric line number")
    target.add_argument("--sentence", default=None, help="Match the lyric line by text")
    target.add_argument("--start", type=float, default=None, help="Window start (s), overrides LRC")
    target.add_argument("--end", type=float, default=None, help="Window end (s), overrides LRC")
    target.add_argument("--pad-before", type=float, default=0.0, help="Lead-in before the line (s)")
    target.add_argument("--pad-after", type=float, default=0.0, help="Tail after the line (s)")
    target.add_argument("--list-lines", action="store_true", help="Print the parsed lyric lines and exit")

    parser.add_argument("--curves", default=None, help="Comma list of curves to write (default: f0,vuv)")
    parser.add_argument("--track", type=int, default=0, help="Track number to attach the curves to")
    parser.add_argument("--device", choices=list(device_utils.RUNTIME_DEVICE_CHOICES),
                        default=device_utils.default_runtime_device())
    parser.add_argument("--rmvpe-model", default=defaults["rmvpe_model"])
    parser.add_argument("--no-replace", action="store_true",
                        help="Only add points outside the target window (keep existing points inside it)")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    defaults = _load_settings_defaults()
    parser = build_argparser(defaults)
    args = parser.parse_args(argv)

    from inference.pipeline.sentence_gacha import parse_lyric_lines, resolve_window, select_line

    if not args.audio.is_file():
        logger.error(f"audio not found: {args.audio}")
        return 2
    if not args.base.is_file():
        logger.error(f"base USTX not found: {args.base}")
        return 2
    out_path = args.output or args.base.with_name(f"{args.base.stem}_curves.ustx")
    if out_path.resolve() == args.base.resolve():
        logger.error("output would overwrite the base USTX; pick another path")
        return 2
    try:
        curve_names = _parse_curves(args.curves)
    except ValueError as error:
        logger.error(str(error))
        return 2

    lines = []
    if args.lyrics_file is not None:
        if not args.lyrics_file.is_file():
            logger.error(f"lyrics file not found: {args.lyrics_file}")
            return 2
        lines = parse_lyric_lines(args.lyrics_file.read_text(encoding="utf-8"))
    if args.list_lines:
        for line in lines:
            stamp = f"{line.start:8.3f}-{line.end:8.3f}" if line.start is not None and line.end is not None else "      -        -"
            print(f"{line.index + 1:3d}  {stamp}  {line.text}")
        return 0

    from inference.io.audio_io import load_audio

    waveform, sr = load_audio(args.audio, 44100)
    total_duration = len(waveform) / 44100.0
    try:
        line = None
        if args.line is not None or args.sentence:
            if not lines:
                raise ValueError("--line/--sentence needs --lyrics-file")
            line = select_line(lines, line_number=args.line, sentence=args.sentence)
        if args.start is None and args.end is None and line is None:
            window = (0.0, total_duration)
        else:
            window = resolve_window(
                lines, line=line, start=args.start, end=args.end,
                pad_before=args.pad_before, pad_after=args.pad_after,
                total_duration=total_duration,
            )
    except (ValueError, FileNotFoundError) as error:
        logger.error(str(error))
        return 2
    start, end = window
    logger.info(f"[Curves] window {start:.3f}-{end:.3f}s")

    clip = waveform[int(round(start * 44100)):int(round(end * 44100))]
    if clip.size < int(0.2 * 44100):
        logger.error("window is too short")
        return 2

    from inference.pipeline.auto_lyric_hybrid import _extract_pitch_curve
    from inference.pipeline.ustx_curves import build_f0_runs, build_vuv_runs, inject_curves

    rmvpe = _extract_pitch_curve(
        clip, 44100,
        rmvpe_model_path=args.rmvpe_model,
        output_format_set=["ustx"],
        output_pitch_curve=True,
        device=args.device,
        cancel_checker=None,
    )
    if rmvpe is None:
        logger.error("RMVPE did not produce a pitch curve")
        return 2
    rmvpe = replace(rmvpe, time_offset=start)

    with args.base.open("r", encoding="utf-8") as f:
        project = yaml.safe_load(f) or {}
    tempos = project.get("tempos") or [{"position": 0, "bpm": project.get("bpm") or 120}]
    resolution = int(project.get("resolution") or 480)
    part = (project.get("voice_parts") or [{}])[0]
    part_offset = int(part.get("position") or 0)

    curves = {}
    if "f0" in curve_names:
        curves["f0"] = build_f0_runs(rmvpe, tempos, resolution, part_offset)
    if "vuv" in curve_names:
        curves["vuv"] = build_vuv_runs(rmvpe, tempos, resolution, part_offset)

    summary = inject_curves(
        args.base, out_path, curves,
        window_seconds=None if args.no_replace else (start, end),
        track=args.track,
        preserve_existing=args.no_replace,
    )
    for abbr, info in summary["curves"].items():
        logger.info(
            f"[Curves] {abbr}: {info['runs']} run(s), {info['values']} value(s) over ticks "
            f"{info['x_range'][0]}-{info['x_range'][1]}, y {info['y_range'][0]}-{info['y_range'][1]}"
        )
    logger.info(f"[Curves] output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
