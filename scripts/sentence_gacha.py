"""Rerun one lyric line with different decoding params and collect candidates.

    python scripts/sentence_gacha.py song.flac --lyrics-file song.lrc \
        --line 12 --takes 8 --alignments tifa,hfa -o gacha/ \
        --language ja --lyric-format kana --device dml --tempo 71.7 --pitch-curve

Splice the chosen take back into a full-song export:

    python scripts/sentence_gacha.py --apply gacha/ --take tifa-03 \
        --base song.ustx -o song_fixed.ustx
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.auto_lyric_cli import (  # noqa: E402
    DEFAULT_LYRIC_FORMAT,
    LANGUAGE_CHOICES,
    LYRIC_FORMAT_CHOICES,
    LYRIC_FORMATS_BY_LANGUAGE,
    _load_settings_defaults,
)

logger = logging.getLogger(__name__)


def _parse_floats(text: str | None) -> list[float] | None:
    if not text:
        return None
    return [float(part) for part in text.replace(" ", "").split(",") if part]


def _parse_ints(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(part) for part in text.replace(" ", "").split(",") if part]


def _parse_engines(text: str | None, fallback: str) -> list[str]:
    raw = text or fallback
    engines = [part.strip().lower() for part in raw.split(",") if part.strip()]
    for engine in engines:
        if engine not in ("tifa", "hfa"):
            raise ValueError(f"unknown alignment engine {engine!r} (valid: tifa, hfa)")
    if not engines:
        raise ValueError("at least one alignment engine is required")
    return engines


def build_argparser(defaults: dict) -> argparse.ArgumentParser:
    device_utils = importlib.import_module("inference.device_utils")
    parser = argparse.ArgumentParser(
        prog="sentence_gacha",
        description="Rerun one lyric line with different GAME/alignment params (sentence gacha).",
    )
    parser.add_argument("audio", nargs="?", type=Path, help="Audio file to process")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: <audio>_gacha next to the audio)")

    target = parser.add_argument_group("target")
    target.add_argument("--lyrics-file", type=Path, default=None,
                        help="UTF-8 lyrics file, plain text or .lrc")
    target.add_argument("--lyrics", default="", help="Lyrics text (alternative to --lyrics-file)")
    target.add_argument("--line", type=int, default=None, help="1-based lyric line number")
    target.add_argument("--sentence", default=None, help="Match the lyric line by its text")
    target.add_argument("--start", type=float, default=None, help="Window start (s), overrides LRC")
    target.add_argument("--end", type=float, default=None, help="Window end (s), overrides LRC")
    target.add_argument("--pad-before", type=float, default=0.5, help="Lead-in before the line (s)")
    target.add_argument("--pad-after", type=float, default=0.3, help="Tail after the line (s)")
    target.add_argument("--list-lines", action="store_true", help="Print the parsed lyric lines and exit")

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--strategy", choices=("random", "grid", "repeat"), default="random",
                          help="How to pick decoding params (default: random)")
    sampling.add_argument("--takes", type=int, default=8,
                          help="Param sets per alignment engine (default: 8)")
    sampling.add_argument("--seed", type=int, default=0, help="Random sampling seed")
    sampling.add_argument("--alignments", default=None,
                          help="Comma list of alignment engines to sweep (default: the --aligner value)")
    sampling.add_argument("--seg-threshold", default=None, help="Comma list (grid mode)")
    sampling.add_argument("--seg-radius", default=None, help="Comma list (grid mode)")
    sampling.add_argument("--est-threshold", default=None, help="Comma list (grid mode)")
    sampling.add_argument("--t0", default=None, help="Comma list (grid mode)")
    sampling.add_argument("--nsteps", default=None, help="Comma list (grid mode)")

    parser.add_argument("--language", choices=LANGUAGE_CHOICES, default="zh")
    parser.add_argument("--lyric-format", choices=LYRIC_FORMAT_CHOICES, default=None)
    parser.add_argument("--chinese-asr", choices=("pinyin", "qwen"), default=defaults["chinese_asr_engine"])
    parser.add_argument("--japanese-asr", choices=("romaji", "qwen"), default=defaults["japanese_asr_engine"])
    parser.add_argument("--japanese-g2p", choices=("kashi-g2p-onnx", "pyopenjtalk"),
                        default=defaults.get("japanese_g2p_engine", "kashi-g2p-onnx"))
    parser.add_argument("--aligner", choices=("tifa", "hfa"), default=defaults["alignment_engine"])
    parser.add_argument("--slicing", choices=("smart", "heuristic", "default", "grid"), default="smart")
    parser.add_argument("--min-seconds", type=float, default=defaults["slice_min_sec"])
    parser.add_argument("--max-seconds", type=float, default=defaults["slice_max_sec"])
    parser.add_argument("--device", choices=list(device_utils.RUNTIME_DEVICE_CHOICES),
                        default=device_utils.default_runtime_device())
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--asr-batch-size", type=int, default=defaults["asr_batch_size"])
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--pitch-curve", action=argparse.BooleanOptionalAction,
                        default=defaults["output_pitch_curve"])
    parser.add_argument("--game-model", default=defaults["game_model"])
    parser.add_argument("--hfa-model", default=defaults["hfa_model"])
    parser.add_argument("--tifa-model", default=defaults["tifa_model"])
    parser.add_argument("--kashi-g2p-model", default=defaults["kashi_g2p_model"])
    parser.add_argument("--asr-model", default=defaults["asr_model"])
    parser.add_argument("--phoneme-asr-model", default=defaults["phoneme_asr_model"])
    parser.add_argument("--pinyin-asr-model", default=defaults["pinyin_asr_model"])
    parser.add_argument("--rmvpe-model", default=defaults["rmvpe_model"])
    parser.add_argument("--seg-threshold-default", type=float, default=defaults["seg_threshold"],
                        help=argparse.SUPPRESS)

    apply_group = parser.add_argument_group("apply")
    apply_group.add_argument("--apply", type=Path, default=None,
                             help="Gacha output dir: splice a chosen take into --base")
    apply_group.add_argument("--take", default=None, help="Take label to splice (e.g. tifa-03)")
    apply_group.add_argument("--base", type=Path, default=None, help="Full-song USTX to patch")
    return parser


def _run_apply(args) -> int:
    from inference.pipeline.sentence_gacha import load_manifest, splice_take_into_ustx

    if not args.take:
        logger.error("--apply needs --take")
        return 2
    if args.base is None or not args.base.is_file():
        logger.error("--apply needs an existing --base USTX")
        return 2
    if args.output_dir is None:
        logger.error("--apply needs -o/--output-dir for the patched USTX")
        return 2
    manifest_path = args.apply / "manifest.json"
    if not manifest_path.is_file():
        legacy = sorted(args.apply.glob("*_manifest.json"))
        if len(legacy) != 1:
            logger.error(f"manifest.json not found in {args.apply}")
            return 2
        manifest_path = legacy[0]
    manifest = load_manifest(manifest_path)
    entry = next((item for item in manifest["takes"] if item["take"] == args.take), None)
    if entry is None:
        labels = ", ".join(item["take"] for item in manifest["takes"])
        logger.error(f"take {args.take!r} not found (available: {labels})")
        return 2
    take_path = args.apply / entry["ustx"]
    if not take_path.is_file():
        logger.error(f"take file missing: {take_path}")
        return 2
    window = (
        float(manifest.get("splice_start", manifest["window_start"])),
        float(manifest.get("splice_end", manifest["window_end"])),
    )
    out_path = args.output_dir
    if out_path.suffix.lower() != ".ustx":
        out_path = out_path / f"{args.base.stem}_patched.ustx"
    removed, inserted = splice_take_into_ustx(args.base, take_path, out_path, window)
    logger.info(
        f"Spliced {args.take} into {out_path} "
        f"({window[0]:.3f}-{window[1]:.3f}s, removed {removed}, inserted {inserted})."
    )
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    defaults = _load_settings_defaults()
    parser = build_argparser(defaults)
    args = parser.parse_args(argv)

    if args.apply is not None:
        return _run_apply(args)

    if args.audio is None:
        parser.error("an audio file is required unless --apply is used")
    if not args.audio.is_file():
        logger.error(f"audio not found: {args.audio}")
        return 2
    if args.lyric_format is None:
        args.lyric_format = defaults.get(
            f"lyric_output_mode_{args.language}", DEFAULT_LYRIC_FORMAT[args.language]
        )
    if args.lyric_format not in LYRIC_FORMATS_BY_LANGUAGE[args.language]:
        valid = ", ".join(LYRIC_FORMATS_BY_LANGUAGE[args.language])
        logger.error(f"lyric format {args.lyric_format!r} is not valid for {args.language} (valid: {valid})")
        return 2
    if args.nsteps is not None and any(int(v) <= 0 for v in _parse_ints(args.nsteps) or []):
        logger.error("--nsteps must be positive")
        return 2
    if args.t0 is not None and any(not 0.0 <= float(v) < 1.0 for v in _parse_floats(args.t0) or []):
        logger.error("--t0 values must satisfy 0 <= t0 < 1")
        return 2

    raw_lyrics = ""
    if args.lyrics_file is not None:
        if not args.lyrics_file.is_file():
            logger.error(f"lyrics file not found: {args.lyrics_file}")
            return 2
        raw_lyrics = args.lyrics_file.read_text(encoding="utf-8")
    elif args.lyrics:
        raw_lyrics = args.lyrics
    if not raw_lyrics.strip():
        logger.error("give --lyrics-file or --lyrics")
        return 2

    from inference.io.audio_io import load_audio
    from inference.pipeline.sentence_gacha import (
        SAMPLE_RATE,
        GachaConfig,
        build_takes,
        clip_lyrics_for_window,
        parse_lyric_lines,
        resolve_window,
        select_line,
    )

    lines = parse_lyric_lines(raw_lyrics)
    if args.list_lines:
        for line in lines:
            stamp = f"{line.start:8.3f}-{line.end if line.end is not None else float('nan'):8.3f}" \
                if line.start is not None else "      -        -"
            print(f"{line.index + 1:3d}  {stamp}  {line.text}")
        return 0
    if not lines:
        logger.error("no lyric lines found")
        return 2

    try:
        line = None
        if args.line is not None or args.sentence:
            line = select_line(lines, line_number=args.line, sentence=args.sentence)
        waveform, _ = load_audio(args.audio, SAMPLE_RATE)
        total_duration = len(waveform) / float(SAMPLE_RATE)
        window = resolve_window(
            lines, line=line, start=args.start, end=args.end,
            pad_before=args.pad_before, pad_after=args.pad_after,
            total_duration=total_duration,
        )
        if args.start is not None and args.end is not None:
            splice_range = (min(args.start, args.end), max(args.start, args.end))
        elif line is not None and line.start is not None and line.end is not None:
            splice_range = (line.start, line.end)
        else:
            splice_range = window
        engines = _parse_engines(args.alignments, args.aligner)
        if args.language == "yue" and "hfa" in engines:
            raise ValueError("Cantonese ('yue') is not supported by HubertFA (HFA); use --alignments tifa")
        takes = build_takes(
            engines=engines,
            strategy=args.strategy,
            takes=args.takes,
            seed=args.seed,
            base={
                "seg_threshold": args.seg_threshold_default,
                "seg_radius": defaults["seg_radius"],
                "est_threshold": defaults["est_threshold"],
                "t0": defaults["t0"],
                "nsteps": defaults["nsteps"],
            },
            seg_thresholds=_parse_floats(args.seg_threshold),
            seg_radii=_parse_floats(args.seg_radius),
            est_thresholds=_parse_floats(args.est_threshold),
            t0_values=_parse_floats(args.t0),
            nsteps_values=_parse_ints(args.nsteps),
        )
    except (ValueError, FileNotFoundError) as error:
        logger.error(str(error))
        return 2

    logger.info(
        f"Clip {window[0]:.3f}-{window[1]:.3f}s, splice region "
        f"{splice_range[0]:.3f}-{splice_range[1]:.3f}s"
    )
    clip_lines_text = clip_lyrics_for_window(
        lines, window[0], window[1], fallback=line.text if line is not None else ""
    )
    if line is None:
        lyric_text = clip_lines_text
        logger.warning("No --line/--sentence: using every line that overlaps the window.")
    else:
        lyric_text = line.text
        # Keep only the target line's text when it is the sole line in the clip.
        if line.text not in clip_lines_text.splitlines():
            clip_lines_text = line.text

    if args.output_dir is None:
        args.output_dir = args.audio.with_name(f"{args.audio.stem}_gacha")
    output_key = args.audio.stem
    if len(engines) > 1:
        logger.info(
            f"Sweeping {len(engines)} alignment engines x {args.takes} param set(s) "
            f"= {len(takes)} take(s); alignment runs once per engine."
        )

    import application.pipeline as application_pipeline
    from inference.pipeline.sentence_gacha import run_gacha

    config = GachaConfig(
        audio_path=str(args.audio),
        output_dir=Path(args.output_dir),
        output_key=output_key,
        language=args.language,
        lyric_output_mode=args.lyric_format,
        device=args.device,
        game_model_dir=args.game_model,
        hfa_model_dir=args.hfa_model,
        asr_model_path=args.asr_model,
        tifa_model_path=args.tifa_model,
        tempo=args.tempo,
        batch_size=args.batch_size,
        asr_batch_size=args.asr_batch_size,
        slicing_method=args.slicing,
        slice_min_sec=args.min_seconds,
        slice_max_sec=args.max_seconds,
        pitch_curve=args.pitch_curve,
        rmvpe_model_path=args.rmvpe_model,
        phoneme_asr_model_path=args.phoneme_asr_model,
        pinyin_asr_model_path=args.pinyin_asr_model,
        chinese_asr_engine=args.chinese_asr,
        japanese_asr_engine=args.japanese_asr,
        japanese_g2p_engine=args.japanese_g2p,
        kashi_g2p_model_path=args.kashi_g2p_model,
    )

    session = application_pipeline.open_asr_session()
    config.asr_session = session
    try:
        results = run_gacha(
            config, takes, window,
            lyric_window_text=lyric_text,
            clip_lines_text=clip_lines_text,
            line=line,
            splice_range=splice_range,
            preloaded_audio=(waveform, SAMPLE_RATE),
        )
    finally:
        session.close()

    for result in results:
        diag = result.diagnostics
        logger.info(
            f"{result.params.label}: {diag['note_count']} note(s), "
            f"min {diag['min_duration']:.3f}s, flags {diag['flag_score']} "
            f"[{diag['lyric_sequence']}]"
        )
    logger.info(f"Outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
