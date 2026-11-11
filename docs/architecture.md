# Vocal2Midi Architecture

This document describes the current structure of the Vocal2Midi repository: an ONNX-first
inference stack with a Fluent (PySide6) desktop shell and a set of batch/sentence CLI tools.

User-facing setup, model folders, device rules and CLI flags live in [README.md](../README.md).
This document covers the internal layering and data flow.

## Design goals

The codebase is organized to keep UI concerns, application orchestration, and inference runtimes separate:

```text
gui -> application -> inference
scripts -> application / inference
```

The intended dependency direction is one-way:

- `gui/` collects settings and shows progress; it does not reach into runtime details when an
  application-layer entrypoint exists.
- `application/` coordinates jobs, validates configuration, and maps failures onto typed errors.
- `inference/` owns transcription, alignment, pitch extraction, slicing, quantization, and export.
- `scripts/` are thin CLI frontends over the same application/inference entrypoints.

## Top-level structure

Representative modules (not exhaustive):

```text
application/
  config.py            PipelineConfig dataclass + slice-bound validation
  pipeline.py          run_auto_lyric_job(): model-path validation, error mapping
  sentence_jobs.py     sentence-repair jobs shared by GUI and CLI
  exceptions.py        Vocal2MidiError / ModelNotFoundError / CancellationError / ...

gui/
  fluent_main.py       Windows Fluent shell
  macos_main.py        macOS shell
  auto_lyric_view.py   main transcription page
  sentence_repair_view.py  hard-sentence gacha / reference-curve page
  model_config_view.py engine + model-path selection
  global_settings_view.py
  fluent_worker.py     background job thread

inference/
  API/                 asr_api, game_api, hfa_api, lfa_api, rmvpe_api,
                       slicer_api, ustx_api, vsqx_api
  TiFA/                default forced aligner (runtime, aligner, g2p, textgrid)
  HubertFA/            optional forced aligner
  LyricFA/             lyric matching / normalization helpers
  game/                GAME note extraction + word/note boundary alignment
  qwen3asr_dml/        Qwen3-ASR ONNX encoder + llama.cpp decoder
  romaji_asr/          Japanese mora-level ASR
  pinyin_asr/          Chinese pinyin-level ASR
  kashi_g2p_ja/        Japanese ONNX G2P (beam-search candidates for TiFA)
  ja_onnx_g2p.py       legacy Japanese ONNX G2P entry
  t2s_utils.py         traditional/simplified Chinese conversion
  slicer/              RMS / smart slicing
  quant/               quantization.py, smart_quantize.py
  io/                  audio_io (load_audio), note_io (MIDI/TXT/CSV)
  pipeline/
    auto_lyric_hybrid.py  end-to-end workflow
    lyric_alignment.py    ASR + G2P + aligner orchestration
    ustx_curves.py        f0/vuv masked reference curves for USTX
    sentence_gacha.py     per-sentence candidate takes + splicing
    memory_utils.py       free_memory()
  device_utils.py      device normalization, DXGI adapter probing, provider resolution

scripts/
  auto_lyric_cli.py    single/batch full pipeline
  slice_asr_cli.py     folder-oriented slice + ASR batch
  sentence_gacha.py    rerun one lyric line with a parameter grid
  sentence_curves.py   inject pitch/voicing reference curves into a USTX
```

## Main runtime model

| Component | Backend | Location |
| --- | --- | --- |
| Qwen3-ASR encoder | ONNX Runtime | `inference/qwen3asr_dml/` |
| Qwen3-ASR decoder | `llama.cpp` (Metal / Vulkan / CPU) | `inference/qwen3asr_dml/llama.py` |
| RomajiASR | ONNX Runtime | `inference/romaji_asr/` |
| PinyinASR | ONNX Runtime | `inference/pinyin_asr/` |
| kashi-g2p | ONNX Runtime | `inference/kashi_g2p_ja/` |
| TiFA (default aligner) | ONNX Runtime | `inference/TiFA/` |
| HubertFA (optional aligner) | ONNX Runtime | `inference/HubertFA/` |
| GAME | ONNX Runtime | `inference/game/` |
| RMVPE | ONNX Runtime | `inference/API/rmvpe_api.py` |

Device normalization is centralized in `inference/device_utils.py`:

- Windows defaults to `dml` (DirectML); other platforms default to `cpu`.
- Visible device choices are `dml`/`cpu` on Windows and `cpu`/`metal` elsewhere.
- Legacy values (`cuda`, `directml`, `gpu`, empty string) normalize to `dml`.
- ONNX providers fall back to `CPUExecutionProvider` when DirectML is unavailable.
- The llama.cpp decoder picks Metal on Apple Silicon, otherwise Vulkan when the ggml Vulkan
  backend library is present, otherwise CPU.

## Main user entrypoints

### Desktop GUI

- `app_fluent.py` -> `gui/fluent_main.py` (Windows), `gui/macos_main.py` (macOS)

The GUI collects settings, launches worker threads (`gui/fluent_worker.py`), and routes work
into the application layer.

### Application-layer job entry

- `application/pipeline.py`: `run_auto_lyric_job(PipelineConfig)`

Validates only the models the selected engines will actually load: GAME always; TiFA or
HubertFA depending on `alignment_engine` (Cantonese always TiFA); the direct phoneme ASR
model when pinyin/romaji is selected, otherwise Qwen; kashi-g2p when selected for Japanese;
RMVPE when pitch curves are requested for USTX/VSQX. Failures become `ModelNotFoundError`,
cancellation becomes `CancellationError`, anything else becomes `Vocal2MidiError`.

- `application/sentence_jobs.py`: sentence-repair helpers (USTX header reading, reference-curve
  export) shared by the GUI and the sentence CLI scripts.

### Hybrid pipeline

- `inference/pipeline/auto_lyric_hybrid.py`: `auto_lyric_hybrid_pipeline()`

Stages:

1. load audio at 44.1 kHz (`inference/io/audio_io.py`)
2. optional RMVPE pitch-curve extraction
3. slicing (RMS / smart, optionally guided by the RMVPE voiced mask)
4. ASR: Qwen3-ASR text, or direct RomajiASR (ja) / PinyinASR (zh) phonemes
5. lyric matching + G2P -> `.lab` phoneme sequences
6. forced alignment: **TiFA by default**, HubertFA selectable
7. GAME note extraction (aligned, with per-chunk pitch-only fallback)
8. vibrato-tail collapse, optional quantization, export

## Key inference APIs

### ASR

- `inference/API/asr_api.py`

Responsibilities:

- load and cache the Qwen3-ASR, RomajiASR and PinyinASR runtimes (per-path/device caches, lock-guarded)
- batch transcription for chunked audio
- isolate Qwen work in a subprocess; `asr_session` lets a batch reuse one spawned worker

Backends: `inference/qwen3asr_dml/`, `inference/romaji_asr/`, `inference/pinyin_asr/`.

Engine selection mirrors `application/pipeline.py`: the direct phoneme engines *replace*
Qwen rather than adding to it, and degrade to Qwen text ASR + G2P when their model does not resolve.

### Lyric matching and G2P

- `inference/API/lfa_api.py`, `inference/LyricFA/`
- `inference/TiFA/g2p/`, `inference/TiFA/g2p_pipeline.py`, `inference/kashi_g2p_ja/`
- `inference/t2s_utils.py`

Responsibilities:

- create lyric matchers and normalize ASR text or phoneme output
- generate `.lab` alignment input
- Japanese G2P engines: `kashi-g2p-onnx` (default, lattice beam-search candidates for TiFA)
  or `pyopenjtalk`
- Chinese traditional/simplified normalization and lyric display modes

### Forced alignment

- TiFA (default): `inference/TiFA/runtime.py`, `inference/TiFA/aligner.py`, `inference/TiFA/textgrid.py`
- HubertFA (optional): `inference/API/hfa_api.py`, `inference/HubertFA/`
- Orchestration: `inference/pipeline/lyric_alignment.py` (`_run_lyric_alignment`, `_AlignmentOutcome`)

Responsibilities:

- build inputs from chunk `.wav` + `.lab` pairs
- run phoneme-level forced alignment and produce `pred_dict` / `chars_dict`
- export TextGrid artifacts (and chunk WAVs when the `chunks` output is requested)

Cantonese (`yue`) is TiFA-only: HubertFA has no Cantonese dictionary support and is disabled
for it in the GUI and ignored in application-layer validation.

### Note extraction

- `inference/API/game_api.py`
- `inference/game/`

Responsibilities:

- load and cache the GAME ONNX model set (`_GAME_MODEL_CACHE`, keyed by path + device)
- `extract_pitches_and_align()` when alignment succeeded: notes carry lyrics and vowel boundaries
- `extract_pitches_only()` when there is no usable alignment: pitch-only notes
- align note boundaries to word durations (`inference/game/alignment_utils.py`)

Fallback is per chunk, not per song: chunks missing usable alignment are collected and rerun
through `extract_pitches_only()` while the rest keep their lyrics.

### RMVPE

- `inference/API/rmvpe_api.py`

Responsibilities:

- load the RMVPE ONNX model
- compute frame-level salience and F0
- provide interpolated MIDI pitch curves for USTX/VSQX export
- optionally supply voiced/unvoiced guidance for smart slicing

### Slicing

- `inference/API/slicer_api.py`, `inference/slicer/slicer2.py`

Responsibilities:

- default slicing plus heuristic/smart paths
- slice-duration bounds validated in `application/config.py` (0-60 s, default 5-10 s)
- optional RMVPE-assisted voiced/unvoiced guidance

### Quantization

- `inference/quant/quantization.py`, `inference/quant/smart_quantize.py`

Two modes: `simple` (grid snapping) and `smart` (default; rhythmic simplification).
`quantization_step <= 0` disables quantization entirely regardless of mode.

## Data flow

```text
audio (44.1 kHz)
  -> optional RMVPE pitch curve
  -> slicing
  -> ASR (Qwen3-ASR text | RomajiASR | PinyinASR)
  -> lyric matching + G2P -> .lab
  -> forced alignment (TiFA default | HubertFA)
  -> GAME note extraction
       aligned chunks        -> notes with lyrics
       unaligned chunks      -> pitch-only fallback
  -> word/note boundary alignment
  -> vibrato-tail collapse
  -> quantization (simple | smart, only when step > 0)
  -> export
```

No-lyrics path (`output_lyrics=False`):

```text
audio
  -> optional RMVPE
  -> slicing
  -> GAME pitch-only extraction
  -> export
```

## Export layer

Relevant modules:

- `inference/io/note_io.py` — MIDI, TXT, CSV
- `inference/API/ustx_api.py` — OpenUtau USTX, including the per-note `pitd` curve derived from
  RMVPE (median filter, adaptive smoothing, short-gap fill)
- `inference/API/vsqx_api.py` — VOCALOID VSQX
- `inference/pipeline/ustx_curves.py` — read-only `f0` / `vuv` **masked curves** injected into a
  USTX part: `f0` is the sung pitch in MIDI x 100, `vuv` marks voiced spans, and gaps mean silence.
  These are reference data for hand-editing, not render input. Ticks are part-relative and
  quantized to the USTX curve grid.

Written artifacts: `.mid`, `.ustx`, `.vsqx`, `.txt`, `.csv`, TextGrid, chunk `.wav`, `.lab`,
and the ASR match log when lyrics are enabled.

## Sentence repair path

For passages the decoder gets wrong, the pipeline can be rerun on a single lyric line:

- `inference/pipeline/sentence_gacha.py` — parse timestamped lyric lines, resolve the audio
  window, build a grid of decoding parameters (`TakeParams`), run the takes, diagnose each
  result, write reports/manifests. `select_target_part()` picks the voice part overlapping the
  window (note times include the part offset); `splice_take_into_ustx()` replaces that part's
  window notes and `preview_splice()` dry-runs the counts/target for the confirm dialog.
  `add_reference_take_into_ustx()` auditions the take on one reusable `V2M参考` track cloning the
  target track's singer/phonemizer, muting vocal tracks while keeping wave-only accompaniment;
  `snapshot_mix()` / `restore_mix_into_ustx()` save and restore per-track mute/solo.
- `application/sentence_jobs.py` — GUI/CLI-shared jobs: `export_reference_curves()`,
  `splice_take_with_handshake()`, `add_reference_take_with_handshake()`,
  `restore_mix_with_handshake()` and `restore_backup_with_handshake()` (in-place write with a
  rolling `<stem>_backup.ustx` plus a timestamped `<stem>_backups/` history, newest 10 kept;
  `list_backups()`), `bar_beat_label()` for OpenUtau-style positions.
- `application/openutau_bridge.py` — find the running OpenUtau instance (process snapshot →
  data dir → `prefs.json` recents), default the base USTX to its open project, and run the
  save handshake in `<DataPath>/v2m-bridge/` (`save-request.json` → `save-response.json`).
  Patch aborts instead of overwriting when a running instance cannot save.
- `gui/sentence_repair_view.py` — the interactive page for the same flow.
- `scripts/sentence_gacha.py`, `scripts/sentence_curves.py` — CLI frontends.

The reload side lives in the OpenUtau fork (`feat/v2m-bridge`, `OpenUtau/Util/V2mBridge.cs`):
it answers save requests on the UI thread and watches the open project file, silently
reloading when it changes externally (clean document) or prompting when edits are unsaved.

## Runtime and cancellation behavior

- background execution from the GUI worker thread
- cancellation via a `cancel_checker` callback polled between stages and inside chunk loops,
  raising `InterruptedError` (mapped to `CancellationError` at the application boundary)
- chunk batching for ASR and GAME, with DML batch size capped by available video memory
- model caches are keyed by path + device and reused across songs in batch flows
- loaded DirectML sessions are deliberately kept alive between runs: tearing a session down
  per song can access-violate onnxruntime on the next one
- `free_memory()` in `inference/pipeline/memory_utils.py` is only `gc.collect()`; intermediate
  chunk WAV/`.lab`/TextGrid files live in `tempfile.TemporaryDirectory()` and are cleaned automatically

## CLI paths

- `scripts/auto_lyric_cli.py` — full pipeline for one file or a folder; `--aligner tifa|hfa`,
  `--chinese-asr pinyin|qwen`, `--japanese-asr romaji|qwen`
- `scripts/slice_asr_cli.py` — folder-oriented slice + Qwen3-ASR batch, saves chunks and `.lab`,
  optionally keeping ASR/RMVPE runtimes alive across the batch
- `scripts/sentence_gacha.py` — per-line candidate takes and splicing
- `scripts/sentence_curves.py` — write `f0`/`vuv` reference curves into an existing USTX

## Current rough edges

- `environment.yml` still describes the legacy Torch/CUDA conda environment; `requirements.txt`
  is the current runtime dependency list.
- Torch-era vendored code remains in `inference/HubertFA/tools/metrics.py`,
  `inference/TiFA/decoding.py` and `inference/qwen3asr_dml/gguf/lazy.py`; none of it is on the
  main inference path.
- The required config field is still named `hfa_model_dir` even though TiFA is the default
  aligner; the TiFA path uses `tifa_model_path`.

None of these change the intended direction: ONNX-first inference, platform-aware `llama.cpp`,
and stable application-layer entrypoints.
