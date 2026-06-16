# AfriSpeech DataGen — synthetic speech data generator

Turn a **text dataset** into **synthetic TTS training audio** — streamed through
[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), voice-cloning built-in
male/female reference speakers. It writes WAVs (resampled to your target rate)
+ a manifest locally, runs multiple model instances in parallel, and resumes
where it left off.

> **GPU recommended** for speed — OmniVoice auto-detects CUDA and falls back to
> CPU if none is available (~3 GB VRAM per instance in fp32, ~2 GB in fp16).
> A T4 fits 4 instances (fp32) or 6 (fp16).

## Supported languages

OmniVoice can synthesise **646 languages**. Feed it text in any of them — the
model handles language detection automatically. For best results, pass
`--lang` when pushing a MeloTTS manifest (for the metadata label), but
generation itself is language-agnostic.

## Run in the cloud (free T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AfriSpeech/afrispeech-datagen/blob/main/notebooks/afrispeech_datagen.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/AfriSpeech/afrispeech-datagen/blob/main/notebooks/afrispeech_datagen.ipynb)

Pick a **GPU** runtime for full speed (Colab: `Runtime → Change runtime type → T4`;
Kaggle: `Settings → Accelerator → GPU`, and **Internet ON**). The notebook runs
on CPU too — just slower.

## Install (local)

```bash
git clone https://github.com/AfriSpeech/afrispeech-datagen.git
cd afrispeech-datagen
python3 -m venv .venv && source .venv/bin/activate
sudo apt-get install -y ffmpeg          # system dependency
pip install -e .                        # gives you the `afrispeech-datagen` command
```

GPU is optional — the tool runs on CPU without any extra flags.

## Quickstart

Source is **either** an HF dataset column **or** a plain text file (one sentence
per line). Output is written in your chosen **TTS format** (`--formats`).

```bash
# Preview 5 clips first (hear it before a big run)
afrispeech-datagen --dataset ghananlpcommunity/your-text-dataset --text-column text --preview 5

# From an HF dataset → 5 h, LJSpeech layout (default), into data/<name>
afrispeech-datagen --dataset ghananlpcommunity/your-text-dataset --text-column text \
    --hours 5 --name twi-run --formats ljspeech

# From your own sentences (one per line) → a Piper dataset
afrispeech-datagen --text-file sentences.txt --hours 2 --formats piper

# Multiple formats at once, then push to your HF repo
afrispeech-datagen --text-file sentences.txt --hours 2 --formats ljspeech,vits,melo \
    --lang TWI --push you/my-synth

# Resume: re-run the same command — finished rows are skipped
```

## Output — ready for your TTS trainer

Everything lands in `data/<name>/` (override with `--out`):

```
data/twi-run/
  wavs/<id>.wav            mono, silence-trimmed, at --sample-rate (default 24000)
  manifest.jsonl           full info: id, file, text, gender, speaker, duration
  progress.json            resume state (re-run to continue)
  # + the manifest(s) for the formats you asked for:
  metadata.csv             ljspeech  →  id|text|text
  metadata.piper.csv       piper     →  id|speaker|text   (metadata.csv if piper only)
  filelist.txt, speakers.txt   vits  →  wavs/<id>.wav|sid|text
  metadata.list            melo      →  wavs/<id>.wav|speaker|LANG|text
```

Point your framework's data-prep at this folder — `wavs/` + the matching manifest
is exactly what LJSpeech/Coqui, Piper, VITS, and MeloTTS expect. The transcript
is written verbatim (no normalisation — that's the framework's job).

## Options

| flag | meaning |
|------|---------|
| `--dataset ID` / `--text-column COL` | source: HF dataset — only the named column is fetched (audio and other columns are skipped) |
| `--text-file PATH` | source: a .txt file, one sentence per line |
| `--max-samples N` | use at most N input rows (independent of `--hours`) |
| `--config` / `--split` | dataset config / split (default split `train`) |
| `--hours H` | target hours of audio (default 1.0 unless `--max-samples` is set) |
| `--voices custom\|male\|female` | speaker selection (default `custom`) |
| `--male-pct N` | %% male in `custom` mode (deterministic per row) |
| `--max-chars N` | skip rows longer than this (default 400) |
| `--sample-rate HZ` | output WAV rate (default 24000 — OmniVoice native; e.g. 22050 for older frameworks) |
| `--precision fp32\|fp16\|bf16` | model precision (default fp32) — see Performance |
| `--instances N` | parallel model instances (default: auto by VRAM) |
| `--cfg` / `--steps` | CFG guidance scale / MaskGIT decoding steps (defaults 2.0 / 32) |
| `--formats …` | TTS manifests to write: `ljspeech,piper,vits,melo` (default `ljspeech`) |
| `--lang CODE` | language code for the `melo` manifest |
| `--name` / `--out` | run name (→ `data/<name>`) or explicit output dir |
| `--push REPO [--private]` | upload the finished run to an HF dataset repo |
| `--token` | HF token (else `HF_TOKEN` env) — for gated datasets/models |
| `--preview N` | generate N preview clips and exit |
| `--list-datasets` | list datasets under the AfriSpeech org |

Generation stops at whichever limit comes first — `--hours`, `--max-samples`, or
the end of the dataset. Use `--max-samples 500` (with no `--hours`) to generate
from exactly the first 500 usable rows of your input.

Resuming is automatic: point `--name`/`--out` at an existing run folder (or just
re-run the same command) and it reads `progress.json` and skips finished rows.

## Performance & GPU

- **Automatic GPU/CPU.** OmniVoice detects CUDA at load time and uses it
  automatically; no flags needed. On CPU it still works, just slower.
- **Parallel instances.** Several model copies pull rows off a shared queue in
  parallel (~3 GB VRAM each in fp32, ~2 GB in fp16). Auto-sized by VRAM — e.g.
  **4 instances on a T4** (fp32) or **6** (fp16). Override with `--instances N`.
- **Precision** (`--precision`):
  - `fp32` — default, safest, highest quality.
  - `fp16` — ~half the VRAM (more parallel instances) and faster on most GPUs;
    preview a few clips before a big run.
  - `bf16` — more numerically stable than fp16, **but needs an Ampere+ GPU
    (A100/L4/H100) — not a T4**.
- **Sample rate.** OmniVoice synthesises natively at **24 kHz**; output is
  resampled to `--sample-rate` (default 24000) to match your framework.
- **Decoding steps** (`--steps`). Default is 32 (OmniVoice MaskGIT default).
  Use `--steps 16` for roughly 2× faster generation at slightly lower quality.

## Use as a library

```python
from afrispeech_datagen import generate, export_formats

# From an HF dataset column, or pass texts=[...] for your own sentences
summary = generate(out_dir="data/run", dataset="org/ds", text_column="text",
                   target_hours=5, voices="custom", male_pct=50)
export_formats("data/run", ["ljspeech", "vits"], lang="TWI")
print(summary)   # {'rows': ..., 'hours': ..., 'errors': ..., 'out_dir': ..., 'run_id': ...}
```

## Tests

```bash
pip install pytest
pytest tests/        # GPU-free helpers + CLI parsing
```

## Project layout

```
afrispeech_datagen/
  cli.py             the `afrispeech-datagen` command
  generator.py       voice-clone, silence-trim, parallel run, resume, TTS-format export
  speakers/          built-in male/female reference wav + text
notebooks/afrispeech_datagen.ipynb   Colab/Kaggle runner
tests/
```

## License

CC-BY-4.0
