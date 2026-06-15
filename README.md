# AfriSpeech DataGen — synthetic speech data generator

Turn a **text dataset** into **synthetic TTS training audio** — streamed through
the AfriSpeech VoxCPM model, voice-cloning built-in male/female reference
speakers. It writes 16 kHz WAVs + a manifest locally, runs multiple model
instances in parallel, and resumes where it left off.

A command-line port of the
[Hugging Face Space](https://huggingface.co/spaces/AfriSpeech/VoxCPM-Synthetic-Data-Generator) —
no UI, no queue, just run it.

> **A GPU is required** for usable speed (VoxCPM is a neural TTS model; ~4.5 GB
> VRAM per instance). A T4 works; bigger GPUs run more instances in parallel.

## Run in the cloud (free T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AfriSpeech/afrispeech-datagen/blob/main/notebooks/afrispeech_datagen.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/AfriSpeech/afrispeech-datagen/blob/main/notebooks/afrispeech_datagen.ipynb)

Pick a **GPU** runtime (Colab: `Runtime → Change runtime type → T4`; Kaggle:
`Settings → Accelerator → GPU`, and **Internet ON**).

## Install (local)

```bash
git clone https://github.com/AfriSpeech/afrispeech-datagen.git
cd afrispeech-datagen
python3 -m venv .venv && source .venv/bin/activate
sudo apt-get install -y ffmpeg          # system dependency
pip install -e .                        # gives you the `afrispeech-datagen` command
```

## Quickstart

```bash
# 1) Preview 5 clips to hear the result before a big run
afrispeech-datagen --dataset ghananlpcommunity/your-text-dataset --text-column text --preview 5

# 2) Generate 5 hours (50/50 male/female) into data/<name>
afrispeech-datagen --dataset ghananlpcommunity/your-text-dataset --text-column text \
    --hours 5 --name twi-run

# 3) Resume: re-run the same command — finished rows are skipped
# 4) Push the finished run to your own HF dataset repo
afrispeech-datagen --dataset … --text-column text --hours 5 --name twi-run --push you/my-synth
```

Find a dataset's text column by inspecting it on the Hub, or list org datasets
with `afrispeech-datagen --list-datasets`.

## Output

Everything lands in `data/<name>/` (override with `--out`):

```
data/twi-run/
  audio/<rowidx>_<runid>.wav   16 kHz mono, silence-trimmed
  manifest.jsonl               id, file, text, gender, speaker, duration per row
  progress.json                resume state (re-run to continue)
```

The manifest is LJSpeech-adjacent and easy to adapt to any TTS trainer.

## Options

| flag | meaning |
|------|---------|
| `--dataset ID` | source text dataset on the HF Hub (required) |
| `--text-column COL` | column holding the text (required) |
| `--config` / `--split` | dataset config / split (default split `train`) |
| `--hours H` | target hours of audio to generate |
| `--voices custom\|male\|female` | speaker selection (default `custom`) |
| `--male-pct N` | %% male in `custom` mode (deterministic per row) |
| `--max-chars N` | skip rows longer than this (default 400) |
| `--instances N` | parallel model instances (default: auto by VRAM) |
| `--cfg` / `--steps` | VoxCPM CFG value / inference timesteps |
| `--name` / `--out` | run name (→ `data/<name>`) or explicit output dir |
| `--push REPO [--private]` | upload the finished run to an HF dataset repo |
| `--token` | HF token (else `HF_TOKEN` env) — needed for gated datasets/models |
| `--preview N` | generate N preview clips and exit |
| `--list-datasets` | list datasets under the AfriSpeech org |

Resuming is automatic: point `--name`/`--out` at an existing run folder (or just
re-run the same command) and it reads `progress.json` and skips finished rows.

## Use as a library

```python
from afrispeech_datagen import generate, preview

preview(dataset="org/ds", text_column="text", out_dir="data/run", n=5)
summary = generate(dataset="org/ds", text_column="text", out_dir="data/run",
                   target_hours=5, voices="custom", male_pct=50)
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
  generator.py       model load, voice-clone, silence-trim, parallel run, resume
  speakers/          built-in male/female reference wav + text
notebooks/afrispeech_datagen.ipynb   Colab/Kaggle (GPU) runner
tests/
```

## License

CC-BY-4.0
