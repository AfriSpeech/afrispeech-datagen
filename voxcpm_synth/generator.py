"""Core synthetic-speech generation with VoxCPM (voice-cloned male/female).

Streams a text dataset, synthesises each row with the AfriSpeech VoxCPM model
using the built-in male/female reference speakers, trims silence, and writes
16 kHz mono WAVs + a manifest locally. Supports parallel model instances,
a target-hours budget, and resume. Ported from the Hugging Face Space, minus the
task queue / OAuth / email / usage logging.
"""

from __future__ import annotations

import json
import os
import re
import queue
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("MODELSCOPE_CACHE", "/tmp/modelscope_cache")

MODEL_ID = "AfriSpeech/voxcpm-afrispeech-full-inference-20260606"
SAMPLE_RATE = 16000
SILENCE_TOP_DB = 30
SILENCE_MAX_GAP_S = 0.3

_SPEAKER_DIR = Path(__file__).resolve().parent / "speakers"
SPEAKERS = {
    "male":   {"wav": str(_SPEAKER_DIR / "male.wav"),   "txt": _SPEAKER_DIR / "male.txt"},
    "female": {"wav": str(_SPEAKER_DIR / "female.wav"), "txt": _SPEAKER_DIR / "female.txt"},
}
for _g, _s in SPEAKERS.items():
    _s["text"] = Path(_s["txt"]).read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# Pure helpers (no model / GPU needed — unit-testable)
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def pick_gender(idx: int, mode: str, male_pct: int) -> str:
    """Deterministic per-row voice so resumed runs assign the same speaker."""
    if mode in ("male", "all male"):
        return "male"
    if mode in ("female", "all female"):
        return "female"
    return "male" if (idx * 2654435761) % 100 < male_pct else "female"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-") or "run"


def trim_silences(wav, sr=SAMPLE_RATE, top_db=SILENCE_TOP_DB, max_gap_s=SILENCE_MAX_GAP_S):
    """Collapse long internal silences while keeping a small natural gap."""
    import librosa

    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.ndim != 1 or wav.size == 0:
        return wav
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav
    max_gap = int(max_gap_s * sr)
    pieces, prev_end = [], None
    for start, end in intervals:
        if prev_end is not None:
            keep = min(start - prev_end, max_gap)
            if keep > 0:
                pieces.append(wav[prev_end:prev_end + keep])
        pieces.append(wav[start:end])
        prev_end = end
    return np.concatenate(pieces)


def auto_instances() -> int:
    """Parallel model instances that fit in VRAM (~4.5 GB each)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 1
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return max(1, int((vram_gb * 0.8) // 4.5))
    except Exception:
        return 1


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_instance(model_id: str = MODEL_ID):
    from voxcpm import VoxCPM
    try:
        return VoxCPM.from_pretrained(model_id, load_denoiser=False)
    except TypeError:
        return VoxCPM.from_pretrained(model_id)


def _generate_one(model, caches: dict, text: str, gender: str,
                  cfg_value: float, steps: int) -> np.ndarray:
    if gender not in caches:
        sp = SPEAKERS[gender]
        caches[gender] = model.tts_model.build_prompt_cache(
            prompt_text=sp["text"], prompt_wav_path=sp["wav"]
        )
    wav, _, _ = model.tts_model.generate_with_prompt_cache(
        target_text=text,
        prompt_cache=caches[gender],
        max_len=4096,
        cfg_value=float(cfg_value),
        inference_timesteps=int(steps),
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )
    if hasattr(wav, "cpu"):
        wav = wav.squeeze(0).cpu().numpy()
    return trim_silences(wav)


# --------------------------------------------------------------------------- #
# Generation run (local output, parallel workers, resume)
# --------------------------------------------------------------------------- #
class _Run:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.feeding_done = False
        self.rows: dict[str, dict] = {}
        self.total_seconds = 0.0
        self.errors = 0
        self.fatal = ""
        self.run_id = ""
        self.audio_dir = ""
        self.cfg_value = 2.0
        self.steps = 10


def _worker(run: _Run, model_id: str):
    try:
        model = load_instance(model_id)
    except Exception as e:  # noqa: BLE001
        run.fatal = run.fatal or f"model load failed: {e}"
        run.stop.set()
        return
    caches: dict = {}
    while not (run.stop.is_set() and run.q.empty()):
        try:
            idx, text, gender = run.q.get(timeout=2)
        except queue.Empty:
            if run.feeding_done:
                break
            continue
        try:
            wav = _generate_one(model, caches, text, gender, run.cfg_value, run.steps)
            dur = float(len(wav)) / SAMPLE_RATE
            rel = f"audio/{idx:07d}_{run.run_id}.wav"
            out = os.path.join(run.audio_dir, f"{idx:07d}_{run.run_id}.wav")
            tmp = out + ".tmp"
            sf.write(tmp, wav, SAMPLE_RATE, subtype="PCM_16")
            os.replace(tmp, out)
            with run.lock:
                run.rows[str(idx)] = {
                    "id": f"{run.run_id}_{idx:07d}", "file": rel, "text": text,
                    "gender": gender, "speaker": gender, "duration": round(dur, 3),
                }
                run.total_seconds += dur
        except Exception as e:  # noqa: BLE001
            with run.lock:
                run.errors += 1
        finally:
            run.q.task_done()


def _write_manifest(out_dir: str, run: _Run, meta: dict):
    with run.lock:
        rows = dict(run.rows)
        total = run.total_seconds
    progress = {**meta, "run_id": run.run_id, "total_seconds": round(total, 2),
                "rows": rows,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    Path(out_dir, "progress.json").write_text(
        json.dumps(progress, ensure_ascii=False), encoding="utf-8")
    with open(Path(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for k in sorted(rows, key=int):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")


def generate(
    *,
    dataset: str,
    text_column: str,
    out_dir: str,
    config: str | None = None,
    split: str = "train",
    target_hours: float = 1.0,
    voices: str = "custom",
    male_pct: int = 50,
    instances: int | None = None,
    cfg_value: float = 2.0,
    steps: int = 10,
    max_chars: int = 400,
    model_id: str = MODEL_ID,
    token: str | None = None,
    save_every: int = 200,
    on_clip=None,
    progress=None,
):
    """Generate synthetic speech for a dataset into ``out_dir``.

    Writes ``audio/*.wav`` (16 kHz mono), ``manifest.jsonl`` and ``progress.json``.
    Resumes automatically if ``out_dir/progress.json`` exists (skips done rows).
    ``on_clip(duration)`` fires per generated clip; ``progress(msg)`` for status.
    Returns a summary dict.
    """
    from datasets import load_dataset

    out_dir = str(out_dir)
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    target_seconds = max(0.0, float(target_hours)) * 3600

    run = _Run()
    run.cfg_value, run.steps = cfg_value, steps
    run.audio_dir = os.path.join(out_dir, "audio")

    # Resume from a prior progress.json in this folder.
    prog_path = Path(out_dir, "progress.json")
    if prog_path.exists():
        prev = json.loads(prog_path.read_text(encoding="utf-8"))
        run.run_id = prev.get("run_id") or uuid.uuid4().hex[:8]
        run.rows = dict(prev.get("rows", {}))
        run.total_seconds = float(prev.get("total_seconds", 0.0))
        if progress:
            progress(f"resuming {out_dir} — {len(run.rows)} rows / "
                     f"{run.total_seconds/3600:.2f} h already done")
    else:
        run.run_id = uuid.uuid4().hex[:8]

    n_inst = instances if instances and instances > 0 else auto_instances()
    if progress:
        progress(f"loading {n_inst} model instance(s)…")
    workers = [threading.Thread(target=_worker, args=(run, model_id), daemon=True)
               for _ in range(n_inst)]
    for w in workers:
        w.start()

    meta = {"model_id": model_id, "dataset": dataset, "config": config or "",
            "split": split, "text_column": text_column, "voices": voices,
            "male_pct": male_pct, "target_hours": target_hours}

    ds = load_dataset(dataset, config or None, split=split, streaming=True, token=token)
    staged = 0
    for idx, ex in enumerate(ds):
        if run.stop.is_set() or run.fatal:
            break
        if run.total_seconds >= target_seconds:
            break
        if str(idx) in run.rows:
            continue
        text = clean_text(ex.get(text_column, ""))
        if not (2 <= len(text) <= max_chars):
            continue
        gender = pick_gender(idx, voices, male_pct)
        while not run.stop.is_set():
            try:
                run.q.put((idx, text, gender), timeout=2)
                break
            except queue.Full:
                pass
        # surface progress as clips complete
        done = len(run.rows)
        if on_clip:
            # report cumulative seconds via the run state
            on_clip(run.total_seconds)
        staged += 1
        if staged >= save_every:
            _write_manifest(out_dir, run, meta)
            staged = 0

    run.feeding_done = True
    for w in workers:
        w.join()
    _write_manifest(out_dir, run, meta)

    if run.fatal:
        raise RuntimeError(run.fatal)
    return {"rows": len(run.rows), "hours": run.total_seconds / 3600,
            "errors": run.errors, "out_dir": out_dir, "run_id": run.run_id}


def preview(*, dataset, text_column, out_dir, config=None, split="train",
            voices="custom", male_pct=50, cfg_value=2.0, steps=10, n=5,
            max_chars=400, model_id=MODEL_ID, token=None):
    """Generate ``n`` preview clips into ``out_dir/preview`` and return their info."""
    from datasets import load_dataset

    pdir = Path(out_dir, "preview")
    pdir.mkdir(parents=True, exist_ok=True)
    model = load_instance(model_id)
    caches: dict = {}
    ds = load_dataset(dataset, config or None, split=split, streaming=True, token=token)
    out = []
    for idx, ex in enumerate(ds):
        if len(out) >= n:
            break
        text = clean_text(ex.get(text_column, ""))
        if not (2 <= len(text) <= max_chars):
            continue
        gender = pick_gender(idx, voices, male_pct)
        wav = _generate_one(model, caches, text, gender, cfg_value, steps)
        path = pdir / f"preview_{len(out)+1}_{gender}.wav"
        sf.write(str(path), wav, SAMPLE_RATE, subtype="PCM_16")
        out.append({"file": str(path), "gender": gender,
                    "duration": round(len(wav) / SAMPLE_RATE, 2), "text": text})
    return out
