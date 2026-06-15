"""Tests for the GPU-free helpers and CLI parsing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxcpm_synth import clean_text, pick_gender, sanitize_name, trim_silences, SPEAKERS
from voxcpm_synth import cli


def test_clean_text():
    assert clean_text("  hello\n world  \t x ") == "hello world x"
    assert clean_text("a\n\nb") == "a b"


def test_pick_gender_modes():
    assert pick_gender(0, "male", 50) == "male"
    assert pick_gender(7, "all male", 50) == "male"
    assert pick_gender(0, "female", 50) == "female"
    assert pick_gender(3, "all female", 50) == "female"


def test_pick_gender_custom_deterministic_and_split():
    # deterministic: same idx -> same voice
    assert pick_gender(42, "custom", 50) == pick_gender(42, "custom", 50)
    # all-male / all-female extremes via pct
    assert all(pick_gender(i, "custom", 100) == "male" for i in range(50))
    assert all(pick_gender(i, "custom", 0) == "female" for i in range(50))
    # ~50/50 split is reasonably balanced
    males = sum(pick_gender(i, "custom", 50) == "male" for i in range(1000))
    assert 400 < males < 600


def test_sanitize_name():
    assert sanitize_name("My Run #1!") == "My-Run-1"
    assert sanitize_name("   ") == "run"
    assert sanitize_name("twi_run-2") == "twi_run-2"


def test_trim_silences_keeps_audio_and_shortens_gaps():
    sr = 16000
    tone = np.sin(np.linspace(0, 50, sr)).astype("float32")  # 1s of signal
    gap = np.zeros(sr * 2, dtype="float32")                  # 2s silence
    wav = np.concatenate([tone, gap, tone])
    out = trim_silences(wav, sr=sr)
    assert out.size > 0
    assert out.size < wav.size            # the long gap was collapsed
    assert out.size >= 2 * sr             # both tone segments survive


def test_speakers_loaded():
    for g in ("male", "female"):
        assert SPEAKERS[g]["text"]                       # reference transcript present
        assert Path(SPEAKERS[g]["wav"]).exists()         # reference wav bundled


def test_cli_parser_and_requirements():
    a = cli.build_parser().parse_args(
        ["--dataset", "org/ds", "--text-column", "text", "--hours", "5",
         "--voices", "custom", "--male-pct", "60", "--name", "run1"])
    assert a.dataset == "org/ds" and a.text_column == "text"
    assert a.hours == 5 and a.voices == "custom" and a.male_pct == 60

    # dataset + text-column required for a run
    try:
        cli.main(["--split", "train"])
    except SystemExit as e:
        assert "required" in str(e)
    else:
        raise AssertionError("expected SystemExit without --dataset/--text-column")
