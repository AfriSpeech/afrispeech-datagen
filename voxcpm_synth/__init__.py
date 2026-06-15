"""VoxCPM synthetic-speech data generator — turn text datasets into TTS audio.

Voice-clones built-in male/female reference speakers with the AfriSpeech VoxCPM
model, writing 16 kHz WAVs + a manifest, with parallel instances and resume.
"""

from .generator import (
    MODEL_ID,
    SAMPLE_RATE,
    SPEAKERS,
    auto_instances,
    clean_text,
    generate,
    pick_gender,
    preview,
    sanitize_name,
    trim_silences,
)

__all__ = [
    "MODEL_ID",
    "SAMPLE_RATE",
    "SPEAKERS",
    "auto_instances",
    "clean_text",
    "generate",
    "pick_gender",
    "preview",
    "sanitize_name",
    "trim_silences",
]

__version__ = "0.1.0"
