"""Paquete del pipeline de ingesta (Silero VAD + Faster-Whisper)."""

from app.pipeline.processor import (
    AudioIngestionPipeline,
    TranscriptionResult,
    get_pipeline,
    process_long_audio,
    process_single_clip,
)

__all__ = [
    "AudioIngestionPipeline",
    "TranscriptionResult",
    "get_pipeline",
    "process_single_clip",
    "process_long_audio",
]
