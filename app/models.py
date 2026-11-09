"""
Modelos ORM en SQLAlchemy 2.0 para AccentLab Web:
- Speaker: Metadatos sociolingüísticos y dialectales del hablante.
- AudioSegment: Clip de audio normalizado (WAV 24kHz 16-bit Mono) y transcripción.
- AcousticMetrics: Parámetros prosódicos, formánticos, de calidad y cortes glotales.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utc_now() -> datetime:
    """Devuelve la fecha y hora actual en UTC con información de zona horaria."""
    return datetime.now(timezone.utc)


class Speaker(Base):
    """Representa un hablante registrado con perfil sociolingüístico regional."""

    __tablename__ = "speakers"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    speaker_code: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)  # 'M', 'F', 'Otro'
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    subregion: Mapped[str] = mapped_column(
        String(64), nullable=False, default="Centro"
    )  # Oriente, Centro, Sur, Poniente, Costa, Otro
    accent_intensity: Mapped[str] = mapped_column(
        String(32), nullable=False, default="Medio"
    )  # Suave, Medio, Marcado
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )

    segments: Mapped[list["AudioSegment"]] = relationship(
        "AudioSegment",
        back_populates="speaker",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="desc(AudioSegment.created_at)",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serializa el hablante a un diccionario compatible con JSON."""
        return {
            "id": self.id,
            "speaker_code": self.speaker_code,
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "subregion": self.subregion,
            "accent_intensity": self.accent_intensity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AudioSegment(Base):
    """Clip de voz segmentado y normalizado listo para análisis acústico y entrenamiento TTS/VC."""

    __tablename__ = "audio_segments"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    speaker_id: Mapped[int] = mapped_column(
        ForeignKey("speakers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="mic"
    )  # 'mic' | 'upload_vad'
    prompt_mode: Mapped[str] = mapped_column(
        String(32), nullable=False, default="spontaneous"
    )  # 'trap_phrase' | 'spontaneous' | 'long_audio'
    prompt_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    duration_sec: Mapped[float] = mapped_column(Float, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False, default=24000)
    orig_start_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    orig_end_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    transcript: Mapped[str] = mapped_column(Text, nullable=False, default="")
    raw_transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    asr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # --- Campos Lingüísticos v2.0 (Maya Yucateco Norma 1984 + G2P + Morfosintaxis) ---
    language_code: Mapped[str] = mapped_column(
        String(16), nullable=False, default="es_yuc", index=True
    )  # 'yua' | 'es_yuc' | 'yua_es'
    ipa_phonemes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tts_phonetic_tokens: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    grammatical_analysis: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )

    speaker: Mapped["Speaker"] = relationship("Speaker", back_populates="segments")
    metrics: Mapped[Optional["AcousticMetrics"]] = relationship(
        "AcousticMetrics",
        back_populates="segment",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(
        self,
        include_speaker: bool = True,
        include_metrics: bool = True,
    ) -> dict[str, Any]:
        """Serializa el segmento de audio junto con su hablante y métricas acústicas."""
        data: dict[str, Any] = {
            "id": self.id,
            "speaker_id": self.speaker_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "audio_url": f"/media/wavs/{self.filename}",
            "source_type": self.source_type,
            "prompt_mode": self.prompt_mode,
            "prompt_text": self.prompt_text,
            "duration_sec": round(self.duration_sec, 3),
            "sample_rate": self.sample_rate,
            "orig_start_sec": (
                round(self.orig_start_sec, 3)
                if self.orig_start_sec is not None
                else None
            ),
            "orig_end_sec": (
                round(self.orig_end_sec, 3)
                if self.orig_end_sec is not None
                else None
            ),
            "transcript": self.transcript,
            "raw_transcript": self.raw_transcript,
            "asr_confidence": (
                round(self.asr_confidence, 4)
                if self.asr_confidence is not None
                else None
            ),
            "is_validated": self.is_validated,
            "language_code": self.language_code or "es_yuc",
            "ipa_phonemes": self.ipa_phonemes,
            "tts_phonetic_tokens": self.tts_phonetic_tokens,
            "grammatical_analysis": self.grammatical_analysis,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_speaker and self.speaker is not None:
            data["speaker"] = self.speaker.to_dict()
        if include_metrics:
            data["metrics"] = self.metrics.to_dict() if self.metrics is not None else None
        return data


class AcousticMetrics(Base):
    """Métricas acústicas y prosódicas extraídas mediante Praat (Parselmouth) y NumPy."""

    __tablename__ = "acoustic_metrics"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    segment_id: Mapped[int] = mapped_column(
        ForeignKey("audio_segments.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )

    # --- Frecuencia Fundamental (F0) y Entonación ("Cantadito") ---
    f0_mean_hz: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f0_median_hz: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f0_min_hz: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f0_max_hz: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f0_semitones_std: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f0_semitones_range: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f0_curve_semitones: Mapped[list[float]] = mapped_column(
        JSON, nullable=False, default=list
    )

    # --- Estructura Formántica (F1 / F2 y Mapa Vocálico) ---
    f1_mean_hz: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f2_mean_hz: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    vowel_space_points: Mapped[list[dict[str, float]]] = mapped_column(
        JSON, nullable=False, default=list
    )

    # --- Calidad de Señal y Relación Armónico/Ruido ---
    hnr_db: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    snr_db: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    voiced_fraction: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # --- Rasgos Glotales Regionales (Saltillos / Cortes Glotales < 20ms) ---
    glottal_stops_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    glottal_stop_timestamps: Mapped[list[float]] = mapped_column(
        JSON, nullable=False, default=list
    )

    # --- Alineación Fonético-Acústica Maya 1984 (0.0 a 100.0) ---
    phonetic_alignment_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )

    segment: Mapped["AudioSegment"] = relationship(
        "AudioSegment", back_populates="metrics"
    )

    def to_dict(self) -> dict[str, Any]:
        """Serializa las métricas acústicas garantizando valores flotantes redondeados."""
        return {
            "id": self.id,
            "segment_id": self.segment_id,
            "f0_mean_hz": round(self.f0_mean_hz, 2),
            "f0_median_hz": round(self.f0_median_hz, 2),
            "f0_min_hz": round(self.f0_min_hz, 2),
            "f0_max_hz": round(self.f0_max_hz, 2),
            "f0_semitones_std": round(self.f0_semitones_std, 3),
            "f0_semitones_range": round(self.f0_semitones_range, 3),
            "f0_curve_semitones": list(self.f0_curve_semitones or []),
            "f1_mean_hz": round(self.f1_mean_hz, 1),
            "f2_mean_hz": round(self.f2_mean_hz, 1),
            "vowel_space_points": list(self.vowel_space_points or []),
            "hnr_db": round(self.hnr_db, 2),
            "snr_db": round(self.snr_db, 2),
            "voiced_fraction": round(self.voiced_fraction, 3),
            "glottal_stops_count": self.glottal_stops_count,
            "glottal_stop_timestamps": list(self.glottal_stop_timestamps or []),
            "phonetic_alignment_score": round(
                float(self.phonetic_alignment_score or 0.0), 2
            ),
        }
