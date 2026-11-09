"""
Motor de Procesamiento de Señales de Audio (DSP) y Fonética Acústica para AccentLab Web.

Responsabilidades principales:
1. Normalización universal de audio (WebM/Opus, MP3, OGG, M4A, FLAC, WAV) a
   WAV Mono, 24,000 Hz, 16-bit Signed PCM (estándar para XTTS / Coqui / LJSpeech).
2. Extracción acústica con Praat (praat-parselmouth) y NumPy:
   - Frecuencia Fundamental (F0) adaptativa por género y curva de entonación ("cantadito")
     normalizada en Semitonos respecto a la mediana del hablante, re-muestreada a 30 puntos.
   - Formantes vocálicos F1 y F2 (método de Burg en núcleos sonoros de alta energía) y
     nube de puntos (F1, F2) para el mapa acústico de vocales.
   - Relación Armónico-Ruido (HNR vía autocorrelación cruzada en Praat) y Relación
     Señal-Ruido (SNR por energía RMS de tramas activas vs. piso de ruido).
   - Heurística de detección de cortes glotales / saltillos (caídas abruptas de intensidad
     >= 10.5 dB en < 20 ms precedidas de fonación activa, con rechazo de fricativas por ZCR).
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import numpy as np
import parselmouth
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


def _resolve_ffmpeg_binary() -> Optional[str]:
    """Localiza el ejecutable de FFmpeg en el PATH del sistema o mediante imageio-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        exe_path = imageio_ffmpeg.get_ffmpeg_exe()
        exe_dir = str(Path(exe_path).resolve().parent)
        current_path = os.environ.get("PATH", "")
        if exe_dir not in current_path:
            os.environ["PATH"] = f"{exe_dir}{os.pathsep}{current_path}"
        return exe_path
    except Exception:
        return None


_FFMPEG_BIN: Optional[str] = _resolve_ffmpeg_binary()

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from pydub import AudioSegment as PydubAudioSegment

if _FFMPEG_BIN:
    PydubAudioSegment.converter = _FFMPEG_BIN


def _safe_float(value: Any, default: float = 0.0, decimals: int = 4) -> float:
    """Convierte un escalar numérico a float estándar libre de NaN o Infinito."""
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        return round(val, decimals)
    except (TypeError, ValueError):
        return default


@dataclass
class ProsodyAnalysisResult:
    """ Contenedor tipado con todos los resultados acústicos y prosódicos de un segmento."""

    normalized_wav_path: Path
    duration_sec: float
    sample_rate: int = 24000

    # Frecuencia Fundamental (F0) y Semitonos
    f0_mean_hz: float = 0.0
    f0_median_hz: float = 0.0
    f0_min_hz: float = 0.0
    f0_max_hz: float = 0.0
    f0_semitones_std: float = 0.0
    f0_semitones_range: float = 0.0
    f0_curve_semitones: list[float] = field(default_factory=lambda: [0.0] * 30)

    # Estructura Formántica (F1 / F2)
    f1_mean_hz: float = 0.0
    f2_mean_hz: float = 0.0
    vowel_space_points: list[dict[str, float]] = field(default_factory=list)

    # Calidad Acústica
    hnr_db: float = 0.0
    snr_db: float = 0.0
    voiced_fraction: float = 0.0

    # Rasgos Glotales (Cortes Glotales / Saltillos)
    glottal_stops_count: int = 0
    glottal_stop_timestamps: list[float] = field(default_factory=list)

    # Alineación Fonético-Acústica frente al perfil esperado G2P Maya 1984 (0.0 a 100.0)
    phonetic_alignment_score: float = 0.0

    def to_metrics_kwargs(self) -> dict[str, Any]:
        """Devuelve un diccionario listo para instanciar el modelo ORM AcousticMetrics."""
        return {
            "f0_mean_hz": _safe_float(self.f0_mean_hz, 0.0, 2),
            "f0_median_hz": _safe_float(self.f0_median_hz, 0.0, 2),
            "f0_min_hz": _safe_float(self.f0_min_hz, 0.0, 2),
            "f0_max_hz": _safe_float(self.f0_max_hz, 0.0, 2),
            "f0_semitones_std": _safe_float(self.f0_semitones_std, 0.0, 3),
            "f0_semitones_range": _safe_float(self.f0_semitones_range, 0.0, 3),
            "f0_curve_semitones": [
                _safe_float(v, 0.0, 3) for v in self.f0_curve_semitones
            ],
            "f1_mean_hz": _safe_float(self.f1_mean_hz, 0.0, 1),
            "f2_mean_hz": _safe_float(self.f2_mean_hz, 0.0, 1),
            "vowel_space_points": self.vowel_space_points,
            "hnr_db": _safe_float(self.hnr_db, 0.0, 2),
            "snr_db": _safe_float(self.snr_db, 0.0, 2),
            "voiced_fraction": _safe_float(self.voiced_fraction, 0.0, 3),
            "glottal_stops_count": int(self.glottal_stops_count),
            "glottal_stop_timestamps": [
                _safe_float(t, 0.0, 3) for t in self.glottal_stop_timestamps
            ],
            "phonetic_alignment_score": _safe_float(
                self.phonetic_alignment_score, 0.0, 2
            ),
        }


class ProsodyAnalyzer:
    """
    Motor DSP para normalización a WAV 24kHz 16-bit Mono y análisis fonético-acústico
    orientado al español yucateco y variantes dialectales de México.
    """

    TARGET_SR: int = 24000
    CURVE_POINTS: int = 30
    TARGET_PEAK_DBFS: float = -1.0  # Pico objetivo (-1.0 dBFS ≈ 0.89125 lineal)

    def __init__(
        self,
        target_sr: int = TARGET_SR,
        curve_points: int = CURVE_POINTS,
        target_peak_dbfs: float = TARGET_PEAK_DBFS,
    ) -> None:
        self.target_sr = target_sr
        self.curve_points = curve_points
        self.target_peak_dbfs = target_peak_dbfs

    # ==========================================================================
    # 1. NORMALIZACIÓN UNIVERSAL DE AUDIO (WAV MONO 24kHz 16-BIT PCM)
    # ==========================================================================
    def normalize_to_wav_24k(
        self,
        input_path: Path | str,
        output_path: Path | str,
    ) -> tuple[Path, float]:
        """
        Convierte cualquier archivo de audio (WebM del navegador, MP3, OGG, M4A, WAV)
        a WAV Mono de 24,000 Hz y 16-bit Signed PCM con corrección de offset DC y
        normalización suave de pico a -1.0 dBFS.

        Retorna:
            Tupla (ruta_wav_salida, duracion_segundos).
        """
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not in_path.exists() or in_path.stat().st_size == 0:
            raise ValueError(f"El archivo de audio de entrada no existe o está vacío: {in_path}")

        samples = self._decode_any_audio_to_mono_24k(in_path)
        if samples.size == 0:
            raise ValueError(f"No se pudieron extraer muestras de audio desde: {in_path}")

        duration_sec = self.save_mono_24k_pcm16(samples, out_path)
        return out_path, duration_sec

    def decode_to_mono_24k(self, input_path: Path | str) -> np.ndarray:
        """
        Decodifica cualquier archivo de audio a un vector NumPy 1D float64 Mono a 24,000 Hz.
        Útil para segmentación en memoria (ej. Silero VAD) sin múltiples recodificaciones.
        """
        in_path = Path(input_path)
        if not in_path.exists() or in_path.stat().st_size == 0:
            raise ValueError(f"El archivo de audio de entrada no existe o está vacío: {in_path}")
        samples = self._decode_any_audio_to_mono_24k(in_path)
        if samples.size == 0:
            raise ValueError(f"No se pudieron extraer muestras de audio desde: {in_path}")
        return samples

    def save_mono_24k_pcm16(
        self,
        samples: np.ndarray,
        output_path: Path | str,
    ) -> float:
        """
        Elimina offset DC, normaliza el pico a -1.0 dBFS sin recorte (clipping)
        y guarda un arreglo 1D en formato WAV Mono 24kHz 16-bit PCM.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mono = np.asarray(samples, dtype=np.float64)
        if mono.ndim > 1:
            mono = np.mean(mono, axis=-1)

        # Eliminación de componente continua (DC offset)
        mono = mono - float(np.mean(mono))

        # Normalización de pico a -1.0 dBFS con tope de ganancia máxima (+26 dB)
        # para no amplificar ruido térmico de fondo en grabaciones silenciosas.
        peak = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
        if peak > 1.5e-3:
            target_linear = 10.0 ** (self.target_peak_dbfs / 20.0)
            gain = min(target_linear / peak, 20.0)
            mono = mono * gain

        mono = np.clip(mono, -0.9999, 0.9999)
        sf.write(
            str(out_path),
            mono,
            self.target_sr,
            subtype="PCM_16",
            format="WAV",
        )
        return float(mono.shape[0]) / float(self.target_sr)

    def _decode_any_audio_to_mono_24k(self, input_path: Path) -> np.ndarray:
        """
        Decodifica contenedores arbitrarios (incluyendo WebM/Opus sin cabecera de duración
        emitidos por MediaRecorder) hacia un vector NumPy float64 a 24,000 Hz Mono.
        Orden de respaldo: FFmpeg CLI -> PyAV (libav) -> SoundFile/Pydub + Polyphase Resample.
        """
        errors: list[str] = []

        # Estrategia 1: FFmpeg directo (óptimo para WebM/Opus de navegadores, MP3, OGG, M4A)
        ffmpeg_bin = _FFMPEG_BIN or _resolve_ffmpeg_binary()
        if ffmpeg_bin:
            tmp_wav = Path(tempfile.gettempdir()) / f"accentlab_dec_{uuid.uuid4().hex}.wav"
            try:
                cmd = [
                    ffmpeg_bin,
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(self.target_sr),
                    "-acodec",
                    "pcm_s16le",
                    str(tmp_wav),
                ]
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=120,
                )
                if proc.returncode == 0 and tmp_wav.exists() and tmp_wav.stat().st_size > 44:
                    data, sr = sf.read(str(tmp_wav), dtype="float64", always_2d=False)
                    if sr != self.target_sr:
                        data = self._resample_mono(data, sr, self.target_sr)
                    return np.asarray(data, dtype=np.float64)
                stderr_msg = proc.stderr.decode("utf-8", errors="ignore")[-300:]
                errors.append(f"FFmpeg rc={proc.returncode}: {stderr_msg}")
            except Exception as exc:
                errors.append(f"FFmpeg exception: {exc}")
            finally:
                tmp_wav.unlink(missing_ok=True)

        # Estrategia 2: PyAV (integrado con faster-whisper; soporta WebM/Opus, MP3, OGG en C)
        try:
            import av

            chunks: list[np.ndarray] = []
            with av.open(str(input_path)) as container:
                audio_streams = [s for s in container.streams if s.type == "audio"]
                if not audio_streams:
                    raise ValueError("El contenedor no posee pistas de audio.")
                resampler = av.AudioResampler(
                    format="fltp",
                    layout="mono",
                    rate=self.target_sr,
                )
                for frame in container.decode(audio_streams[0]):
                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        arr = r_frame.to_ndarray()
                        chunks.append(arr[0].astype(np.float64))
                # Vaciar muestras remanentes del resampler
                for r_frame in resampler.resample(None):
                    arr = r_frame.to_ndarray()
                    chunks.append(arr[0].astype(np.float64))

            if chunks:
                return np.concatenate(chunks, axis=0)
        except Exception as exc:
            errors.append(f"PyAV fallback: {exc}")

        # Estrategia 3: SoundFile directo + re-muestreo polifásico SciPy (WAV, FLAC, OGG/Vorbis)
        try:
            data, sr = sf.read(str(input_path), dtype="float64", always_2d=False)
            if data.ndim > 1:
                data = np.mean(data, axis=-1)
            if sr != self.target_sr:
                data = self._resample_mono(data, sr, self.target_sr)
            return np.asarray(data, dtype=np.float64)
        except Exception as exc:
            errors.append(f"SoundFile fallback: {exc}")

        # Estrategia 4: Pydub general
        try:
            seg = PydubAudioSegment.from_file(str(input_path))
            seg = seg.set_channels(1).set_frame_rate(self.target_sr).set_sample_width(2)
            raw_samples = np.array(seg.get_array_of_samples(), dtype=np.float64) / 32768.0
            return raw_samples
        except Exception as exc:
            errors.append(f"Pydub fallback: {exc}")

        raise RuntimeError(
            f"Imposible decodificar el archivo de audio '{input_path.name}'. "
            f"Detalles: {' | '.join(errors)}"
        )

    @staticmethod
    def _resample_mono(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Re-muestrea una señal mono con filtrado polifásico exacto libre de aliasing."""
        if orig_sr == target_sr or samples.size == 0:
            return samples
        ratio = Fraction(target_sr, orig_sr).limit_denominator(1000)
        return resample_poly(samples, ratio.numerator, ratio.denominator).astype(np.float64)

    # ==========================================================================
    # 2. PIPELINE COMPLETO DE ANÁLISIS ACÚSTICO Y PROSÓDICO
    # ==========================================================================
    def process_and_analyze(
        self,
        input_path: Path | str,
        output_wav_path: Path | str,
        speaker_gender: Optional[str] = None,
    ) -> ProsodyAnalysisResult:
        """
        Normaliza un archivo de entrada a WAV 24kHz 16-bit Mono y ejecuta
        la batería completa de análisis acústico y prosódico.
        """
        wav_path, _ = self.normalize_to_wav_24k(input_path, output_wav_path)
        return self.analyze_wav(wav_path, speaker_gender=speaker_gender)

    def analyze_wav(
        self,
        wav_path: Path | str,
        speaker_gender: Optional[str] = None,
    ) -> ProsodyAnalysisResult:
        """
        Ejecuta la extracción de F0 en semitonos (30 puntos), formantes F1/F2,
        HNR/SNR y conteo heurístico de cortes glotales sobre un archivo WAV 24kHz.
        """
        path = Path(wav_path)
        samples, sr = sf.read(str(path), dtype="float64", always_2d=False)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=-1)

        duration_sec = float(samples.shape[0]) / float(sr) if sr > 0 else 0.0

        # Protección ante clips ultra-cortos (< 150 ms) donde las ventanas Gaussianas de Praat fallarían
        if duration_sec < 0.15:
            return ProsodyAnalysisResult(
                normalized_wav_path=path,
                duration_sec=_safe_float(duration_sec, 0.0, 3),
                sample_rate=sr,
                f0_curve_semitones=[0.0] * self.curve_points,
            )

        snd = parselmouth.Sound(str(path))
        gender_norm = (speaker_gender or "").strip().upper()

        if gender_norm == "M":
            pitch_floor, pitch_ceiling = 65.0, 300.0
            max_formant = 5000.0
        elif gender_norm == "F":
            pitch_floor, pitch_ceiling = 100.0, 500.0
            max_formant = 5500.0
        else:
            pitch_floor, pitch_ceiling = 75.0, 450.0
            max_formant = 5250.0

        # 1) F0 y Curva del "Cantadito" (30 puntos normalizados en Semitonos)
        pitch_obj, pitch_stats = self._extract_pitch_and_semitones(
            snd=snd,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )

        # 2) Formantes F1, F2 y nube del espacio vocálico
        f1_mean, f2_mean, vowel_points = self._extract_formants(
            snd=snd,
            pitch_obj=pitch_obj,
            max_formant=max_formant,
        )

        # 3) Relación Armónico-Ruido (HNR) y Señal-Ruido (SNR)
        hnr_db, snr_db = self._extract_hnr_and_snr(
            snd=snd,
            samples=samples,
            sr=sr,
            pitch_floor=pitch_floor,
        )

        # 4) Heurística de Cortes Glotales (< 20 ms)
        glottal_count, glottal_times = self._detect_glottal_stops(
            snd=snd,
            samples=samples,
            sr=sr,
            pitch_obj=pitch_obj,
        )

        return ProsodyAnalysisResult(
            normalized_wav_path=path,
            duration_sec=_safe_float(duration_sec, 0.0, 3),
            sample_rate=sr,
            f0_mean_hz=pitch_stats["f0_mean_hz"],
            f0_median_hz=pitch_stats["f0_median_hz"],
            f0_min_hz=pitch_stats["f0_min_hz"],
            f0_max_hz=pitch_stats["f0_max_hz"],
            f0_semitones_std=pitch_stats["f0_semitones_std"],
            f0_semitones_range=pitch_stats["f0_semitones_range"],
            f0_curve_semitones=pitch_stats["f0_curve_semitones"],
            f1_mean_hz=f1_mean,
            f2_mean_hz=f2_mean,
            vowel_space_points=vowel_points,
            hnr_db=hnr_db,
            snr_db=snr_db,
            voiced_fraction=pitch_stats["voiced_fraction"],
            glottal_stops_count=glottal_count,
            glottal_stop_timestamps=glottal_times,
        )

    # ==========================================================================
    # 3. SUB-MOTORES DE EXTRACCIÓN FONÉTICA Y PROSÓDICA
    # ==========================================================================
    def _extract_pitch_and_semitones(
        self,
        snd: parselmouth.Sound,
        pitch_floor: float,
        pitch_ceiling: float,
    ) -> tuple[Optional[parselmouth.Pitch], dict[str, Any]]:
        """
        Extrae el contorno F0 mediante autocorrelación de Boersma (Praat), filtra saltos
        espurios de octava, convierte a semitonos centrados en la mediana del hablante:
            ST(t) = 12 * log2(F0(t) / F0_mediana)
        y re-muestrea el tramo sonoro a exactamente `self.curve_points` (30) puntos.
        """
        empty_stats: dict[str, Any] = {
            "f0_mean_hz": 0.0,
            "f0_median_hz": 0.0,
            "f0_min_hz": 0.0,
            "f0_max_hz": 0.0,
            "f0_semitones_std": 0.0,
            "f0_semitones_range": 0.0,
            "f0_curve_semitones": [0.0] * self.curve_points,
            "voiced_fraction": 0.0,
        }

        try:
            pitch = snd.to_pitch_ac(
                time_step=0.01,
                pitch_floor=pitch_floor,
                max_number_of_candidates=15,
                very_accurate=True,
                silence_threshold=0.03,
                voicing_threshold=0.45,
                octave_cost=0.01,
                octave_jump_cost=0.35,
                voiced_unvoiced_cost=0.14,
                pitch_ceiling=pitch_ceiling,
            )
        except Exception as exc:
            logger.warning("No fue posible extraer Pitch en Praat: %s", exc)
            return None, empty_stats

        times = pitch.xs()
        f0_raw = pitch.selected_array["frequency"].astype(np.float64)
        if f0_raw.size == 0:
            return pitch, empty_stats

        voiced_mask = (f0_raw >= pitch_floor) & (f0_raw <= pitch_ceiling)
        voiced_fraction = float(np.mean(voiced_mask))

        voiced_times = times[voiced_mask]
        voiced_f0 = f0_raw[voiced_mask]

        if voiced_f0.size < 2:
            if voiced_f0.size == 1:
                val = _safe_float(voiced_f0[0], 0.0, 2)
                empty_stats.update(
                    {
                        "f0_mean_hz": val,
                        "f0_median_hz": val,
                        "f0_min_hz": val,
                        "f0_max_hz": val,
                        "voiced_fraction": _safe_float(voiced_fraction, 0.0, 3),
                    }
                )
            return pitch, empty_stats

        # Limpieza de glitches aislados de 1 trama (> 10 semitonos respecto a ambos vecinos)
        if voiced_f0.size >= 5:
            keep = np.ones(voiced_f0.shape[0], dtype=bool)
            for idx in range(1, voiced_f0.shape[0] - 1):
                st_prev = abs(12.0 * math.log2(voiced_f0[idx] / voiced_f0[idx - 1]))
                st_next = abs(12.0 * math.log2(voiced_f0[idx] / voiced_f0[idx + 1]))
                if st_prev > 10.0 and st_next > 10.0:
                    keep[idx] = False
            if np.sum(keep) >= 2:
                voiced_f0 = voiced_f0[keep]
                voiced_times = voiced_times[keep]

        f0_mean = float(np.mean(voiced_f0))
        f0_median = float(np.median(voiced_f0))
        f0_min = float(np.percentile(voiced_f0, 2))
        f0_max = float(np.percentile(voiced_f0, 98))

        # Transformación logarítmica a Semitonos respecto a la mediana tonal del enunciado
        st_voiced = 12.0 * np.log2(voiced_f0 / max(f0_median, 1e-6))
        st_std = float(np.std(st_voiced))
        st_range = float(np.percentile(st_voiced, 95) - np.percentile(st_voiced, 5))

        # Re-muestreo normalizado en el tiempo (inicio a fin de fonación) a 30 puntos
        t_start = float(voiced_times[0])
        t_end = float(voiced_times[-1])
        if t_end <= t_start + 1e-6:
            curve_30 = [0.0] * self.curve_points
        else:
            norm_grid = np.linspace(t_start, t_end, self.curve_points)
            interp_st = np.interp(norm_grid, voiced_times, st_voiced)

            # Suavizado binomial ligero [0.25, 0.50, 0.25] que preserva picos circunflejos
            if interp_st.size >= 3:
                smoothed = interp_st.copy()
                smoothed[1:-1] = (
                    0.25 * interp_st[:-2]
                    + 0.50 * interp_st[1:-1]
                    + 0.25 * interp_st[2:]
                )
                interp_st = smoothed

            curve_30 = [_safe_float(v, 0.0, 3) for v in interp_st.tolist()]

        stats = {
            "f0_mean_hz": _safe_float(f0_mean, 0.0, 2),
            "f0_median_hz": _safe_float(f0_median, 0.0, 2),
            "f0_min_hz": _safe_float(f0_min, 0.0, 2),
            "f0_max_hz": _safe_float(f0_max, 0.0, 2),
            "f0_semitones_std": _safe_float(st_std, 0.0, 3),
            "f0_semitones_range": _safe_float(st_range, 0.0, 3),
            "f0_curve_semitones": curve_30,
            "voiced_fraction": _safe_float(voiced_fraction, 0.0, 3),
        }
        return pitch, stats

    def _extract_formants(
        self,
        snd: parselmouth.Sound,
        pitch_obj: Optional[parselmouth.Pitch],
        max_formant: float = 5250.0,
        max_scatter_points: int = 35,
    ) -> tuple[float, float, list[dict[str, float]]]:
        """
        Extrae F1 y F2 mediante LPC de Burg exclusivamente en núcleos vocálicos
        (tramas sonoras de alta intensidad) para evitar contaminación por consonantes
        fricativas o silencios, devolviendo medias y puntos para el mapa vocálico F1/F2.
        """
        try:
            formant = snd.to_formant_burg(
                time_step=0.01,
                max_number_of_formants=5.0,
                maximum_formant=max_formant,
                window_length=0.025,
                pre_emphasis_from=50.0,
            )
            intensity = snd.to_intensity(
                minimum_pitch=75.0,
                time_step=0.01,
                subtract_mean=True,
            )
        except Exception as exc:
            logger.warning("No fue posible extraer Formantes en Praat: %s", exc)
            return 0.0, 0.0, []

        if pitch_obj is None:
            return 0.0, 0.0, []

        times = pitch_obj.xs()
        f0_vals = pitch_obj.selected_array["frequency"]

        voiced_intensities: list[float] = []
        voiced_candidates: list[float] = []
        for t, f0 in zip(times, f0_vals):
            if f0 > 0.0:
                val_db = intensity.get_value(float(t))
                if val_db is not None and not math.isnan(val_db):
                    voiced_intensities.append(float(val_db))
                    voiced_candidates.append(float(t))

        if not voiced_intensities:
            return 0.0, 0.0, []

        # Umbral energético para aislar núcleos silábicos vocálicos
        vowel_db_threshold = float(np.percentile(voiced_intensities, 85)) - 12.0

        valid_f1: list[float] = []
        valid_f2: list[float] = []
        valid_times: list[float] = []

        for t, int_db in zip(voiced_candidates, voiced_intensities):
            if int_db < vowel_db_threshold:
                continue
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            if f1 is None or f2 is None or math.isnan(f1) or math.isnan(f2):
                continue
            # Rangos fisiológicos de núcleos vocálicos humanos
            if 200.0 <= f1 <= 1100.0 and 700.0 <= f2 <= 3200.0 and (f2 - f1) >= 150.0:
                valid_f1.append(float(f1))
                valid_f2.append(float(f2))
                valid_times.append(float(t))

        if not valid_f1:
            return 0.0, 0.0, []

        f1_mean = _safe_float(float(np.mean(valid_f1)), 0.0, 1)
        f2_mean = _safe_float(float(np.mean(valid_f2)), 0.0, 1)

        # Sub-muestreo uniforme para graficar el espacio vocálico F1/F2 en Chart.js
        total_pts = len(valid_f1)
        if total_pts <= max_scatter_points:
            indices = range(total_pts)
        else:
            indices = np.linspace(0, total_pts - 1, max_scatter_points, dtype=int).tolist()

        scatter_points = [
            {
                "t": _safe_float(valid_times[i], 0.0, 3),
                "f1": _safe_float(valid_f1[i], 0.0, 1),
                "f2": _safe_float(valid_f2[i], 0.0, 1),
            }
            for i in indices
        ]
        return f1_mean, f2_mean, scatter_points

    def _extract_hnr_and_snr(
        self,
        snd: parselmouth.Sound,
        samples: np.ndarray,
        sr: int,
        pitch_floor: float = 75.0,
    ) -> tuple[float, float]:
        """
        Calcula:
        - HNR (Harmonics-to-Noise Ratio, en dB) mediante correlación cruzada en Praat.
        - SNR (Signal-to-Noise Ratio, en dB) comparando potencia RMS de tramas activas de voz
          frente al percentil 10 del piso de ruido de fondo.
        """
        hnr_db = 0.0
        try:
            harmonicity = snd.to_harmonicity_cc(
                time_step=0.01,
                minimum_pitch=pitch_floor,
                silence_threshold=0.1,
                periods_per_window=1.0,
            )
            hnr_values = harmonicity.values[0]
            valid_hnr = hnr_values[hnr_values > -100.0]
            if valid_hnr.size > 0:
                hnr_db = _safe_float(float(np.mean(valid_hnr)), 0.0, 2)
        except Exception as exc:
            logger.warning("No fue posible calcular HNR en Praat: %s", exc)

        # Cálculo de SNR por energía de tramas cortas (ventana 20 ms, salto 10 ms)
        frame_len = max(int(0.020 * sr), 1)
        hop_len = max(int(0.010 * sr), 1)
        snr_db = 0.0
        if samples.size >= frame_len:
            num_frames = 1 + (samples.size - frame_len) // hop_len
            frame_powers = np.empty(num_frames, dtype=np.float64)
            for idx in range(num_frames):
                start = idx * hop_len
                frame = samples[start : start + frame_len]
                frame_powers[idx] = float(np.mean(frame * frame))

            if num_frames >= 4:
                # Excluir tramas extremas si hay margen para no sesgar el piso de ruido con fades de 10ms
                inner_powers = frame_powers[1:-1] if num_frames >= 6 else frame_powers
                speech_thresh = float(np.percentile(inner_powers, 65))
                active_powers = inner_powers[inner_powers >= speech_thresh]
                signal_power = (
                    float(np.mean(active_powers))
                    if active_powers.size > 0
                    else float(np.mean(inner_powers))
                )
                # Percentil 3 captura el colchón de silencio (180ms) incluso en clips densos de 8-10s
                noise_power = max(float(np.percentile(inner_powers, 3)), 1e-10)
                raw_snr = 10.0 * math.log10(max(signal_power, 1e-10) / noise_power)
                snr_db = _safe_float(float(np.clip(raw_snr, 0.0, 60.0)), 0.0, 2)

        return hnr_db, snr_db

    def _detect_glottal_stops(
        self,
        snd: parselmouth.Sound,
        samples: np.ndarray,
        sr: int,
        pitch_obj: Optional[parselmouth.Pitch],
    ) -> tuple[int, list[float]]:
        """
        Heurística acústica para detección de cortes glotales / saltillos ([ʔ]),
        característicos del español yucateco (ruptura de hiatos y oclusiones glotalizadas):

        1. Contorno de intensidad de alta resolución temporal (paso = 2 ms, ventana ~16 ms).
        2. Caída abrupta de intensidad >= 10.5 dB ocurrida en estrictamente < 20 ms (<= 18 ms).
        3. Presencia confirmada de vibración laríngea (F0 > 0) en los 35 ms previos al cierre.
        4. Rechazo de consonantes fricativas/sibilantes (/s/, /f/, /x/) verificando que la tasa
           de cruces por cero (ZCR) durante el valle sea < 2800 Hz.
        5. Verificación de contexto: reanudación vocálica rápida tras el cierre glotal (15-160 ms,
           típico de [VʔV]) o corte glotal prepausal abrupto (>= 15.0 dB en <= 14 ms).
        6. Periodo refractario mínimo de 120 ms entre eventos detectados.
        """
        if snd.get_total_duration() < 0.08 or pitch_obj is None:
            return 0, []

        dt = 0.002  # 2 ms por trama para resolver transitorios < 20 ms
        try:
            intensity = snd.to_intensity(
                minimum_pitch=200.0,
                time_step=dt,
                subtract_mean=True,
            )
        except Exception as exc:
            logger.warning("No fue posible extraer intensidad de alta resolución: %s", exc)
            return 0, []

        times = intensity.xs()
        int_db = intensity.values[0].astype(np.float64)
        n_frames = int_db.size
        if n_frames < 15:
            return 0, []

        speech_peak_db = float(np.percentile(int_db, 92))
        noise_floor_db = float(np.percentile(int_db, 10))
        if (speech_peak_db - noise_floor_db) < 10.0:
            return 0, []

        max_drop_steps = 9  # 9 * 2 ms = 18 ms (< 20 ms)
        sharp_cutoff_steps = 7  # 7 * 2 ms = 14 ms
        min_refractory_sec = 0.120

        detected_timestamps: list[float] = []
        last_event_time = -1.0

        idx = 2
        while idx < n_frames - max_drop_steps - 2:
            t_onset = float(times[idx])

            if (t_onset - last_event_time) < min_refractory_sec:
                idx += 1
                continue

            current_db = float(int_db[idx])
            # Debe partir desde una vocal/sonante activa con energía suficiente
            if current_db < (speech_peak_db - 18.0) or current_db < (noise_floor_db + 10.0):
                idx += 1
                continue

            # Buscar el mínimo local en la ventana inmediata de <= 18 ms (< 20 ms)
            window_end = min(idx + max_drop_steps + 1, n_frames)
            sub_window = int_db[idx + 1 : window_end]
            rel_min_idx = int(np.argmin(sub_window))
            dip_idx = idx + 1 + rel_min_idx
            min_db = float(int_db[dip_idx])
            drop_db = current_db - min_db

            if drop_db < 10.5:
                idx += 1
                continue

            # Verificar fonación activa (F0 > 0) inmediatamente antes del cierre glotal
            pre_voiced = False
            for probe_t in np.linspace(max(0.0, t_onset - 0.035), max(0.0, t_onset - 0.002), 5):
                f0_val = pitch_obj.get_value_at_time(float(probe_t))
                if f0_val is not None and not math.isnan(f0_val) and f0_val > 0.0:
                    pre_voiced = True
                    break

            if not pre_voiced:
                idx += 1
                continue

            # Rechazar fricativas sordas (/s/, /f/, /x/) evaluando ZCR en el valle
            t_dip = float(times[dip_idx])
            s_start = max(0, int((t_dip - 0.006) * sr))
            s_end = min(samples.size, int((t_dip + 0.012) * sr))
            if (s_end - s_start) > 16:
                seg_dip = samples[s_start:s_end]
                zero_crossings = np.sum(np.abs(np.diff(np.signbit(seg_dip))))
                zcr_hz = float(zero_crossings) / (float(s_end - s_start) / float(sr))
                if zcr_hz > 2800.0:
                    idx += 1
                    continue

            # Evaluar contexto post-oclusión:
            # Caso A: Saltillo intervocálico / intragrupo [VʔV] -> recuperación rápida en 15-160 ms
            rec_start_idx = min(n_frames - 1, dip_idx + int(0.014 / dt))
            rec_end_idx = min(n_frames, dip_idx + int(0.160 / dt))
            is_intervocalic_glottal = False
            if rec_end_idx > rec_start_idx:
                post_max_db = float(np.max(int_db[rec_start_idx:rec_end_idx]))
                if (post_max_db - min_db) >= 7.5:
                    # Confirmar reanudación de sonoridad tras la oclusión
                    for probe_t in np.linspace(t_dip + 0.015, min(snd.get_total_duration(), t_dip + 0.160), 8):
                        f0_post = pitch_obj.get_value_at_time(float(probe_t))
                        if f0_post is not None and not math.isnan(f0_post) and f0_post > 0.0:
                            is_intervocalic_glottal = True
                            break

            # Caso B: Corte glotal prepausal abrupto (caída >= 15 dB en <= 14 ms desde núcleo fuerte)
            sharp_sub = int_db[idx + 1 : min(idx + sharp_cutoff_steps + 1, n_frames)]
            sharp_drop_db = current_db - float(np.min(sharp_sub)) if sharp_sub.size > 0 else 0.0
            is_abrupt_cutoff = (
                sharp_drop_db >= 15.0 and current_db >= (speech_peak_db - 12.0)
            )

            if is_intervocalic_glottal or is_abrupt_cutoff:
                event_time = _safe_float(t_onset, 0.0, 3)
                detected_timestamps.append(event_time)
                last_event_time = t_onset
                idx = dip_idx + int(0.040 / dt)
                continue

            idx += 1

        return len(detected_timestamps), detected_timestamps

    # ==========================================================================
    # 4. VALIDACIÓN FONÉTICO-ACÚSTICA FRENTE AL MOTOR G2P MAYA 1984
    # ==========================================================================
    def evaluate_maya_phonetic_compliance(
        self,
        prosody_res: ProsodyAnalysisResult,
        g2p_result: Any,
    ) -> float:
        """
        Evalúa cuantitativamente (0.0 a 100.0) qué tan bien coincide la señal física
        medida en Praat (`prosody_res`) con la estructura fonológica esperada por el
        motor G2P de la Norma de Agosto de 1984 (`g2p_result.expected_acoustic_profile`).

        Dimensiones ponderadas:
        1. Realización Glotal y Laringalizada (40 pts): compara `glottal_stops_count` y
           régimen de HNR (*creaky voice* en `V'V`) contra `expected_vocalic_glottal_drops`
           y `ejective_consonants`.
        2. Contraste Tonal y Excursión en Semitonos (30 pts): verifica picos positivos de F0
           en sílabas de tono alto (`V́V`) y estabilidad/descenso en vocales largas bajas (`VV`).
        3. Calidad de Fonación, SNR y Ritmo Silábico (30 pts): valida relación señal/ruido,
           fracción sonora y tasa de articulación plausible respecto a la longitud fonológica.
        """
        return evaluate_maya_phonetic_compliance(prosody_res, g2p_result)


def evaluate_maya_phonetic_compliance(
    prosody_res: ProsodyAnalysisResult,
    g2p_result: Any,
) -> float:
    """
    Función de evaluación de cumplimiento fonético-acústico (0.0 a 100.0) entre las
    mediciones de Praat (`ProsodyAnalysisResult`) y el perfil fonológico esperado (`MayaG2PResult`).
    """
    if prosody_res is None or g2p_result is None:
        return 0.0

    if prosody_res.duration_sec < 0.20 or prosody_res.voiced_fraction < 0.03:
        return 0.0

    if isinstance(g2p_result, dict):
        profile = g2p_result.get("expected_acoustic_profile") or g2p_result
    else:
        profile = getattr(g2p_result, "expected_acoustic_profile", None) or {}

    word_count = int(profile.get("word_count", 0) or 0)
    if word_count <= 0:
        return 0.0

    exp_glottal_total = int(profile.get("expected_glottal_events", 0) or 0)
    exp_vocalic_drops = int(profile.get("expected_vocalic_glottal_drops", 0) or 0)
    reart_vowels = int(profile.get("rearticulated_vowels", 0) or 0)
    ejectives = int(profile.get("ejective_consonants", 0) or 0)
    high_tones = int(profile.get("high_tone_syllables", 0) or 0)
    low_long = int(profile.get("low_tone_long_vowels", 0) or 0)
    long_sylls = int(profile.get("long_vowel_syllables", 0) or 0)
    short_sylls = int(profile.get("short_vowel_syllables", 0) or 0)

    detected_glottals = max(0, int(prosody_res.glottal_stops_count or 0))
    hnr_db = float(prosody_res.hnr_db or 0.0)
    snr_db = float(prosody_res.snr_db or 0.0)
    voiced_frac = float(prosody_res.voiced_fraction or 0.0)
    st_range = float(prosody_res.f0_semitones_range or 0.0)
    st_std = float(prosody_res.f0_semitones_std or 0.0)
    curve = list(prosody_res.f0_curve_semitones or [])
    max_st = max(curve) if curve else 0.0
    min_st = min(curve) if curve else 0.0

    # --- 1) Cumplimiento Glotal y Eyectivo (0.0 a 40.0 pts) ---
    # Las caídas vocálicas (`V'V`, `V'`) impactan directamente el detector de intensidad <20ms;
    # las eyectivas (`C'`) aportan transitorios gloto-orales parciales (~0.5 peso efectivo).
    target_drops = float(exp_vocalic_drops) + 0.50 * float(ejectives)
    if exp_glottal_events_zero := (exp_glottal_total == 0):
        # En oraciones sin glotalización fonológica, pocos o ningún corte abrupto es lo esperado
        excess = max(0, detected_glottals - 1)
        glottal_score = max(18.0, 40.0 - 6.5 * float(excess))
    else:
        effective_target = max(1.0, target_drops)
        rel_diff = abs(float(detected_glottals) - effective_target) / max(
            effective_target, 1.5
        )
        glottal_score = 40.0 * max(0.28, 1.0 - 0.52 * rel_diff)

        # En maya yucateco real (Frazier 2009), `V'V` suele realizarse como voz laringalizada
        # (*creaky voice*, HNR moderado 5–17 dB) sin silencio oclusivo completo:
        if reart_vowels > 0 and detected_glottals < exp_vocalic_drops:
            if 5.0 <= hnr_db <= 17.5:
                glottal_score = min(40.0, glottal_score + 8.5)

    # --- 2) Cumplimiento Tonal (`V́V` Tono Alto vs `VV` Tono Bajo) (0.0 a 30.0 pts) ---
    if high_tones > 0:
        # Las vocales largas de tono alto (`/áː, éː, íː, óː, úː/`) exigen excursión positiva de F0
        peak_factor = min(1.0, max(0.0, (max_st + 0.3) / 2.1))
        range_factor = min(1.0, max(0.0, st_range / 3.0))
        tonal_score = 10.0 + 20.0 * (0.60 * peak_factor + 0.40 * range_factor)
    elif low_long > 0:
        # Vocales largas de tono bajo (`/àː.../`) favorecen registro sostenido o descenso grave
        low_dip_ok = 1.0 if (min_st <= -0.35 or st_std <= 3.8) else 0.65
        tonal_score = 12.0 + 18.0 * low_dip_ok
    else:
        tonal_score = 26.0 if prosody_res.f0_mean_hz > 55.0 else 14.0

    # --- 3) Calidad de Señal, Sonoridad y Duración Fonológica (0.0 a 30.0 pts) ---
    snr_pts = 12.0 * min(1.0, max(0.15, snr_db / 18.0))
    if 0.22 <= voiced_frac <= 0.92:
        voicing_pts = 10.0
    elif voiced_frac >= 0.10:
        voicing_pts = 6.5
    else:
        voicing_pts = 2.5

    total_nuclei = max(1, long_sylls + short_sylls + reart_vowels)
    expected_min_dur = max(0.35, 0.11 * float(total_nuclei))
    expected_max_dur = max(2.5, 0.68 * float(total_nuclei) + 1.2)
    if expected_min_dur <= prosody_res.duration_sec <= expected_max_dur:
        pacing_pts = 8.0
    else:
        pacing_pts = 4.5

    quality_score = snr_pts + voicing_pts + pacing_pts
    total_score = glottal_score + tonal_score + quality_score
    return _safe_float(float(np.clip(total_score, 0.0, 100.0)), 0.0, 2)
