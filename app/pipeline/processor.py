"""
Pipeline de Ingesta de Audio, Segmentación Inteligente (Silero VAD) y Transcripción
Automática Fiel al Acento (Faster-Whisper) para AccentLab Web.

Características clave:
1. Singleton / Lazy Loading thread-safe para mantener el consumo de VRAM < 2 GB:
   - `faster-whisper` (por defecto `small`, parametrizable vía `ACCENTLAB_WHISPER_MODEL`)
     con `device="cuda"` + `compute_type="int8_float16"` si hay GPU NVIDIA operativa,
     y fallback automático (tanto en carga como en inferencia) a `device="cpu"` + `int8`.
   - Registro automático de DLLs de CUDA empaquetadas en PyTorch sobre Windows para CTranslate2.
   - `silero-vad` cargado una sola vez en CPU (0 MB de VRAM, inferencia en milisegundos).
2. Segmentación prosódica inteligente (`process_long_audio`):
   - Detecta actividad de voz a 16 kHz con Silero VAD y corta sobre la señal maestra a 24 kHz.
   - Agrupa islas de voz y sub-divide monólogos largos por micro-pausas o valles mínimos de
     energía RMS para producir clips estrictamente entre 3.0 y 10.0 segundos sin cortar palabras,
     añadiendo padding protector de 180 ms (configurable 150-200 ms) y fade anti-click de 10 ms.
3. Persistencia transaccional en SQLite (`AudioSegment` + `AcousticMetrics`) con limpieza de
   archivos WAV huérfanos ante cualquier fallo.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sqlalchemy.orm import Session

from app.analyzers.prosody import ProsodyAnalysisResult, ProsodyAnalyzer
from app.database import BASE_DIR, WAVS_DIR
from app.linguistics import MayaG2P1984, MayaGrammaticalAnalyzer
from app.models import AcousticMetrics, AudioSegment, Speaker

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuración por variables de entorno y prompts fonéticos (Español y Maya 1984)
# ==============================================================================
DEFAULT_WHISPER_MODEL: str = os.getenv("ACCENTLAB_WHISPER_MODEL", "small").strip()
DEFAULT_WHISPER_DEVICE: str = os.getenv("ACCENTLAB_DEVICE", "auto").strip().lower()

DEFAULT_REGIONAL_PROMPT: str = os.getenv(
    "ACCENTLAB_INITIAL_PROMPT",
    (
        "Transcripción literal de voz en español mexicano y español yucateco coloquial, "
        "conservando modismos, muletillas y expresiones regionales sin corregir la gramática: "
        "mare, puchis, boxito, tuch, xix, huiro, chichí, fo, jach, préstame, "
        "¿qué pasó, nené?, lo busqué."
    ),
)

DEFAULT_MAYA_1984_PROMPT: str = os.getenv(
    "ACCENTLAB_MAYA_INITIAL_PROMPT",
    (
        "Ma'alob k'iin, Yutsil ja'atskab, Ma'alob u sáastal k'iin, "
        "Ma'alob chi'inil k'iin, Ma'alob áak'ab, Bix a beel, Toj in wóol, "
        "Dios bo'otik, Nib óolal, Ba'ax ka wa'alik, Tu'ux ka bin, "
        "k'a'ana'an, ja'atskab, áak'ab, ts'o'ok, ch'íich', k'áak', "
        "maaya t'aan, in wotoch, ko'ox xíimbal."
    ),
)

VALID_LANGUAGE_MODES: set[str] = {"auto", "yua", "es_yuc"}


_DLL_DIR_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRS: set[str] = set()


def _register_windows_torch_cuda_dlls() -> None:
    """
    En Windows, CTranslate2 (faster-whisper) requiere cublas64_12.dll / cudnn*.dll.
    Registra automáticamente en el buscador de DLLs del proceso tanto los paquetes
    `nvidia/*/bin` (nvidia-cublas-cu12, nvidia-cudnn-cu12) como `torch/lib`, conservando
    las referencias de `os.add_dll_directory` para evitar que el GC las invalide.
    """
    if os.name != "nt":
        return

    candidate_dirs: list[Path] = []
    try:
        import sys

        site_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        if site_nvidia.is_dir():
            for bin_dir in site_nvidia.glob("*/bin"):
                if bin_dir.is_dir():
                    candidate_dirs.append(bin_dir.resolve())
    except Exception as exc:
        logger.debug("No se pudo inspeccionar site-packages/nvidia: %s", exc)

    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            candidate_dirs.append(torch_lib)
    except Exception as exc:
        logger.debug("No se pudo localizar torch/lib: %s", exc)

    for dll_dir in candidate_dirs:
        dir_str = str(dll_dir)
        if dir_str in _REGISTERED_DLL_DIRS:
            continue
        _REGISTERED_DLL_DIRS.add(dir_str)
        try:
            if hasattr(os, "add_dll_directory"):
                _DLL_DIR_HANDLES.append(os.add_dll_directory(dir_str))
            current_path = os.environ.get("PATH", "")
            if dir_str not in current_path:
                os.environ["PATH"] = f"{dir_str}{os.pathsep}{current_path}"
        except Exception as exc:
            logger.debug("No se pudo registrar directorio DLL %s: %s", dir_str, exc)


_register_windows_torch_cuda_dlls()


def _sanitize_code(code: str) -> str:
    """Normaliza el código del hablante para nombres de archivo seguros en cualquier SO."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", code.strip())
    return cleaned or "SPK"


@dataclass
class TranscriptionResult:
    """Resultado estructurado de la transcripción con faster-whisper."""

    text: str
    confidence: Optional[float]
    language: str = "es"


class AudioIngestionPipeline:
    """
    Orquestador Singleton con carga perezosa (Lazy Loading) de Silero VAD,
    Faster-Whisper y ProsodyAnalyzer para ingesta directa y segmentación de audios largos.
    """

    VAD_SR: int = 16000
    MASTER_SR: int = 24000

    def __init__(
        self,
        whisper_model_name: str = DEFAULT_WHISPER_MODEL,
        device_preference: str = DEFAULT_WHISPER_DEVICE,
        initial_prompt: str = DEFAULT_REGIONAL_PROMPT,
    ) -> None:
        self.whisper_model_name = whisper_model_name
        self.device_preference = device_preference
        self.initial_prompt = initial_prompt

        self.analyzer = ProsodyAnalyzer(target_sr=self.MASTER_SR)
        self.maya_g2p = MayaG2P1984()
        self.maya_grammar = MayaGrammaticalAnalyzer(g2p=self.maya_g2p)

        self._whisper_model: Any = None
        self._whisper_active_device: str = "uninitialized"
        self._whisper_lock = threading.Lock()
        self._whisper_infer_lock = threading.Lock()

        self._vad_model: Any = None
        self._get_speech_timestamps: Optional[Callable[..., list[dict[str, Any]]]] = None
        self._vad_lock = threading.Lock()

    # ==========================================================================
    # 1. LAZY LOADING DE MODELOS ML (VRAM < 2 GB + FALLBACK CUDA -> CPU)
    # ==========================================================================
    def _ensure_whisper_loaded(self, force_cpu: bool = False) -> Any:
        """
        Inicializa `faster-whisper` bajo demanda.
        Intenta `cuda` (`int8_float16`) si CTranslate2 o PyTorch detectan GPU NVIDIA,
        y cae automáticamente a `cpu` (`int8`) si no hay GPU o faltan librerías CUDA.
        """
        with self._whisper_lock:
            if self._whisper_model is not None and not force_cpu:
                return self._whisper_model

            _register_windows_torch_cuda_dlls()
            from faster_whisper import WhisperModel

            use_cuda = False
            if not force_cpu and self.device_preference in ("auto", "cuda"):
                try:
                    import ctranslate2

                    use_cuda = ctranslate2.get_cuda_device_count() > 0
                except Exception:
                    use_cuda = False
                if not use_cuda:
                    try:
                        import torch

                        use_cuda = bool(torch.cuda.is_available())
                    except Exception:
                        use_cuda = False

            if use_cuda:
                try:
                    logger.info(
                        "Cargando faster-whisper '%s' en CUDA (compute_type=int8_float16)...",
                        self.whisper_model_name,
                    )
                    self._whisper_model = WhisperModel(
                        self.whisper_model_name,
                        device="cuda",
                        compute_type="int8_float16",
                    )
                    self._whisper_active_device = "cuda"
                    return self._whisper_model
                except Exception as cuda_exc:
                    logger.warning(
                        "Fallo al iniciar faster-whisper en CUDA (%s). Activando fallback CPU int8.",
                        cuda_exc,
                    )

            cpu_threads = max(1, (os.cpu_count() or 4) // 2)
            logger.info(
                "Cargando faster-whisper '%s' en CPU (compute_type=int8, threads=%d)...",
                self.whisper_model_name,
                cpu_threads,
            )
            self._whisper_model = WhisperModel(
                self.whisper_model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
            )
            self._whisper_active_device = "cpu"
            return self._whisper_model

    def _ensure_vad_loaded(self) -> tuple[Any, Callable[..., list[dict[str, Any]]]]:
        """
        Carga Silero VAD en CPU bajo demanda (consumo de VRAM = 0 MB).
        Soporta el paquete oficial `silero-vad` con respaldo vía `torch.hub`.
        """
        if self._vad_model is not None and self._get_speech_timestamps is not None:
            return self._vad_model, self._get_speech_timestamps

        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            logger.info("Cargando modelo oficial silero-vad en CPU...")
            model = load_silero_vad(onnx=False)
            self._vad_model = model
            self._get_speech_timestamps = get_speech_timestamps
            return model, get_speech_timestamps
        except Exception as pkg_exc:
            logger.warning(
                "Paquete 'silero_vad' no disponible directamente (%s). Cargando vía torch.hub...",
                pkg_exc,
            )
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            get_speech_timestamps = utils[0]
            self._vad_model = model
            self._get_speech_timestamps = get_speech_timestamps
            return model, get_speech_timestamps

    # ==========================================================================
    # 2. TRANSCRIPCIÓN FIEL AL ACENTO (FASTER-WHISPER)
    # ==========================================================================
    def _load_audio_16k_float32(self, wav_path: Path | str) -> np.ndarray:
        """
        Lee un archivo de audio como arreglo 1D `np.float32` a 16,000 Hz en memoria.
        Entregar `np.ndarray` directo a `faster-whisper` evita incompatibilidades de
        `decode_audio` con versiones nuevas de PyAV (`metadata_errors`) y cierra de
        inmediato el descriptor de archivo en Windows.
        """
        path = Path(wav_path)
        try:
            samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if samples.ndim > 1:
                samples = np.mean(samples, axis=-1, dtype=np.float32)
        except Exception:
            samples_24k = self.analyzer.decode_to_mono_24k(path)
            samples = samples_24k.astype(np.float32)
            sr = self.MASTER_SR

        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)
        if sr == self.VAD_SR:
            return np.ascontiguousarray(samples, dtype=np.float32)
        if sr == self.MASTER_SR:
            resampled = resample_poly(samples, up=2, down=3)
            return np.ascontiguousarray(resampled, dtype=np.float32)
        resampled = ProsodyAnalyzer._resample_mono(
            samples.astype(np.float64), sr, self.VAD_SR
        )
        return np.ascontiguousarray(resampled, dtype=np.float32)

    def transcribe_wav(
        self,
        wav_path: Path | str,
        has_voicing: bool = True,
        custom_initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe un clip WAV conservando giros regionales mexicanos/yucatecos o Maya 1984
        y estima `asr_confidence` mediante `exp(avg_logprob)` ponderado por duración.
        Si `has_voicing` es False (silencio/ruido puro sin fonación), evita alucinaciones
        del `initial_prompt` retornando cadena vacía de inmediato.
        """
        if not has_voicing:
            return TranscriptionResult(text="", confidence=0.0, language="es")

        audio_16k = self._load_audio_16k_float32(wav_path)
        if audio_16k.size == 0:
            return TranscriptionResult(text="", confidence=0.0, language="es")

        active_prompt = (
            custom_initial_prompt
            if custom_initial_prompt is not None
            else self.initial_prompt
        )

        with self._whisper_infer_lock:
            model = self._ensure_whisper_loaded(force_cpu=False)
            try:
                return self._run_whisper_inference(
                    model, audio_16k, initial_prompt=active_prompt
                )
            except RuntimeError as runtime_exc:
                # Si CTranslate2 falla en la primera inferencia CUDA por falta de cublas/cudnn:
                if self._whisper_active_device == "cuda":
                    logger.warning(
                        "Error de ejecución CUDA durante transcripción (%s). Reintentando en CPU int8...",
                        runtime_exc,
                    )
                    cpu_model = self._ensure_whisper_loaded(force_cpu=True)
                    return self._run_whisper_inference(
                        cpu_model, audio_16k, initial_prompt=active_prompt
                    )
                raise

    def _run_whisper_inference(
        self,
        model: Any,
        audio_16k: np.ndarray,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """Ejecuta `model.transcribe` y consolida texto + confianza probabilística filtrando alucinaciones."""
        segments_iter, info = model.transcribe(
            audio_16k,
            language="es",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt or self.initial_prompt,
            temperature=0.0,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        texts: list[str] = []
        weighted_logprob_sum = 0.0
        total_weight = 0.0

        for seg in segments_iter:
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue
            no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            avg_lp = float(seg.avg_logprob) if seg.avg_logprob is not None else -1.0
            comp_ratio = float(getattr(seg, "compression_ratio", 1.0) or 1.0)
            # Rechazar alucinaciones típicas en silencio o bucles repetitivos
            if (no_speech_prob > 0.72 and avg_lp < -0.75) or comp_ratio > 2.6:
                continue

            texts.append(seg_text)
            seg_dur = max(float(seg.end - seg.start), 0.05)
            weighted_logprob_sum += avg_lp * seg_dur
            total_weight += seg_dur

        full_text = " ".join(texts).strip()
        if not full_text or total_weight <= 0.0:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=getattr(info, "language", "es") or "es",
            )

        mean_logprob = weighted_logprob_sum / total_weight
        confidence = float(np.clip(math.exp(mean_logprob), 0.0, 1.0))
        return TranscriptionResult(
            text=full_text,
            confidence=round(confidence, 4),
            language=getattr(info, "language", "es") or "es",
        )

    # ==========================================================================
    # 3. SEGMENTACIÓN INTELIGENTE CON SILERO VAD (3.0s - 10.0s SIN CORTAR PALABRAS)
    # ==========================================================================
    def detect_smart_segments(
        self,
        samples_24k: np.ndarray,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 10.0,
        pad_ms: int = 180,
    ) -> list[tuple[float, float]]:
        """
        Detecta actividad vocal sobre una copia re-muestreada a 16 kHz usando Silero VAD
        y construye ventanas temporales en segundos `[inicio, fin]` de 3.0 a 10.0 s con
        padding protector (por defecto 180 ms) sin truncar palabras a la mitad.
        """
        total_dur_sec = float(samples_24k.shape[0]) / float(self.MASTER_SR)
        if total_dur_sec < min_duration_sec:
            return []

        pad_sec = max(0.0, float(pad_ms) / 1000.0)
        max_core_sec = max(min_duration_sec, max_duration_sec - 2.0 * pad_sec)

        # Re-muestreo exacto 24 kHz -> 16 kHz (factor 2/3) para Silero VAD
        samples_16k = resample_poly(samples_24k, up=2, down=3).astype(np.float32)

        import torch

        with self._vad_lock:
            vad_model, get_speech_timestamps = self._ensure_vad_loaded()
            if hasattr(vad_model, "reset_states"):
                vad_model.reset_states()

            audio_tensor = torch.from_numpy(samples_16k)
            with torch.no_grad():
                raw_ts = get_speech_timestamps(
                    audio_tensor,
                    vad_model,
                    threshold=0.45,
                    sampling_rate=self.VAD_SR,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=280,  # Preserva cierres oclusivos y saltillos [ʔ]
                    speech_pad_ms=0,
                    return_seconds=False,
                )

            if not raw_ts:
                return []

            # Convertir muestras a segundos y subdividir islas continuas > max_core_sec
            elementary_islands: list[tuple[float, float]] = []
            for item in raw_ts:
                s_sec = float(item["start"]) / float(self.VAD_SR)
                e_sec = float(item["end"]) / float(self.VAD_SR)
                if (e_sec - s_sec) <= max_core_sec:
                    elementary_islands.append((s_sec, e_sec))
                else:
                    sub_islands = self._split_long_island(
                        samples_16k=samples_16k,
                        samples_24k=samples_24k,
                        start_sec=s_sec,
                        end_sec=e_sec,
                        max_core_sec=max_core_sec,
                        vad_model=vad_model,
                        get_speech_timestamps=get_speech_timestamps,
                    )
                    elementary_islands.extend(sub_islands)

        if not elementary_islands:
            return []

        # Agrupar islas consecutivas respetando fronteras prosódicas y rango [3.0s, 10.0s]
        grouped_cores = self._pack_speech_islands(
            islands=elementary_islands,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            pad_sec=pad_sec,
        )

        # Aplicar padding exterior (180 ms) sin invadir islas vecinas
        final_windows: list[tuple[float, float]] = []
        n_groups = len(grouped_cores)
        for idx, (c_start, c_end) in enumerate(grouped_cores):
            left_bound = 0.0
            if idx > 0:
                prev_end = grouped_cores[idx - 1][1]
                left_bound = 0.5 * (prev_end + c_start)

            right_bound = total_dur_sec
            if idx < n_groups - 1:
                next_start = grouped_cores[idx + 1][0]
                right_bound = 0.5 * (c_end + next_start)

            padded_start = max(left_bound, c_start - pad_sec)
            padded_end = min(right_bound, c_end + pad_sec)

            # Si una intervención válida quedó ligeramente bajo min_duration_sec (ej. 2.4s-2.9s)
            # y existe silencio natural adyacente disponible, expandir suavemente hasta min_duration_sec.
            clip_len = padded_end - padded_start
            core_len = c_end - c_start
            if (
                clip_len < min_duration_sec
                and core_len >= max(1.4, 0.50 * min_duration_sec)
                and (right_bound - left_bound) >= min_duration_sec
                and (min_duration_sec - clip_len) <= 1.30
            ):
                deficit = min_duration_sec - clip_len
                grow_left = min(padded_start - left_bound, 0.5 * deficit)
                grow_right = min(right_bound - padded_end, deficit - grow_left)
                rem = deficit - (grow_left + grow_right)
                if rem > 0:
                    grow_left = min(padded_start - left_bound, grow_left + rem)
                padded_start -= grow_left
                padded_end += grow_right
                clip_len = padded_end - padded_start

            # Si el padding excedió ligeramente max_duration_sec, recortar simétricamente el colchón
            if clip_len > max_duration_sec:
                excess = clip_len - max_duration_sec
                padded_start += 0.5 * excess
                padded_end -= 0.5 * excess

            final_len = padded_end - padded_start
            if (min_duration_sec - 1e-3) <= final_len <= (max_duration_sec + 1e-3):
                final_windows.append((round(padded_start, 4), round(padded_end, 4)))

        return final_windows

    def _split_long_island(
        self,
        samples_16k: np.ndarray,
        samples_24k: np.ndarray,
        start_sec: float,
        end_sec: float,
        max_core_sec: float,
        vad_model: Any,
        get_speech_timestamps: Callable[..., list[dict[str, Any]]],
    ) -> list[tuple[float, float]]:
        """
        Divide un bloque continuo de habla > ~9.6 s sin cortar sílabas:
        1) Ejecuta un segundo pase de Silero VAD más sensible a micro-pausas (110 ms).
        2) Si algún fragmento aún excede `max_core_sec`, corta en el valle de mínima
           energía RMS (ventana de 25 ms) dentro de la zona segura [4.5 s, max_core_sec].
        """
        import torch

        s16 = max(0, int(round(start_sec * self.VAD_SR)))
        e16 = min(samples_16k.shape[0], int(round(end_sec * self.VAD_SR)))
        sub_tensor = torch.from_numpy(samples_16k[s16:e16])

        refined: list[tuple[float, float]] = []
        if sub_tensor.numel() > 0:
            if hasattr(vad_model, "reset_states"):
                vad_model.reset_states()
            with torch.no_grad():
                try:
                    sub_ts = get_speech_timestamps(
                        sub_tensor,
                        vad_model,
                        threshold=0.56,
                        sampling_rate=self.VAD_SR,
                        min_speech_duration_ms=250,
                        max_speech_duration_s=max_core_sec,
                        min_silence_duration_ms=110,
                        speech_pad_ms=0,
                        return_seconds=False,
                    )
                except TypeError:
                    # Compatibilidad por si una versión antigua no soporta max_speech_duration_s
                    sub_ts = get_speech_timestamps(
                        sub_tensor,
                        vad_model,
                        threshold=0.56,
                        sampling_rate=self.VAD_SR,
                        min_speech_duration_ms=250,
                        min_silence_duration_ms=110,
                        speech_pad_ms=0,
                        return_seconds=False,
                    )
            for item in sub_ts:
                sub_s = start_sec + float(item["start"]) / float(self.VAD_SR)
                sub_e = start_sec + float(item["end"]) / float(self.VAD_SR)
                refined.append((sub_s, sub_e))

        if not refined:
            refined = [(start_sec, end_sec)]

        # Garantizar que ningún sub-bloque supere max_core_sec buscando valles RMS
        safe_pieces: list[tuple[float, float]] = []
        for piece_s, piece_e in refined:
            curr_s = piece_s
            while (piece_e - curr_s) > max_core_sec:
                split_t = self._find_lowest_energy_point_24k(
                    samples_24k=samples_24k,
                    search_start_sec=curr_s + min(4.5, 0.55 * max_core_sec),
                    search_end_sec=curr_s + max_core_sec,
                )
                safe_pieces.append((curr_s, split_t))
                curr_s = split_t
            if (piece_e - curr_s) >= 0.25:
                safe_pieces.append((curr_s, piece_e))

        return safe_pieces

    def _find_lowest_energy_point_24k(
        self,
        samples_24k: np.ndarray,
        search_start_sec: float,
        search_end_sec: float,
    ) -> float:
        """
        Encuentra el instante de menor energía acústica RMS (ventana 25 ms, salto 5 ms)
        para efectuar un corte limpio en frontera inter-léxica / oclusiva cuando un
        hablante encadena más de 10 segundos continuos sin pausa respiratoria.
        """
        if search_end_sec <= search_start_sec:
            return search_end_sec

        win_samples = int(0.025 * self.MASTER_SR)  # 600 muestras @ 24kHz
        hop_samples = int(0.005 * self.MASTER_SR)  # 120 muestras @ 24kHz
        s_idx = max(0, int(round(search_start_sec * self.MASTER_SR)))
        e_idx = min(samples_24k.shape[0], int(round(search_end_sec * self.MASTER_SR)))

        if (e_idx - s_idx) <= win_samples:
            return 0.5 * (search_start_sec + search_end_sec)

        best_idx = s_idx
        min_energy = float("inf")
        for pos in range(s_idx, e_idx - win_samples + 1, hop_samples):
            frame = samples_24k[pos : pos + win_samples]
            energy = float(np.mean(frame * frame))
            if energy < min_energy:
                min_energy = energy
                best_idx = pos + (win_samples // 2)

        return float(best_idx) / float(self.MASTER_SR)

    @staticmethod
    def _pack_speech_islands(
        islands: list[tuple[float, float]],
        min_duration_sec: float,
        max_duration_sec: float,
        pad_sec: float,
    ) -> list[tuple[float, float]]:
        """
        Empaqueta islas de habla en clips prosódicos óptimos de [3.0, 10.0] s:
        - Une islas cortas (< 3.0 s) con la siguiente si el silencio intermedio <= 2.2 s
          y la duración total con padding no rebasa `max_duration_sec`.
        - Si el clip actual ya es >= 4.2 s y aparece una pausa oracional clara (> 0.55 s),
          cierra el clip para respetar fronteras sintácticas (salvo que la siguiente isla
          sea muy breve y quedaría huérfana).
        - Rescata fragmentos cortos remanentes fusionándolos con su vecino más cercano si caben.
        """
        if not islands:
            return []

        packed: list[tuple[float, float]] = []
        cur_s, cur_e = islands[0]

        for idx in range(1, len(islands)):
            nxt_s, nxt_e = islands[idx]
            gap = nxt_s - cur_e
            cur_padded_dur = (cur_e - cur_s) + 2.0 * pad_sec
            merged_padded_dur = (nxt_e - cur_s) + 2.0 * pad_sec
            nxt_padded_dur = (nxt_e - nxt_s) + 2.0 * pad_sec

            can_fit = merged_padded_dur <= max_duration_sec
            needs_more_dur = cur_padded_dur < min_duration_sec and gap <= 2.2
            natural_continuation = gap <= 0.55 and merged_padded_dur <= 8.8
            rescue_last_tiny = (
                idx == len(islands) - 1
                and nxt_padded_dur < min_duration_sec
                and gap <= 2.0
            )

            if can_fit and (needs_more_dur or natural_continuation or rescue_last_tiny):
                cur_e = nxt_e
            else:
                packed.append((cur_s, cur_e))
                cur_s, cur_e = nxt_s, nxt_e

        packed.append((cur_s, cur_e))

        # Pase de rescate para fragmentos < min_duration_sec que puedan unirse a un vecino
        if len(packed) >= 2:
            rescued: list[tuple[float, float]] = []
            i = 0
            while i < len(packed):
                s_i, e_i = packed[i]
                dur_i = (e_i - s_i) + 2.0 * pad_sec
                if dur_i < min_duration_sec:
                    # Intentar fusionar con el clip previo ya rescatado
                    if rescued:
                        prev_s, prev_e = rescued[-1]
                        if (
                            (s_i - prev_e) <= 2.2
                            and ((e_i - prev_s) + 2.0 * pad_sec) <= max_duration_sec
                        ):
                            rescued[-1] = (prev_s, e_i)
                            i += 1
                            continue
                    # Intentar fusionar hacia adelante con el siguiente clip
                    if (i + 1) < len(packed):
                        nxt_s, nxt_e = packed[i + 1]
                        if (
                            (nxt_s - e_i) <= 2.2
                            and ((nxt_e - s_i) + 2.0 * pad_sec) <= max_duration_sec
                        ):
                            packed[i + 1] = (s_i, nxt_e)
                            i += 1
                            continue
                rescued.append((s_i, e_i))
                i += 1
            packed = rescued

        return packed

    @staticmethod
    def _apply_edge_fade(samples: np.ndarray, sr: int, fade_ms: float = 10.0) -> np.ndarray:
        """Aplica una rampa suave de 10 ms en los bordes para eliminar chasquidos de corte."""
        fade_len = int(round((fade_ms / 1000.0) * sr))
        if fade_len <= 1 or samples.shape[0] < (2 * fade_len):
            return samples
        out = samples.copy()
        ramp_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float64)
        ramp_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float64)
        out[:fade_len] *= ramp_in
        out[-fade_len:] *= ramp_out
        return out

    # ==========================================================================
    # 4. EVALUACIÓN LINGÜÍSTICA MAYA 1984 Y FLUJOS DE INGESTA HACIA SQLITE
    # ==========================================================================
    @staticmethod
    def _normalize_language_mode(language_mode: Optional[str]) -> str:
        mode = (language_mode or "auto").strip().lower()
        return mode if mode in VALID_LANGUAGE_MODES else "auto"

    def _resolve_segment_linguistics(
        self,
        prosody_res: ProsodyAnalysisResult,
        asr_res: TranscriptionResult,
        prompt_mode: str,
        prompt_text: Optional[str],
        language_mode: str,
    ) -> dict[str, Any]:
        """
        Coordina la transcripción final, normalización a Norma de Agosto de 1984,
        análisis G2P (`MayaG2P1984`), glosa morfosintáctica (`MayaGrammaticalAnalyzer`)
        y puntuación de cumplimiento acústico en Praat (`phonetic_alignment_score`).
        """
        lang_mode = self._normalize_language_mode(language_mode)
        clean_prompt = (prompt_text or "").strip()
        prompt_lang = (
            self.maya_grammar.detect_language_code(clean_prompt)
            if clean_prompt
            else "es_yuc"
        )
        is_maya_prompt = bool(clean_prompt) and prompt_lang in ("yua", "yua_es")
        raw_asr_text = (asr_res.text or "").strip()

        has_valid_phonation = (prosody_res.voiced_fraction >= 0.05) and (
            bool(raw_asr_text) or prosody_res.snr_db >= 7.0
        )

        # Si es modo `trap_phrase` con oración maya conocida (o modo sesión `yua`),
        # preservar la ortografía 1984 de referencia cuando hay fonación válida real,
        # evitando que faster-whisper hispanice las eyectivas/apóstrofos.
        if (
            prompt_mode == "trap_phrase"
            and clean_prompt
            and (is_maya_prompt or lang_mode == "yua")
        ):
            final_transcript = (
                self.maya_g2p.normalize_orthography_1984(clean_prompt)
                if has_valid_phonation
                else ""
            )
        else:
            if (lang_mode == "yua" or is_maya_prompt) and raw_asr_text:
                final_transcript = self.maya_g2p.normalize_orthography_1984(raw_asr_text)
            else:
                final_transcript = raw_asr_text

        if final_transcript:
            gram_analysis = self.maya_grammar.analyze(final_transcript)
            g2p_res = self.maya_g2p.phonemize_sentence(final_transcript)
            detected_lang = gram_analysis.get("language_code", "es_yuc")

            if lang_mode == "yua":
                effective_lang = "yua_es" if detected_lang == "yua_es" else "yua"
            elif lang_mode == "es_yuc" and not is_maya_prompt:
                effective_lang = "yua_es" if detected_lang == "yua_es" else "es_yuc"
            else:
                effective_lang = detected_lang

            gram_analysis["language_code"] = effective_lang
            alignment_score = self.analyzer.evaluate_maya_phonetic_compliance(
                prosody_res, g2p_res
            )
            ipa_str = g2p_res.ipa_transcription
            tts_tokens_str = g2p_res.tts_phonetic_tokens
        else:
            effective_lang = "yua" if (lang_mode == "yua" or is_maya_prompt) else "es_yuc"
            gram_analysis = None
            alignment_score = 0.0
            ipa_str = ""
            tts_tokens_str = ""

        prosody_res.phonetic_alignment_score = alignment_score

        if (
            prompt_mode == "trap_phrase"
            and clean_prompt
            and (is_maya_prompt or lang_mode == "yua")
        ):
            raw_conf = float(asr_res.confidence or 0.0)
            acoustic_conf = min(1.0, max(0.0, alignment_score / 100.0))
            effective_conf = (
                round(max(raw_conf, 0.65 * acoustic_conf + 0.35 * raw_conf), 4)
                if has_valid_phonation
                else 0.0
            )
            is_validated = (
                bool(final_transcript)
                and has_valid_phonation
                and (alignment_score >= 35.0 or raw_conf >= 0.15)
            )
        else:
            effective_conf = asr_res.confidence
            is_validated = bool(final_transcript)

        return {
            "transcript": final_transcript,
            "raw_transcript": raw_asr_text,
            "asr_confidence": effective_conf,
            "is_validated": is_validated,
            "language_code": effective_lang,
            "ipa_phonemes": ipa_str,
            "tts_phonetic_tokens": tts_tokens_str,
            "grammatical_analysis": gram_analysis,
            "phonetic_alignment_score": alignment_score,
        }

    def recompute_segment_linguistics(self, segment: AudioSegment) -> None:
        """
        Recalcula `language_code`, `ipa_phonemes`, `tts_phonetic_tokens`,
        `grammatical_analysis` y `phonetic_alignment_score` tras una edición manual
        de `segment.transcript`.
        """
        clean_txt = (segment.transcript or "").strip()
        if not clean_txt:
            segment.ipa_phonemes = ""
            segment.tts_phonetic_tokens = ""
            segment.grammatical_analysis = None
            if segment.metrics is not None:
                segment.metrics.phonetic_alignment_score = 0.0
            return

        gram_res = self.maya_grammar.analyze(clean_txt)
        g2p_res = self.maya_g2p.phonemize_sentence(clean_txt)
        if gram_res.get("language_code") in ("yua", "yua_es"):
            segment.transcript = g2p_res.normalized_1984
        segment.language_code = gram_res.get("language_code", "es_yuc")
        segment.ipa_phonemes = g2p_res.ipa_transcription
        segment.tts_phonetic_tokens = g2p_res.tts_phonetic_tokens
        segment.grammatical_analysis = gram_res

        if segment.metrics is not None:
            m = segment.metrics
            synthetic_prosody = ProsodyAnalysisResult(
                normalized_wav_path=BASE_DIR / segment.file_path,
                duration_sec=float(segment.duration_sec or 0.0),
                sample_rate=int(segment.sample_rate or 24000),
                f0_mean_hz=float(m.f0_mean_hz or 0.0),
                f0_median_hz=float(m.f0_median_hz or 0.0),
                f0_min_hz=float(m.f0_min_hz or 0.0),
                f0_max_hz=float(m.f0_max_hz or 0.0),
                f0_semitones_std=float(m.f0_semitones_std or 0.0),
                f0_semitones_range=float(m.f0_semitones_range or 0.0),
                f0_curve_semitones=list(m.f0_curve_semitones or []),
                f1_mean_hz=float(m.f1_mean_hz or 0.0),
                f2_mean_hz=float(m.f2_mean_hz or 0.0),
                vowel_space_points=list(m.vowel_space_points or []),
                hnr_db=float(m.hnr_db or 0.0),
                snr_db=float(m.snr_db or 0.0),
                voiced_fraction=float(m.voiced_fraction or 0.0),
                glottal_stops_count=int(m.glottal_stops_count or 0),
                glottal_stop_timestamps=list(m.glottal_stop_timestamps or []),
            )
            m.phonetic_alignment_score = self.analyzer.evaluate_maya_phonetic_compliance(
                synthetic_prosody, g2p_res
            )

    def process_single_clip(
        self,
        db: Session,
        speaker_id: int,
        input_audio_path: Path | str,
        prompt_mode: str = "spontaneous",
        prompt_text: Optional[str] = None,
        language_mode: str = "auto",
        cleanup_input: bool = False,
    ) -> AudioSegment:
        """
        Procesa un clip grabado directamente desde el navegador (WebM/WAV/OGG/MP3):
        1. Normaliza a WAV Mono 24kHz 16-bit PCM en `data/wavs/`.
        2. Extrae métricas prosódicas, formantes, HNR/SNR y cortes glotales con `ProsodyAnalyzer`.
        3. Transcribe con `faster-whisper` (usando `DEFAULT_MAYA_1984_PROMPT` si es Maya).
        4. Ejecuta `MayaG2P1984`, `MayaGrammaticalAnalyzer` y validación acústica Praat.
        5. Persiste `AudioSegment` y `AcousticMetrics` en SQLite.
        """
        speaker = db.get(Speaker, speaker_id)
        if speaker is None:
            raise ValueError(f"No existe el hablante con id={speaker_id}.")

        lang_mode = self._normalize_language_mode(language_mode)
        clean_prompt = prompt_text.strip() if prompt_text else ""
        prompt_is_maya = bool(clean_prompt) and (
            self.maya_grammar.detect_language_code(clean_prompt) in ("yua", "yua_es")
        )
        use_maya_asr_prompt = (lang_mode == "yua") or prompt_is_maya

        in_path = Path(input_audio_path)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:6]
        safe_code = _sanitize_code(speaker.speaker_code)
        filename = f"{safe_code}_{stamp}_{uid}.wav"
        wav_path = WAVS_DIR / filename

        try:
            _, duration_sec = self.analyzer.normalize_to_wav_24k(in_path, wav_path)
            if duration_sec < 0.25:
                wav_path.unlink(missing_ok=True)
                raise ValueError(
                    f"El clip grabado es demasiado corto ({duration_sec:.2f}s). "
                    "Graba al menos 0.5 segundos de voz."
                )

            prosody_res = self.analyzer.analyze_wav(
                wav_path, speaker_gender=speaker.gender
            )
            asr_res = self.transcribe_wav(
                wav_path,
                has_voicing=(prosody_res.voiced_fraction >= 0.025),
                custom_initial_prompt=(
                    DEFAULT_MAYA_1984_PROMPT if use_maya_asr_prompt else None
                ),
            )

            ling_bundle = self._resolve_segment_linguistics(
                prosody_res=prosody_res,
                asr_res=asr_res,
                prompt_mode=prompt_mode,
                prompt_text=clean_prompt or None,
                language_mode=lang_mode,
            )

            rel_path = wav_path.relative_to(BASE_DIR).as_posix()
            segment = AudioSegment(
                speaker_id=speaker.id,
                filename=filename,
                file_path=rel_path,
                source_type="mic",
                prompt_mode=prompt_mode,
                prompt_text=clean_prompt or None,
                duration_sec=prosody_res.duration_sec,
                sample_rate=prosody_res.sample_rate,
                orig_start_sec=0.0,
                orig_end_sec=prosody_res.duration_sec,
                transcript=ling_bundle["transcript"],
                raw_transcript=ling_bundle["raw_transcript"],
                asr_confidence=ling_bundle["asr_confidence"],
                is_validated=ling_bundle["is_validated"],
                language_code=ling_bundle["language_code"],
                ipa_phonemes=ling_bundle["ipa_phonemes"],
                tts_phonetic_tokens=ling_bundle["tts_phonetic_tokens"],
                grammatical_analysis=ling_bundle["grammatical_analysis"],
            )
            db.add(segment)
            db.flush()  # Obtiene segment.id antes de insertar AcousticMetrics

            metrics = AcousticMetrics(
                segment_id=segment.id,
                **prosody_res.to_metrics_kwargs(),
            )
            db.add(metrics)
            db.commit()
            db.refresh(segment)
            # Asegurar relaciones cargadas en sesión antes de retornar
            _ = segment.speaker
            _ = segment.metrics
            return segment

        except Exception:
            db.rollback()
            wav_path.unlink(missing_ok=True)
            raise
        finally:
            if cleanup_input:
                in_path.unlink(missing_ok=True)

    def process_long_audio(
        self,
        db: Session,
        speaker_id: int,
        input_audio_path: Path | str,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 10.0,
        pad_ms: int = 180,
        language_mode: str = "auto",
        cleanup_input: bool = False,
    ) -> list[AudioSegment]:
        """
        Procesa un archivo de audio largo (entrevista, podcast, monólogo):
        1. Decodifica la señal maestra a 24 kHz Mono en memoria.
        2. Segmenta con Silero VAD (a 16 kHz internos) en fragmentos de 3.0 a 10.0 s
           con padding de 180 ms sin cortar palabras.
        3. Guarda cada recorte en `data/wavs/` (WAV 24kHz 16-bit PCM), ejecuta análisis
           acústico con `ProsodyAnalyzer`, transcribe con `faster-whisper` y evalúa G2P Maya 1984.
        4. Persiste todos los registros `AudioSegment` y `AcousticMetrics` en SQLite.
        """
        speaker = db.get(Speaker, speaker_id)
        if speaker is None:
            raise ValueError(f"No existe el hablante con id={speaker_id}.")

        lang_mode = self._normalize_language_mode(language_mode)
        in_path = Path(input_audio_path)
        created_wav_paths: list[Path] = []

        try:
            samples_24k = self.analyzer.decode_to_mono_24k(in_path)
            windows = self.detect_smart_segments(
                samples_24k=samples_24k,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                pad_ms=pad_ms,
            )

            if not windows:
                return []

            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            batch_uid = uuid.uuid4().hex[:4]
            safe_code = _sanitize_code(speaker.speaker_code)
            persisted_segments: list[AudioSegment] = []

            for idx, (start_sec, end_sec) in enumerate(windows, start=1):
                s_sample = max(0, int(round(start_sec * self.MASTER_SR)))
                e_sample = min(
                    samples_24k.shape[0], int(round(end_sec * self.MASTER_SR))
                )
                chunk = samples_24k[s_sample:e_sample]
                if chunk.size == 0:
                    continue

                chunk = self._apply_edge_fade(chunk, self.MASTER_SR, fade_ms=10.0)

                filename = f"{safe_code}_{stamp}_{batch_uid}_{idx:03d}.wav"
                wav_path = WAVS_DIR / filename
                self.analyzer.save_mono_24k_pcm16(chunk, wav_path)

                prosody_res = self.analyzer.analyze_wav(
                    wav_path, speaker_gender=speaker.gender
                )
                asr_res = self.transcribe_wav(
                    wav_path,
                    has_voicing=(prosody_res.voiced_fraction >= 0.025),
                    custom_initial_prompt=(
                        DEFAULT_MAYA_1984_PROMPT if lang_mode == "yua" else None
                    ),
                )

                # Descartar fragmentos puramente ruidosos sin habla inteligible ni fonación
                if not asr_res.text.strip() and prosody_res.voiced_fraction < 0.08:
                    wav_path.unlink(missing_ok=True)
                    continue

                ling_bundle = self._resolve_segment_linguistics(
                    prosody_res=prosody_res,
                    asr_res=asr_res,
                    prompt_mode="long_audio",
                    prompt_text=None,
                    language_mode=lang_mode,
                )

                created_wav_paths.append(wav_path)
                rel_path = wav_path.relative_to(BASE_DIR).as_posix()

                segment = AudioSegment(
                    speaker_id=speaker.id,
                    filename=filename,
                    file_path=rel_path,
                    source_type="upload_vad",
                    prompt_mode="long_audio",
                    prompt_text=None,
                    duration_sec=prosody_res.duration_sec,
                    sample_rate=prosody_res.sample_rate,
                    orig_start_sec=start_sec,
                    orig_end_sec=end_sec,
                    transcript=ling_bundle["transcript"],
                    raw_transcript=ling_bundle["raw_transcript"],
                    asr_confidence=ling_bundle["asr_confidence"],
                    is_validated=ling_bundle["is_validated"],
                    language_code=ling_bundle["language_code"],
                    ipa_phonemes=ling_bundle["ipa_phonemes"],
                    tts_phonetic_tokens=ling_bundle["tts_phonetic_tokens"],
                    grammatical_analysis=ling_bundle["grammatical_analysis"],
                )
                db.add(segment)
                db.flush()

                metrics = AcousticMetrics(
                    segment_id=segment.id,
                    **prosody_res.to_metrics_kwargs(),
                )
                db.add(metrics)
                persisted_segments.append(segment)

            db.commit()
            for seg in persisted_segments:
                db.refresh(seg)
                _ = seg.speaker
                _ = seg.metrics

            return persisted_segments

        except Exception:
            db.rollback()
            for orphan_wav in created_wav_paths:
                orphan_wav.unlink(missing_ok=True)
            raise
        finally:
            if cleanup_input:
                in_path.unlink(missing_ok=True)


# ==============================================================================
# Instancia Singleton global y funciones fachada para la capa FastAPI (Fase 3)
# ==============================================================================
_PIPELINE_INSTANCE: Optional[AudioIngestionPipeline] = None
_PIPELINE_INIT_LOCK = threading.Lock()


def get_pipeline() -> AudioIngestionPipeline:
    """Devuelve la instancia Singleton de `AudioIngestionPipeline`."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        with _PIPELINE_INIT_LOCK:
            if _PIPELINE_INSTANCE is None:
                _PIPELINE_INSTANCE = AudioIngestionPipeline()
    return _PIPELINE_INSTANCE


def process_single_clip(
    db: Session,
    speaker_id: int,
    input_audio_path: Path | str,
    prompt_mode: str = "spontaneous",
    prompt_text: Optional[str] = None,
    language_mode: str = "auto",
    cleanup_input: bool = False,
) -> AudioSegment:
    """Función fachada para procesar un clip individual grabado en la web."""
    return get_pipeline().process_single_clip(
        db=db,
        speaker_id=speaker_id,
        input_audio_path=input_audio_path,
        prompt_mode=prompt_mode,
        prompt_text=prompt_text,
        language_mode=language_mode,
        cleanup_input=cleanup_input,
    )


def process_long_audio(
    db: Session,
    speaker_id: int,
    input_audio_path: Path | str,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 10.0,
    pad_ms: int = 180,
    language_mode: str = "auto",
    cleanup_input: bool = False,
) -> list[AudioSegment]:
    """Función fachada para segmentar, analizar y transcribir un audio largo con Silero VAD."""
    return get_pipeline().process_long_audio(
        db=db,
        speaker_id=speaker_id,
        input_audio_path=input_audio_path,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        pad_ms=pad_ms,
        language_mode=language_mode,
        cleanup_input=cleanup_input,
    )
