"""
API Backend Principal en FastAPI para AccentLab Web.

Provee:
- Inicialización automática de SQLite (SQLAlchemy 2.0) mediante `lifespan`.
- Montaje de archivos estáticos (`/static`), audios normalizados (`/media/wavs`)
  y datasets exportados (`/media/exports`), sirviendo la SPA en `GET /`.
- Catálogo fonético de "Frases Trampa" diseñadas para español yucateco y mexicano.
- CRUD de hablantes (`Speaker`), estadísticas globales/dialectales (`/api/stats`),
  grabación directa desde navegador (`/api/record-clip`), segmentación automática de
  audios largos con Silero VAD + Faster-Whisper (`/api/upload-long-audio`), edición
  manual de transcripciones, eliminación física + lógica de clips y exportación de
  datasets en formato Coqui / XTTS-v2 / LJSpeech (`metadata.csv` + `manifest.json` + `.zip`).
"""

from __future__ import annotations

import gc
import json
import logging
import re
import shutil
import time
import uuid
import zipfile
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.database import (
    BASE_DIR,
    EXPORTS_DIR,
    RAW_UPLOADS_DIR,
    WAVS_DIR,
    get_db,
    init_db,
)
from app.linguistics import MayaG2P1984, MayaGrammaticalAnalyzer
from app.models import AcousticMetrics, AudioSegment, Speaker
from app.pipeline.processor import get_pipeline, process_long_audio, process_single_clip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("accentlab.api")

_MAYA_G2P = MayaG2P1984()
_MAYA_GRAMMAR = MayaGrammaticalAnalyzer(g2p=_MAYA_G2P)

STATIC_DIR: Path = Path(__file__).resolve().parent / "static"
for _folder in (STATIC_DIR, RAW_UPLOADS_DIR, WAVS_DIR, EXPORTS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Catálogo Fonético de "Frases Trampa" (Maya Yucateco Norma 1984 + Español Yucateco)
# ==============================================================================
TRAP_PHRASES: list[dict[str, str]] = [
    # --- LENGUA MAYA YUCATECA (NORMA ORTOGRÁFICA DE AGOSTO DE 1984) ---
    {
        "id": "MAYA_SAL_01",
        "language": "yua",
        "category": "Maya 1984 — Saludos y Variantes Pragmáticas",
        "phenomenon": (
            "Saludo diurno estándar pan-peninsular: vocal rearticulada /a̰ʔa/ + implosiva /ɓ/ "
            "en 'Ma'alob' y eyectiva velar /kʼ/ + vocal larga baja /ìː/ en 'k'iin'."
        ),
        "text": "Ma'alob k'iin",
    },
    {
        "id": "MAYA_SAL_02",
        "language": "yua",
        "category": "Maya 1984 — Saludos y Variantes Pragmáticas",
        "phenomenon": (
            "Saludo matutino tradicional ('Buena mañana'): construcción relacional 'yutsil' "
            "seguida de vocal rearticulada /a̰ʔa/, africada /ts/ e implosiva final en 'ja'atskab'."
        ),
        "text": "Yutsil ja'atskab",
    },
    {
        "id": "MAYA_SAL_03",
        "language": "yua",
        "category": "Maya 1984 — Saludos y Variantes Pragmáticas",
        "phenomenon": (
            "Saludo augural de amanecer ('Que amanezca bien el día'): contrasta /a̰ʔa/ en "
            "'Ma'alob', tono alto largo /áː/ en 'sáastal' y tono bajo largo /ìː/ en 'k'iin'."
        ),
        "text": "Ma'alob u sáastal k'iin",
    },
    {
        "id": "MAYA_SAL_04",
        "language": "yua",
        "category": "Maya 1984 — Saludos y Variantes Pragmáticas",
        "phenomenon": (
            "Intercambio dialógico completo: fricativa /ʃ/ en 'Bix', vocal larga baja en 'beel', "
            "tono alto con prefijo epentético w- en 'wóol' y rearticulación /o̰ʔo/ en 'bo'otik'."
        ),
        "text": "¿Bix a beel? Toj in wóol, Dios bo'otik.",
    },
    {
        "id": "MAYA_REART_01",
        "language": "yua",
        "category": "Maya 1984 — Vocales Rearticuladas [V'V]",
        "phenomenon": (
            "Doble núcleo rearticulado laringalizado /a̰ʔa/ dentro de una misma palabra en "
            "'k'a'ana'an' ([kʼa̰ʔana̰ʔan]), junto con eyectiva alveolar /tʼ/ en 't'aan'."
        ),
        "text": "Jach k'a'ana'an in kanik maaya t'aan sáansamal.",
    },
    {
        "id": "MAYA_REART_02",
        "language": "yua",
        "category": "Maya 1984 — Vocales Rearticuladas [V'V]",
        "phenomenon": (
            "Secuencia de vocales rearticuladas posteriores /o̰ʔo/ en 'ts'o'ok' y /ṵʔu/ en "
            "'bu'ul', contrastadas con vocal larga alta /éː/ en 'yéetel'."
        ),
        "text": "Ts'o'ok in janal bu'ul yéetel waaj ichil in wotoch.",
    },
    {
        "id": "MAYA_REART_03",
        "language": "yua",
        "category": "Maya 1984 — Vocales Rearticuladas [V'V]",
        "phenomenon": (
            "Cuatro núcleos glotalizados en conversación fluida: 'ba'ax' (/a̰ʔa/), 'wa'alik' "
            "(/a̰ʔa/), 'ko'ox' (/o̰ʔo/) y 'te'elo'' (/ḛʔe/ + cierre glotal final /oʔ/)."
        ),
        "text": "¿Ba'ax ka wa'alik? Ko'ox xíimbal te'elo'.",
    },
    {
        "id": "MAYA_EJEC_01",
        "language": "yua",
        "category": "Maya 1984 — Consonantes Eyectivas [C']",
        "phenomenon": (
            "Alta densidad de consonantes glotalizadas eyectivas: doble /tʃʼ/ en 'ch'íich'o'', "
            "/kʼ/ en 'k'aay' y 'yóok'ol', y doble /kʼ/ con tono alto en 'k'áak'o''."
        ),
        "text": "Le ch'íich'o' táan u k'aay yóok'ol le k'áak'o'.",
    },
    {
        "id": "MAYA_EJEC_02",
        "language": "yua",
        "category": "Maya 1984 — Consonantes Eyectivas [C']",
        "phenomenon": (
            "Contraste entre tres puntos de articulación eyectivos: africada alveolar /tsʼ/ "
            "('ts'íibtik'), velar /kʼ/ + cierre glotal final ('k'aaba'') y dental /tʼ/ ('t'aan')."
        ),
        "text": "Taak in ts'íibtik u k'aaba' in kaajal ich maaya t'aan.",
    },
    {
        "id": "MAYA_EJEC_03",
        "language": "yua",
        "category": "Maya 1984 — Consonantes Eyectivas [C']",
        "phenomenon": (
            "Ataque bilabial eyectivo /pʼ/ seguido de cierre glotal posvocálico en 'P'o'' "
            "([pʼoʔ]), oposición eyectiva/implosiva en 'k'ab' ([kʼaɓ]) y sufijo plural '-o'ob'."
        ),
        "text": "P'o' a k'ab táanil ti' a janal yéetel a láak'o'ob.",
    },
    {
        "id": "MAYA_TON_01",
        "language": "yua",
        "category": "Maya 1984 — Contraste Tonal [V́V vs VV]",
        "phenomenon": (
            "Oposición acústica directa entre vocales largas de tono alto /áː/ ('Sáansamal', "
            "'sáastal') y vocales largas de tono bajo sostenido ('k'iin', 'kool', 'yuum')."
        ),
        "text": "Sáansamal ku sáastal ma'alob k'iin tu kool in yuum.",
    },
    {
        "id": "MAYA_TON_02",
        "language": "yua",
        "category": "Maya 1984 — Contraste Tonal [V́V vs VV]",
        "phenomenon": (
            "Alternancia simétrica de picos tonales altos ('áak'ab', 'táan', 'síis') frente "
            "a núcleos largos graves ('taal', 'iik'', 'kaajo'') con cierres glotales."
        ),
        "text": "Ma'alob áak'ab, táan u taal síis iik' te' kaajo'.",
    },
    {
        "id": "MAYA_TON_03",
        "language": "yua",
        "category": "Maya 1984 — Contraste Tonal [V́V vs VV]",
        "phenomenon": (
            "Contraste entre tonos bajos largos iniciales ('peek'', 'miis') y tonos altos "
            "largos centrales ('yéetel', 'táan', 'jáal') cerrando con clítico distal '-o''."
        ),
        "text": "In peek' yéetel in miis táan u wenelo'ob tu jáal le bejo'.",
    },
    {
        "id": "MAYA_TON_04",
        "language": "yua",
        "category": "Maya 1984 — Contraste Tonal [V́V vs VV]",
        "phenomenon": (
            "Contraste de cantidad y glotalización entre vocal rearticulada ('Tu'ux'), "
            "cierre tópico ('bejla'e''), vocales cortas ('ka bin') y larga baja ('kool')."
        ),
        "text": "¿Tu'ux ka bin bejla'e'? Yan in bin meyaj ich kool.",
    },
    # --- ESPAÑOL YUCATECO Y VARIANTES DE MÉXICO ---
    {
        "id": "YUC_GLOT_01",
        "language": "es_yuc",
        "category": "Cortes Glotales / Hiatos [ʔ]",
        "phenomenon": (
            "Fronteras vocálicas (/i#i/, /a#i/, /a#o/, /a#e/) que inducen saltillos "
            "intervocálicos [ʔ] en lugar de la sinalefa estándar."
        ),
        "text": "Mi hijo iba a ir a otra esquina a esperar a Elena.",
    },
    {
        "id": "YUC_GLOT_02",
        "language": "es_yuc",
        "category": "Cortes Glotales / Hiatos [ʔ]",
        "phenomenon": (
            "Ataques vocálicos tónicos tras vocal (/a#'o/, /o#a/, /a#a/) propensos a "
            "oclusión glotal inicial en español yucateco."
        ),
        "text": "La otra hora estuvo haciendo un aire horrible allá afuera.",
    },
    {
        "id": "YUC_GLOT_03",
        "language": "es_yuc",
        "category": "Cortes Glotales y Sustrato Maya",
        "phenomenon": (
            "Interjección regional ('¡fo!') con cierre glotal abrupto, aspiración en "
            "'hamaca' y fricativa postalveolar /ʃ/ en 'xux'."
        ),
        "text": "¡Fo, mira ese xux! Se metió hasta abajo de la hamaca de la chichí.",
    },
    {
        "id": "YUC_CANT_01",
        "language": "es_yuc",
        "category": "Entonación Circunfleja ('Cantadito')",
        "phenomenon": (
            "Pregunta pronominal seguida de disyuntiva con alargamiento nuclear y "
            "contorno tonal circunflejo ascendente-descendente."
        ),
        "text": "¿Y tú qué vas a hacer hoy en la tarde, vas a venir o no?",
    },
    {
        "id": "YUC_CANT_02",
        "language": "es_yuc",
        "category": "Entonación Circunfleja ('Cantadito')",
        "phenomenon": (
            "Exclamación enfática inicial ('¡Mare!') seguida de interrogativa total "
            "polar con tonema circunflejo regional."
        ),
        "text": "¡Mare, qué calor está haciendo hoy! ¿No quieres tomar pozole bien frío?",
    },
    {
        "id": "YUC_CANT_03",
        "language": "es_yuc",
        "category": "Entonación Circunfleja ('Cantadito')",
        "phenomenon": (
            "Vocativo agudo afectivo ('nené') con pico elevado de F0 y contraste "
            "prosódico entre interrogación y exhortación."
        ),
        "text": "¿Dónde lo dejaste, nené? ¡Te dije que lo guardaras temprano!",
    },
    {
        "id": "YUC_CANT_04",
        "language": "es_yuc",
        "category": "Alargamiento Vocálico y Foco Prosódico",
        "phenomenon": (
            "Alargamiento de vocal tónica nuclear en énfasis ('bien lejos') con "
            "excursión amplia en semitonos."
        ),
        "text": "Está bien lejos su casa, tienes que agarrar dos camiones para llegar.",
    },
    {
        "id": "YUC_NAS_01",
        "language": "es_yuc",
        "category": "Bilabialización Nasal Final [-m]",
        "phenomenon": (
            "Palabras oxítonas terminadas en /-n/ ('Don', 'Ramón', 'pan', 'jamón', "
            "'San', 'Juan') que favorecen cierre bilabial [-m]."
        ),
        "text": "Don Ramón compró mucho pan con jamón allá en San Juan.",
    },
    {
        "id": "YUC_NAS_02",
        "language": "es_yuc",
        "category": "Bilabialización Nasal Final [-m]",
        "phenomenon": (
            "Nasales finales en pausa y coda ('Yucatán', 'buen', 'con', 'carbón', "
            "'rincón') junto con oclusiva bilabial final en 'pib'."
        ),
        "text": "En Yucatán hacen un buen pib con carbón en el rincón del patio.",
    },
    {
        "id": "YUC_NAS_03",
        "language": "es_yuc",
        "category": "Nasal Final en Contorno Interrogativo",
        "phenomenon": (
            "Combinación de curva interrogativa regional con múltiples codas nasales "
            "agudas ('quién', 'portón', 'camión', 'tren')."
        ),
        "text": "¿Quién dejó abierto el portón cuando pasaron el camión y el tren?",
    },
    {
        "id": "YUC_MOD_01",
        "language": "es_yuc",
        "category": "Léxico y Sintaxis Regional",
        "phenomenon": (
            "Uso semántico regional de 'buscar' con valor de 'encontrar' y "
            "acentuación contrastiva en cláusula adversativa."
        ),
        "text": "Lo busqué y lo busqué por todo el cuarto, pero no lo busqué.",
    },
    {
        "id": "YUC_MOD_02",
        "language": "es_yuc",
        "category": "Modismos y Fricativa /ʃ/",
        "phenomenon": (
            "Realización de /ʃ/ en 'boxito' y 'xix', combinada con ataque exclamativo "
            "en '¡Puchis!' y timbre vocálico coloquial."
        ),
        "text": "¡Puchis, boxito, ya dejaste puro xix de pan encima de la mesa!",
    },
    {
        "id": "YUC_MOD_03",
        "language": "es_yuc",
        "category": "Calco Funcional Cotidiano",
        "phenomenon": (
            "Uso de 'prestar' por 'pedir prestado' y ritmo silábico fluido en "
            "habla espontánea dirigida."
        ),
        "text": "Préstame tu lápiz para apuntar su nombre, ahorita te lo devuelvo.",
    },
    {
        "id": "YUC_MOD_04",
        "language": "es_yuc",
        "category": "Sustrato Léxico y Choque Vocálico",
        "phenomenon": (
            "Africada palatal final en 'tuch', diptongo en 'huiro' y frontera "
            "vocálica /a#a/ en 'la albarrada' con potencial corte glotal."
        ),
        "text": "Ese huiro se lastimó su tuch cuando brincó la albarrada del solar.",
    },
    {
        "id": "MEX_GEN_01",
        "language": "es_yuc",
        "category": "Contraste Coloquial Pan-Mexicano",
        "phenomenon": (
            "Oración de referencia del español mexicano general para contrastar "
            "cadencia circunfleja del centro frente a la entonación peninsular yucateca."
        ),
        "text": "¡Híjole, mano, ahorita mismo nos lanzamos al mercado por unos tacos bien servidos!",
    },
]


# ==============================================================================
# Esquemas Pydantic v2 (Validación de Entrada y Contratos JSON)
# ==============================================================================
class SpeakerCreate(BaseModel):
    """Datos de entrada para registrar un nuevo hablante."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=120, description="Nombre o alias")
    gender: str = Field(..., pattern=r"^(M|F|Otro)$", description="Género: M, F u Otro")
    age: int = Field(..., ge=5, le=110, description="Edad en años")
    subregion: str = Field(
        default="Centro",
        min_length=2,
        max_length=64,
        description="Subregión dialectal (Oriente, Centro, Sur, Poniente, Costa, Otro)",
    )
    accent_intensity: str = Field(
        default="Medio",
        pattern=r"^(Suave|Medio|Marcado)$",
        description="Intensidad percibida del acento",
    )
    speaker_code: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Código opcional (si se omite se autogenera ej. YUC_ORI_001)",
    )


class TranscriptUpdate(BaseModel):
    """Payload para edición manual y validación de la transcripción de un segmento."""

    transcript: str = Field(..., min_length=0, max_length=2000)
    is_validated: bool = Field(default=True)


class MayaAnalyzeRequest(BaseModel):
    """Payload para el analizador fonológico G2P Maya 1984 y morfosintáctico en vivo."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Oración o frase en Maya Yucateco (1984 o variante colonial) o Español Yucateco",
    )


class DatasetExportRequest(BaseModel):
    """Filtros y formato fonológico para exportar el dataset (Coqui / XTTS / VITS / Piper / LJSpeech)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    dataset_name: str = Field(
        default="accentlab_yuc_dataset",
        min_length=2,
        max_length=80,
    )
    speaker_id: Optional[int] = Field(default=None)
    subregion: Optional[str] = Field(default=None)
    language_code: Optional[str] = Field(
        default=None,
        description="Filtrar opcionalmente por código de lengua: yua, es_yuc, yua_es",
    )
    validated_only: bool = Field(default=True)
    min_snr_db: Optional[float] = Field(default=None, ge=0.0, le=60.0)
    text_format: str = Field(
        default="orthographic_1984",
        pattern=r"^(orthographic_1984|ipa_phonemes|tts_tokens|dual_orth_ipa)$",
        description=(
            "Formato del campo de texto en metadata.csv: orthographic_1984 | "
            "ipa_phonemes | tts_tokens | dual_orth_ipa"
        ),
    )


# ==============================================================================
# Funciones Auxiliares Internas
# ==============================================================================
_SUBREGION_PREFIX_MAP: dict[str, str] = {
    "oriente": "ORI",
    "centro": "CEN",
    "sur": "SUR",
    "poniente": "PON",
    "costa": "COS",
    "norte": "NOR",
    "otro": "OTR",
}


def _generate_unique_speaker_code(db: Session, subregion: str) -> str:
    """Genera un código único legible para el hablante, p. ej. `YUC_ORI_001`."""
    norm_sub = subregion.strip().lower()
    sub_code = _SUBREGION_PREFIX_MAP.get(norm_sub)
    if not sub_code:
        ascii_letters = re.sub(r"[^A-Za-z]", "", subregion).upper()
        sub_code = (ascii_letters[:3] or "REG").ljust(3, "X")

    prefix = f"YUC_{sub_code}_"
    existing_codes = set(
        db.scalars(
            select(Speaker.speaker_code).where(Speaker.speaker_code.like(f"{prefix}%"))
        ).all()
    )

    seq = len(existing_codes) + 1
    while True:
        candidate = f"{prefix}{seq:03d}"
        if candidate not in existing_codes:
            # Verificación global adicional por seguridad
            exists_any = db.scalar(
                select(Speaker.id).where(Speaker.speaker_code == candidate)
            )
            if exists_any is None:
                return candidate
        seq += 1


def _save_upload_temporarily(upload: UploadFile, prefix: str = "raw") -> Path:
    """Guarda un `UploadFile` entrante en `RAW_UPLOADS_DIR` preservando su extensión."""
    orig_name = upload.filename or ""
    ext = Path(orig_name).suffix.lower()
    if not ext or len(ext) > 8:
        content_type = (upload.content_type or "").lower()
        if "ogg" in content_type:
            ext = ".ogg"
        elif "wav" in content_type:
            ext = ".wav"
        elif "mp4" in content_type or "m4a" in content_type:
            ext = ".m4a"
        elif "mpeg" in content_type or "mp3" in content_type:
            ext = ".mp3"
        else:
            ext = ".webm"

    temp_name = f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = RAW_UPLOADS_DIR / temp_name
    try:
        upload.file.seek(0)
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
    finally:
        try:
            upload.file.close()
        except Exception:
            pass

    if not dest_path.exists() or dest_path.stat().st_size == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo de audio recibido está vacío (0 bytes).",
        )
    return dest_path


def _resolve_segment_wav_path(segment: AudioSegment) -> Path:
    """Localiza la ruta física del archivo `.wav` asociado a un `AudioSegment`."""
    direct_in_wavs = WAVS_DIR / segment.filename
    if direct_in_wavs.exists():
        return direct_in_wavs
    rel_candidate = BASE_DIR / segment.file_path
    return rel_candidate


def _safe_unlink_file(path: Path, retries: int = 5, delay_sec: float = 0.06) -> bool:
    """
    Elimina un archivo en disco de forma resiliente en Windows, reintentando brevemente
    si un socket HTTP de `<audio>` o el indexador del SO mantiene el descriptor abierto.
    """
    if not path.exists():
        return False
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            if attempt < retries - 1:
                gc.collect()
                time.sleep(delay_sec)
            else:
                logger.warning(
                    "No se pudo borrar inmediatamente '%s' por bloqueo temporal en Windows.",
                    path,
                )
        except OSError as exc:
            logger.warning("Error de E/S al eliminar '%s': %s", path, exc)
            break
    return not path.exists()


# ==============================================================================
# Ciclo de Vida (Lifespan), Aplicación FastAPI y Montaje de Estáticos/Media
# ==============================================================================
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Inicializa tablas en SQLite y directorios de trabajo al arrancar el servidor."""
    logger.info("Inicializando base de datos SQLite en AccentLab Web...")
    init_db()
    yield
    logger.info("Apagando servidor AccentLab Web.")


app = FastAPI(
    title="AccentLab Web API — Edición Maya Yucateco 1984 & Español Regional",
    description=(
        "Sistema Full-Stack v2.0 para recolección, segmentación Silero VAD, transcripción "
        "Faster-Whisper, motor fonológico G2P Maya Yucateco (Norma Ortográfica de Agosto de 1984), "
        "glosador morfosintáctico Leipzig y análisis acústico-prosódico Praat para datasets TTS/VC."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montaje de directorios multimedia y recursos estáticos del Frontend
app.mount("/media/wavs", StaticFiles(directory=str(WAVS_DIR)), name="media_wavs")
app.mount("/media/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="media_exports")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================================
# Ruta Raíz (Entrega del Dashboard SPA)
# ==============================================================================
@app.get("/", include_in_schema=False)
async def serve_spa_index() -> Any:
    """Sirve el Dashboard Web SPA (`app/static/index.html`)."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return HTMLResponse(
        content=(
            "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
            "<title>AccentLab Web API</title></head>"
            "<body style='background:#0f172a;color:#e2e8f0;font-family:sans-serif;padding:2rem;'>"
            "<h2>AccentLab Web v2.0 Backend Operativo</h2>"
            "<p>La API está activa. Documentación interactiva en <a style='color:#38bdf8' href='/docs'>/docs</a>.</p>"
            "</body></html>"
        ),
        status_code=200,
    )


# ==============================================================================
# 1. Catálogo de Frases Trampa Fonéticas y Laboratorio Lingüístico Maya 1984
# ==============================================================================
@app.get("/api/prompts", tags=["Guion Fonético"])
def get_phonetic_prompts(
    category: Optional[str] = Query(default=None, description="Filtrar por categoría fonética"),
    language: Optional[str] = Query(default=None, description="Filtrar por lengua: yua | es_yuc"),
) -> dict[str, Any]:
    """Devuelve el catálogo de 'Frases Trampa' (Maya Yucateco Norma 1984 y Español Regional)."""
    items = TRAP_PHRASES
    if language and language.strip().lower() not in ("", "auto"):
        lang_lower = language.strip().lower()
        items = [p for p in items if p.get("language", "es_yuc") == lang_lower]
    if category:
        cat_lower = category.strip().lower()
        items = [p for p in items if cat_lower in p["category"].lower()]

    categories = list(dict.fromkeys(p["category"] for p in TRAP_PHRASES))
    return {
        "count": len(items),
        "categories": categories,
        "prompts": items,
    }


@app.post("/api/linguistics/analyze-maya", tags=["Lingüística Maya 1984"])
def analyze_maya_linguistics(payload: MayaAnalyzeRequest) -> dict[str, Any]:
    """
    Analiza al instante cualquier oración en Maya Yucateco (Norma 1984 o grafía pre-1984)
    o Español Yucateco, devolviendo:
    - Normalización ortográfica estricta de Agosto de 1984.
    - Transcripción fonética AFI (IPA).
    - Secuencia de tokens fonéticos desambiguados para entrenadores TTS (`XTTS-v2`/`VITS`/`Piper`).
    - Desglose contextual de los 4 tipos de apóstrofo (`C'`, `V'V`, `V'`, `'V`).
    - Glosa morfosintáctica interlineal (estilo Leipzig) palabra por palabra y traducción semántica.
    """
    analysis = _MAYA_GRAMMAR.analyze(payload.text)
    g2p_dict = analysis["g2p"]
    profile = g2p_dict["expected_acoustic_profile"]

    return {
        "status": "ok",
        "original_text": analysis["original_text"],
        "normalized_1984": g2p_dict["normalized_1984"],
        "language_code": analysis["language_code"],
        "language_label": analysis["language_label"],
        "ipa_transcription": g2p_dict["ipa_transcription"],
        "tts_phonetic_tokens": g2p_dict["tts_phonetic_tokens"],
        "Spanish_Approx_Readout": g2p_dict["Spanish_Approx_Readout"],
        "apostrophe_breakdown": profile["apostrophe_breakdown"],
        "expected_acoustic_profile": profile,
        "words_g2p": g2p_dict["words"],
        "syntactic_structure": analysis["syntactic_structure"],
        "semantic_translation_es": analysis["semantic_translation_es"],
        "literal_gloss_es": analysis["literal_gloss_es"],
        "interlinear_segmentation": analysis["interlinear_segmentation"],
        "interlinear_leipzig_gloss": analysis["interlinear_leipzig_gloss"],
        "pragmatic_notes": analysis["pragmatic_notes"],
        "equivalent_variants": analysis["equivalent_variants"],
        "discontinuous_frames": analysis["discontinuous_frames"],
        "morphemes": analysis["morphemes"],
        "g2p": g2p_dict,
    }


# ==============================================================================
# 2. Gestión de Hablantes y Estadísticas Acústicas del Corpus
# ==============================================================================
@app.get("/api/speakers", tags=["Hablantes"])
def list_speakers(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Lista todos los hablantes registrados junto con el conteo y duración de sus clips."""
    speakers = db.scalars(select(Speaker).order_by(Speaker.created_at.desc())).all()

    agg_rows = db.execute(
        select(
            AudioSegment.speaker_id,
            func.count(AudioSegment.id),
            func.coalesce(func.sum(AudioSegment.duration_sec), 0.0),
        ).group_by(AudioSegment.speaker_id)
    ).all()
    stats_by_spk: dict[int, tuple[int, float]] = {
        int(row[0]): (int(row[1]), float(row[2])) for row in agg_rows
    }

    result_list: list[dict[str, Any]] = []
    for spk in speakers:
        seg_count, total_dur = stats_by_spk.get(spk.id, (0, 0.0))
        item = spk.to_dict()
        item["segment_count"] = seg_count
        item["total_duration_sec"] = round(total_dur, 2)
        result_list.append(item)

    return {"count": len(result_list), "speakers": result_list}


@app.post("/api/speakers", status_code=status.HTTP_201_CREATED, tags=["Hablantes"])
def create_speaker(
    payload: SpeakerCreate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Registra un nuevo hablante generando automáticamente su `speaker_code` si no se envía."""
    if payload.speaker_code:
        clean_code = re.sub(r"[^A-Za-z0-9_-]+", "_", payload.speaker_code.strip()).upper()
        existing = db.scalar(select(Speaker).where(Speaker.speaker_code == clean_code))
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"El código de hablante '{clean_code}' ya está registrado.",
            )
    else:
        clean_code = _generate_unique_speaker_code(db, payload.subregion)

    speaker = Speaker(
        speaker_code=clean_code,
        name=payload.name,
        gender=payload.gender,
        age=payload.age,
        subregion=payload.subregion,
        accent_intensity=payload.accent_intensity,
    )
    db.add(speaker)
    db.commit()
    db.refresh(speaker)

    spk_dict = speaker.to_dict()
    spk_dict["segment_count"] = 0
    spk_dict["total_duration_sec"] = 0.0
    return {"status": "ok", "speaker": spk_dict}


@app.delete("/api/speakers/{speaker_id}", tags=["Hablantes"])
def delete_speaker(speaker_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Elimina un hablante, sus segmentos asociados en SQLite y todos sus `.wav` físicos."""
    speaker = db.scalar(
        select(Speaker)
        .options(joinedload(Speaker.segments))
        .where(Speaker.id == speaker_id)
    )
    if speaker is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró el hablante con id={speaker_id}.",
        )

    wav_files_to_remove = [_resolve_segment_wav_path(seg) for seg in speaker.segments]
    removed_clips = len(wav_files_to_remove)
    code = speaker.speaker_code

    db.delete(speaker)
    db.commit()

    for wav_path in wav_files_to_remove:
        _safe_unlink_file(wav_path)

    return {
        "status": "deleted",
        "speaker_id": speaker_id,
        "speaker_code": code,
        "deleted_segments": removed_clips,
    }


@app.get("/api/stats", tags=["Estadísticas"])
def get_corpus_stats(db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Calcula métricas globales y desglosadas por subregión para alimentar los indicadores
    del Dashboard (total de hablantes, horas/minutos limpios, variación de 'cantadito',
    cortes glotales y formantes medios).
    """
    total_speakers = int(db.scalar(select(func.count(Speaker.id))) or 0)
    speakers_by_sub: dict[str, int] = defaultdict(int)
    for sub_val, count_val in db.execute(
        select(Speaker.subregion, func.count(Speaker.id)).group_by(Speaker.subregion)
    ).all():
        speakers_by_sub[str(sub_val)] = int(count_val)

    segments = (
        db.scalars(
            select(AudioSegment).options(
                joinedload(AudioSegment.speaker),
                joinedload(AudioSegment.metrics),
            )
        )
        .unique()
        .all()
    )

    total_segments = len(segments)
    validated_segments = sum(1 for s in segments if s.is_validated)
    maya_segments = sum(
        1 for s in segments if (s.language_code or "es_yuc") in ("yua", "yua_es")
    )
    total_duration_sec = sum(float(s.duration_sec or 0.0) for s in segments)

    f0_vals: list[float] = []
    st_std_vals: list[float] = []
    glottal_vals: list[int] = []
    f1_vals: list[float] = []
    f2_vals: list[float] = []
    hnr_vals: list[float] = []
    snr_vals: list[float] = []
    align_vals: list[float] = []

    sub_buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "clips": 0,
            "duration_sec": 0.0,
            "st_stds": [],
            "glottals": [],
            "f0s": [],
            "f1s": [],
            "f2s": [],
        }
    )

    for seg in segments:
        sub_name = seg.speaker.subregion if seg.speaker else "Desconocida"
        bucket = sub_buckets[sub_name]
        bucket["clips"] += 1
        bucket["duration_sec"] += float(seg.duration_sec or 0.0)

        m = seg.metrics
        if m is not None:
            glottal_vals.append(int(m.glottal_stops_count or 0))
            bucket["glottals"].append(int(m.glottal_stops_count or 0))

            if m.f0_mean_hz > 0:
                f0_vals.append(float(m.f0_mean_hz))
                bucket["f0s"].append(float(m.f0_mean_hz))
            if m.f0_semitones_std > 0:
                st_std_vals.append(float(m.f0_semitones_std))
                bucket["st_stds"].append(float(m.f0_semitones_std))
            if m.f1_mean_hz > 0:
                f1_vals.append(float(m.f1_mean_hz))
                bucket["f1s"].append(float(m.f1_mean_hz))
            if m.f2_mean_hz > 0:
                f2_vals.append(float(m.f2_mean_hz))
                bucket["f2s"].append(float(m.f2_mean_hz))
            if m.hnr_db != 0:
                hnr_vals.append(float(m.hnr_db))
            if m.snr_db > 0:
                snr_vals.append(float(m.snr_db))
            if (m.phonetic_alignment_score or 0.0) > 0:
                align_vals.append(float(m.phonetic_alignment_score))

    def _avg(items: list[float] | list[int], ndigits: int = 2) -> float:
        return round(sum(items) / len(items), ndigits) if items else 0.0

    all_subregions = sorted(set(speakers_by_sub.keys()) | set(sub_buckets.keys()))
    by_subregion: list[dict[str, Any]] = []
    for sub in all_subregions:
        b = sub_buckets[sub]
        by_subregion.append(
            {
                "subregion": sub,
                "speakers": speakers_by_sub.get(sub, 0),
                "segments": b["clips"],
                "duration_sec": round(b["duration_sec"], 2),
                "duration_min": round(b["duration_sec"] / 60.0, 2),
                "avg_f0_hz": _avg(b["f0s"], 2),
                "avg_semitones_std": _avg(b["st_stds"], 3),
                "avg_glottal_stops": _avg(b["glottals"], 2),
                "avg_f1_hz": _avg(b["f1s"], 1),
                "avg_f2_hz": _avg(b["f2s"], 1),
            }
        )

    return {
        "total_speakers": total_speakers,
        "total_segments": total_segments,
        "validated_segments": validated_segments,
        "maya_segments": maya_segments,
        "spanish_segments": total_segments - maya_segments,
        "total_duration_sec": round(total_duration_sec, 2),
        "total_duration_min": round(total_duration_sec / 60.0, 2),
        "total_duration_hours": round(total_duration_sec / 3600.0, 4),
        "avg_f0_hz": _avg(f0_vals, 2),
        "avg_semitones_std": _avg(st_std_vals, 3),
        "avg_glottal_stops": _avg(glottal_vals, 2),
        "total_glottal_stops": int(sum(glottal_vals)),
        "avg_f1_hz": _avg(f1_vals, 1),
        "avg_f2_hz": _avg(f2_vals, 1),
        "avg_hnr_db": _avg(hnr_vals, 2),
        "avg_snr_db": _avg(snr_vals, 2),
        "avg_phonetic_alignment_score": _avg(align_vals, 2),
        "by_subregion": by_subregion,
    }


# ==============================================================================
# 3. Ingesta de Audio (Grabación Directa en Navegador y Carga de Audio Largo VAD)
# ==============================================================================
@app.post("/api/record-clip", status_code=status.HTTP_201_CREATED, tags=["Ingesta y DSP"])
def record_direct_clip(
    speaker_id: int = Form(..., description="ID del hablante en SQLite"),
    prompt_mode: str = Form(default="spontaneous", description="trap_phrase | spontaneous"),
    prompt_text: Optional[str] = Form(default=None, description="Texto de frase trampa si aplica"),
    language_mode: str = Form(
        default="auto", description="Modo de idioma de sesión: auto | yua | es_yuc"
    ),
    audio: UploadFile = File(..., description="Blob de audio capturado por MediaRecorder"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Recibe un clip directo desde el micrófono web, lo normaliza a WAV 24kHz 16-bit Mono,
    extrae la curva F0 en semitonos, formantes, HNR/SNR y cortes glotales, transcribe con
    `faster-whisper` (protegiendo la Norma Maya 1984), ejecuta G2P + Glosa y persiste en SQLite.
    """
    raw_path = _save_upload_temporarily(audio, prefix="mic")
    try:
        segment = process_single_clip(
            db=db,
            speaker_id=speaker_id,
            input_audio_path=raw_path,
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
            language_mode=language_mode,
            cleanup_input=True,
        )
        return {
            "status": "ok",
            "segment": segment.to_dict(include_speaker=True, include_metrics=True),
        }
    except ValueError as val_exc:
        _safe_unlink_file(raw_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_exc),
        ) from val_exc
    except Exception as exc:
        _safe_unlink_file(raw_path)
        logger.exception("Error procesando clip directo: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la grabación: {exc}",
        ) from exc


@app.post("/api/upload-long-audio", status_code=status.HTTP_201_CREATED, tags=["Ingesta y DSP"])
def upload_and_segment_long_audio(
    speaker_id: int = Form(..., description="ID del hablante asociado al audio largo"),
    min_duration_sec: float = Form(default=3.0, ge=1.0, le=15.0),
    max_duration_sec: float = Form(default=10.0, ge=3.0, le=30.0),
    pad_ms: int = Form(default=180, ge=0, le=500),
    language_mode: str = Form(
        default="auto", description="Modo de idioma de sesión: auto | yua | es_yuc"
    ),
    audio: UploadFile = File(..., description="Archivo de entrevista, podcast o monólogo"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Recibe un archivo de audio largo (WAV, MP3, OGG, M4A, WebM), lo recorta inteligentemente
    con Silero VAD en clips de 3.0 a 10.0 segundos sin cortar palabras, analiza cada clip con
    Praat, transcribe con `faster-whisper` y ejecuta G2P Maya 1984 / glosa gramatical.
    """
    if min_duration_sec >= max_duration_sec:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`min_duration_sec` debe ser estrictamente menor que `max_duration_sec`.",
        )

    raw_path = _save_upload_temporarily(audio, prefix="long")
    try:
        segments = process_long_audio(
            db=db,
            speaker_id=speaker_id,
            input_audio_path=raw_path,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            pad_ms=pad_ms,
            language_mode=language_mode,
            cleanup_input=True,
        )
        serialized = [
            seg.to_dict(include_speaker=True, include_metrics=True) for seg in segments
        ]
        total_dur = round(sum(s.duration_sec for s in segments), 2)
        message = (
            f"Se generaron y analizaron {len(segments)} clips ({total_dur}s en total)."
            if segments
            else (
                f"No se detectaron segmentos de voz limpia entre "
                f"{min_duration_sec:.1f}s y {max_duration_sec:.1f}s en el archivo enviado."
            )
        )
        return {
            "status": "ok",
            "message": message,
            "count": len(serialized),
            "total_duration_sec": total_dur,
            "segments": serialized,
        }
    except ValueError as val_exc:
        _safe_unlink_file(raw_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_exc),
        ) from val_exc
    except Exception as exc:
        _safe_unlink_file(raw_path)
        logger.exception("Error segmentando audio largo: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al segmentar y analizar el audio largo: {exc}",
        ) from exc


# ==============================================================================
# 4. Explorador, Consulta, Edición de Transcripción y Borrado de Segmentos
# ==============================================================================
@app.get("/api/segments", tags=["Dataset y Segmentos"])
def list_segments(
    speaker_id: Optional[int] = Query(default=None),
    subregion: Optional[str] = Query(default=None),
    language_code: Optional[str] = Query(default=None),
    source_type: Optional[str] = Query(default=None),
    validated_only: bool = Query(default=False),
    search: Optional[str] = Query(default=None, description="Búsqueda en transcripción, AFI o hablante"),
    limit: int = Query(default=300, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Lista los segmentos procesados en SQLite con soporte de filtrado multicriterio."""
    stmt = (
        select(AudioSegment)
        .join(Speaker, AudioSegment.speaker_id == Speaker.id)
        .options(
            joinedload(AudioSegment.speaker),
            joinedload(AudioSegment.metrics),
        )
    )

    if speaker_id is not None:
        stmt = stmt.where(AudioSegment.speaker_id == speaker_id)
    if subregion:
        stmt = stmt.where(func.lower(Speaker.subregion) == subregion.strip().lower())
    if language_code and language_code.strip():
        stmt = stmt.where(AudioSegment.language_code == language_code.strip().lower())
    if source_type:
        stmt = stmt.where(AudioSegment.source_type == source_type.strip())
    if validated_only:
        stmt = stmt.where(AudioSegment.is_validated.is_(True))
    if search and search.strip():
        pattern = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                AudioSegment.transcript.ilike(pattern),
                AudioSegment.ipa_phonemes.ilike(pattern),
                AudioSegment.filename.ilike(pattern),
                Speaker.name.ilike(pattern),
                Speaker.speaker_code.ilike(pattern),
            )
        )

    stmt = stmt.order_by(AudioSegment.created_at.desc()).offset(offset).limit(limit)
    rows = db.scalars(stmt).unique().all()

    return {
        "count": len(rows),
        "segments": [
            seg.to_dict(include_speaker=True, include_metrics=True) for seg in rows
        ],
    }


@app.get("/api/segments/{segment_id}", tags=["Dataset y Segmentos"])
def get_single_segment(
    segment_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Obtiene el detalle completo y métricas acústicas de un segmento específico."""
    segment = db.scalar(
        select(AudioSegment)
        .options(
            joinedload(AudioSegment.speaker),
            joinedload(AudioSegment.metrics),
        )
        .where(AudioSegment.id == segment_id)
    )
    if segment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe el segmento con id={segment_id}.",
        )
    return {"segment": segment.to_dict(include_speaker=True, include_metrics=True)}


@app.put("/api/segments/{segment_id}/transcript", tags=["Dataset y Segmentos"])
def update_segment_transcript(
    segment_id: int,
    payload: TranscriptUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Actualiza manualmente la transcripción (Maya 1984 o modismos regionales),
    recalcula automáticamente G2P + Glosa + Alineación Acústica y guarda en SQLite.
    """
    segment = db.scalar(
        select(AudioSegment)
        .options(
            joinedload(AudioSegment.speaker),
            joinedload(AudioSegment.metrics),
        )
        .where(AudioSegment.id == segment_id)
    )
    if segment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe el segmento con id={segment_id}.",
        )

    segment.transcript = payload.transcript.strip()
    segment.is_validated = payload.is_validated
    get_pipeline().recompute_segment_linguistics(segment)
    db.commit()
    db.refresh(segment)

    return {
        "status": "ok",
        "segment": segment.to_dict(include_speaker=True, include_metrics=True),
    }


@app.delete("/api/segments/{segment_id}", tags=["Dataset y Segmentos"])
def delete_segment(
    segment_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Elimina un segmento defectuoso tanto de SQLite como su archivo `.wav` físico en disco."""
    segment = db.get(AudioSegment, segment_id)
    if segment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe el segmento con id={segment_id}.",
        )

    wav_path = _resolve_segment_wav_path(segment)
    filename = segment.filename

    db.delete(segment)
    db.commit()

    wav_deleted = _safe_unlink_file(wav_path)

    return {
        "status": "deleted",
        "id": segment_id,
        "filename": filename,
        "wav_deleted_from_disk": wav_deleted,
    }


# ==============================================================================
# 5. Exportación de Dataset (Formato Coqui / XTTS-v2 / VITS / Piper / LJSpeech + ZIP)
# ==============================================================================
@app.post("/api/export-dataset", tags=["Exportación TTS/VC"])
def export_tts_dataset(
    payload: Optional[DatasetExportRequest] = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Genera un paquete de entrenamiento listo para Coqui TTS / XTTS-v2 / VITS / Piper / LJSpeech:
    - Copia los `.wav` (24kHz 16-bit Mono) filtrados a `data/exports/{nombre}/wavs/`.
    - Escribe `metadata.csv` en el formato seleccionado (`text_format`):
      * `"orthographic_1984"`: `wavs/clip.wav|Ma'alob k'iin|SPK`
      * `"ipa_phonemes"`     : `wavs/clip.wav|ma̰ʔaloɓ kʼìːn|SPK`
      * `"tts_tokens"`       : `wavs/clip.wav|m a_reart l o b_imp | k_ej ii_low n|SPK`
      * `"dual_orth_ipa"`    : `wavs/clip.wav|Ma'alob k'iin|ma̰ʔaloɓ kʼìːn|SPK`
    - Escribe `manifest.json` con metadatos sociolingüísticos, fonológicos y métricas acústicas.
    - Comprime todo el dataset en un archivo `.zip` descargable vía `/media/exports/...`.
    """
    req = payload or DatasetExportRequest()

    stmt = (
        select(AudioSegment)
        .join(Speaker, AudioSegment.speaker_id == Speaker.id)
        .outerjoin(AcousticMetrics, AcousticMetrics.segment_id == AudioSegment.id)
        .options(
            joinedload(AudioSegment.speaker),
            joinedload(AudioSegment.metrics),
        )
    )

    if req.speaker_id is not None:
        stmt = stmt.where(AudioSegment.speaker_id == req.speaker_id)
    if req.subregion:
        stmt = stmt.where(func.lower(Speaker.subregion) == req.subregion.strip().lower())
    if req.language_code:
        stmt = stmt.where(AudioSegment.language_code == req.language_code.strip().lower())
    if req.validated_only:
        stmt = stmt.where(AudioSegment.is_validated.is_(True))
    if req.min_snr_db is not None:
        stmt = stmt.where(AcousticMetrics.snr_db >= req.min_snr_db)

    stmt = stmt.order_by(Speaker.speaker_code.asc(), AudioSegment.created_at.asc())
    candidates = db.scalars(stmt).unique().all()

    # Filtrar clips cuyo archivo físico exista y posean transcripción no vacía si son validados
    valid_records: list[tuple[AudioSegment, Path]] = []
    for seg in candidates:
        clean_txt = (seg.transcript or "").strip()
        if req.validated_only and not clean_txt:
            continue
        src_wav = _resolve_segment_wav_path(seg)
        if src_wav.is_file():
            valid_records.append((seg, src_wav))

    if not valid_records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No existen clips que cumplan los criterios de exportación "
                "(verifica que haya clips con transcripción validada y archivos .wav en disco)."
            ),
        )

    safe_base = re.sub(r"[^A-Za-z0-9_-]+", "_", req.dataset_name.strip()).strip("_")
    if not safe_base:
        safe_base = "accentlab_dataset"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_folder_name = f"{safe_base}_{stamp}"

    export_dir = EXPORTS_DIR / export_folder_name
    export_wavs_dir = export_dir / "wavs"
    export_wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_lines: list[str] = []
    manifest_entries: list[dict[str, Any]] = []
    exported_speakers: set[str] = set()
    total_duration_sec = 0.0

    format_headers = {
        "orthographic_1984": "wavs/filename.wav|orthographic_1984|speaker_code",
        "ipa_phonemes": "wavs/filename.wav|ipa_phonemes|speaker_code",
        "tts_tokens": "wavs/filename.wav|tts_phonetic_tokens|speaker_code",
        "dual_orth_ipa": "wavs/filename.wav|orthographic_1984|ipa_phonemes|speaker_code",
    }

    for seg, src_wav in valid_records:
        dst_wav = export_wavs_dir / seg.filename
        shutil.copy2(src_wav, dst_wav)

        raw_clean_txt = re.sub(
            r"\s+",
            " ",
            (seg.transcript or "").replace("|", " "),
        ).strip()

        # Garantizar representaciones G2P incluso para clips preexistentes de v1.0
        g2p_fallback = _MAYA_G2P.phonemize_sentence(raw_clean_txt)
        lang_code = seg.language_code or _MAYA_GRAMMAR.detect_language_code(raw_clean_txt)

        if lang_code in ("yua", "yua_es"):
            orth_1984 = re.sub(
                r"\s+",
                " ",
                g2p_fallback.normalized_1984.replace("|", " "),
            ).strip()
        else:
            orth_1984 = raw_clean_txt

        ipa_raw = (seg.ipa_phonemes or g2p_fallback.ipa_transcription or "").strip()
        ipa_clean = re.sub(r"\s+", " ", ipa_raw.strip("[]").replace("|", " ")).strip()

        tts_tokens_str = re.sub(
            r"\s+",
            " ",
            (seg.tts_phonetic_tokens or g2p_fallback.tts_phonetic_tokens or "").strip(),
        )

        speaker_code = seg.speaker.speaker_code if seg.speaker else f"SPK_{seg.speaker_id}"
        exported_speakers.add(speaker_code)
        total_duration_sec += float(seg.duration_sec or 0.0)

        rel_wav_ref = f"wavs/{seg.filename}"
        if req.text_format == "ipa_phonemes":
            primary_text = ipa_clean
            metadata_lines.append(f"{rel_wav_ref}|{ipa_clean}|{speaker_code}")
        elif req.text_format == "tts_tokens":
            primary_text = tts_tokens_str
            metadata_lines.append(f"{rel_wav_ref}|{tts_tokens_str}|{speaker_code}")
        elif req.text_format == "dual_orth_ipa":
            primary_text = orth_1984
            metadata_lines.append(f"{rel_wav_ref}|{orth_1984}|{ipa_clean}|{speaker_code}")
        else:
            primary_text = orth_1984
            metadata_lines.append(f"{rel_wav_ref}|{orth_1984}|{speaker_code}")

        manifest_entries.append(
            {
                "audio_filepath": rel_wav_ref,
                "language_code": lang_code,
                "text": primary_text,
                "text_orthographic_1984": orth_1984,
                "ipa_phonemes": ipa_clean,
                "tts_phonetic_tokens": tts_tokens_str,
                "raw_whisper_text": seg.raw_transcript,
                "duration_sec": round(seg.duration_sec, 3),
                "sample_rate": seg.sample_rate,
                "is_validated": seg.is_validated,
                "grammatical_analysis": seg.grammatical_analysis,
                "speaker": seg.speaker.to_dict() if seg.speaker else None,
                "metrics": seg.metrics.to_dict() if seg.metrics else None,
            }
        )

    metadata_csv_path = export_dir / "metadata.csv"
    metadata_csv_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8", newline="\n")

    manifest_payload = {
        "dataset_name": export_folder_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio_format": "WAV Mono 24000Hz 16-bit Signed PCM",
        "text_format": req.text_format,
        "metadata_format": format_headers.get(
            req.text_format, "wavs/filename.wav|transcript|speaker_code"
        ),
        "filters": {
            "speaker_id": req.speaker_id,
            "subregion": req.subregion,
            "language_code": req.language_code,
            "validated_only": req.validated_only,
            "min_snr_db": req.min_snr_db,
        },
        "summary": {
            "total_clips": len(manifest_entries),
            "total_speakers": len(exported_speakers),
            "total_duration_sec": round(total_duration_sec, 2),
            "total_duration_min": round(total_duration_sec / 60.0, 2),
        },
        "items": manifest_entries,
    }
    manifest_json_path = export_dir / "manifest.json"
    manifest_json_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Generar archivo comprimido .zip para descarga directa desde el navegador
    zip_filename = f"{export_folder_name}.zip"
    zip_path = EXPORTS_DIR / zip_filename
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_item in sorted(export_dir.rglob("*")):
            if file_item.is_file():
                arcname = Path(export_folder_name) / file_item.relative_to(export_dir)
                zf.write(file_item, arcname=arcname.as_posix())

    return {
        "status": "ok",
        "export_name": export_folder_name,
        "export_dir": export_dir.as_posix(),
        "text_format": req.text_format,
        "zip_filename": zip_filename,
        "download_url": f"/media/exports/{zip_filename}",
        "metadata_csv_url": f"/media/exports/{export_folder_name}/metadata.csv",
        "manifest_json_url": f"/media/exports/{export_folder_name}/manifest.json",
        "exported_clips": len(manifest_entries),
        "exported_speakers": len(exported_speakers),
        "total_duration_sec": round(total_duration_sec, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
