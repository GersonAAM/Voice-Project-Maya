"""
Configuración de Base de Datos SQLite local con SQLAlchemy 2.0 y gestión de rutas de datos.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ==============================================================================
# Rutas canónicas del proyecto y almacenamiento de audio
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_UPLOADS_DIR: Path = DATA_DIR / "raw_uploads"
WAVS_DIR: Path = DATA_DIR / "wavs"
EXPORTS_DIR: Path = DATA_DIR / "exports"

for _directory in (DATA_DIR, RAW_UPLOADS_DIR, WAVS_DIR, EXPORTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

DB_PATH: Path = DATA_DIR / "accentlab.db"
DATABASE_URL: str = os.getenv(
    "ACCENTLAB_DATABASE_URL",
    f"sqlite:///{DB_PATH.as_posix()}",
)

# ==============================================================================
# Motor SQLAlchemy 2.0 y optimizaciones de concurrencia/integridad para SQLite
# ==============================================================================
connect_args: dict[str, Any] = (
    {"check_same_thread": False, "timeout": 30}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True,
)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
    """Activa claves foráneas (CASCADE) y modo WAL en cada conexión SQLite."""
    if DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Clase base declarativa de SQLAlchemy 2.0 para todos los modelos ORM."""


def get_db() -> Generator[Session, None, None]:
    """Generador de sesión de base de datos para inyección de dependencias en FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_safe_migrations() -> None:
    """
    Ejecuta migraciones incrementales idempotentes (`ALTER TABLE ... ADD COLUMN`)
    sobre `accentlab.db` para actualizar bases de datos v1.0 hacia v2.0 (Maya 1984)
    sin perder hablantes, grabaciones ni métricas acústicas previas.
    """
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    audio_segment_migrations: list[tuple[str, str]] = [
        ("language_code", "VARCHAR(16) NOT NULL DEFAULT 'es_yuc'"),
        ("ipa_phonemes", "TEXT"),
        ("tts_phonetic_tokens", "TEXT"),
        ("grammatical_analysis", "JSON"),
    ]
    acoustic_metrics_migrations: list[tuple[str, str]] = [
        ("phonetic_alignment_score", "FLOAT NOT NULL DEFAULT 0.0"),
    ]

    with engine.begin() as conn:
        if "audio_segments" in table_names:
            existing_seg_cols = {
                col["name"] for col in inspector.get_columns("audio_segments")
            }
            for col_name, col_ddl in audio_segment_migrations:
                if col_name not in existing_seg_cols:
                    conn.execute(
                        text(
                            f"ALTER TABLE audio_segments ADD COLUMN {col_name} {col_ddl}"
                        )
                    )

        if "acoustic_metrics" in table_names:
            existing_met_cols = {
                col["name"] for col in inspector.get_columns("acoustic_metrics")
            }
            for col_name, col_ddl in acoustic_metrics_migrations:
                if col_name not in existing_met_cols:
                    conn.execute(
                        text(
                            f"ALTER TABLE acoustic_metrics ADD COLUMN {col_name} {col_ddl}"
                        )
                    )


def init_db() -> None:
    """Registra los modelos ORM, crea tablas faltantes y aplica migraciones seguras v2.0."""
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _run_safe_migrations()
