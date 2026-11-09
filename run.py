"""
Punto de entrada rápido para iniciar AccentLab Web con Uvicorn.
Uso:
    python run.py
"""

from __future__ import annotations

import os
import uvicorn


def main() -> None:
    host = os.getenv("ACCENTLAB_HOST", "127.0.0.1")
    port = int(os.getenv("ACCENTLAB_PORT", "8000"))
    reload_enabled = os.getenv("ACCENTLAB_RELOAD", "1").strip() in ("1", "true", "yes")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
