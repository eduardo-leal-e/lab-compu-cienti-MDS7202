"""Ejecuta las pruebas de Lab4 solo cuando está disponible el CSV del curso."""

from __future__ import annotations

from pathlib import Path
from subprocess import run
from sys import executable

RUTA_CSV = Path("data/cru_country_tmp_tidy.csv")


def main() -> int:
    """Ejecuta pytest o explica por qué no se puede ejecutar."""
    if not RUTA_CSV.exists():
        print("CSV del curso no disponible; se omiten las pruebas de Lab4.")
        return 0

    return run([executable, "-m", "pytest"], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
