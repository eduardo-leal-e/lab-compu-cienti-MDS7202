import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


from agroalerta.datos import cargar_lecturas
from agroalerta.reporte import contar_riesgos
from agroalerta.sensores import (
    SensorHumedad,
    SensorTemperatura,
    SensorViento,
)


def main():
    parser = argparse.ArgumentParser(description="AgroAlerta")
    parser.add_argument("--fecha", default="2026-06-15")
    args = parser.parse_args()

    sensores = [
        SensorTemperatura(0, 40),
        SensorViento(25),
        SensorHumedad(85),
    ]

    lecturas = cargar_lecturas(Path("data/lecturas.csv"), args.fecha)

    conteo = contar_riesgos(sensores, lecturas)

    print(f"Estación Parcela Norte — {args.fecha}")

    for sensor in sensores:
        print(
            f"{sensor.nombre.capitalize()}: {conteo[sensor.nombre]} lecturas en riesgo"
        )

    total = sum(conteo.values())
    print(f"\nTotal: {total} situaciones de riesgo")


if __name__ == "__main__":
    main()
