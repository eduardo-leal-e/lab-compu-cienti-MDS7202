from agroalerta.reporte import contar_riesgos
from agroalerta.sensores import (
    SensorTemperatura,
    SensorViento,
)


def test_temperatura_bajo_cero_es_riesgosa():
    sensor = SensorTemperatura(0, 40)

    assert sensor.es_riesgo(-2)


def test_temperatura_templada_no_es_riesgosa():
    sensor = SensorTemperatura(0, 40)

    assert not sensor.es_riesgo(18)


def test_viento_normal_no_es_riesgoso():
    sensor = SensorViento(25)

    assert not sensor.es_riesgo(10)


def test_contar_riesgos():
    sensores = [
        SensorTemperatura(0, 40),
        SensorViento(25),
    ]

    lecturas = {
        "temperatura": [18, -2, 42],
        "viento": [10, 30],
    }

    conteo = contar_riesgos(sensores, lecturas)

    assert conteo == {
        "temperatura": 2,
        "viento": 1,
    }
