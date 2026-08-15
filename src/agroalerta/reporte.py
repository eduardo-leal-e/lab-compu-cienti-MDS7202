def contar_riesgos(sensores, lecturas):
    conteo = {}

    for sensor in sensores:
        conteo[sensor.nombre] = 0

        for valor in lecturas.get(sensor.nombre, []):
            if sensor.es_riesgo(valor):
                conteo[sensor.nombre] += 1

    return conteo
