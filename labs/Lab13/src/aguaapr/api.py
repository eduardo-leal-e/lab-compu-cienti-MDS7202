"""Rutas de inferencia: información del modelo, vida y predicción."""

from fastapi import APIRouter, Depends

from src.aguaapr.artifacts import a_tabla, predecir
from src.aguaapr.constants import RANGOS
from src.aguaapr.contracts import RespuestaPrediccion, Servicios
from src.aguaapr.deps import crear_dependencia_clave
from src.aguaapr.validation import MedicionAgua


def crear_router_inferencia(servicios: Servicios) -> APIRouter:
    """Construye el router con tres rutas.

    - `GET /`: describe el modelo. Responde las métricas de validación de la
      metadata, la entrada (cada columna con su rango) y la
      salida.
    - `GET /health`: ruta de vida, pública. Responde
      `{"estado": "ok"}`.
    - `POST /predict`: recibe una `MedicionAgua` en el cuerpo y responde una
      `RespuestaPrediccion` con la predicción y su probabilidad. Exige la
      clave de API con `crear_dependencia_clave`.
    """
    router = APIRouter()

    @router.get("/")
    def informacion():
        return {
            "metricas_validacion": servicios.metadata["metricas_validacion"],
            "entrada": RANGOS,
            "salida": {"prediccion": "1 si es potable, 0 si no es potable"},
        }

    @router.get("/health")
    def health():
        return {"estado": "ok"}

    @router.post(
        "/predict",
        response_model=RespuestaPrediccion,
        dependencies=[Depends(crear_dependencia_clave(servicios))],
    )
    def predict(medicion: MedicionAgua):
        clases, probabilidades = predecir(servicios.modelo, a_tabla([medicion]))
        return RespuestaPrediccion(
            prediccion=int(clases[0]),
            probabilidad=round(float(probabilidades[0]), 4),
        )

    return router
