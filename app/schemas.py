"""Modelos Pydantic para la API."""
from pydantic import BaseModel, field_validator


class InputData(BaseModel):
    """Cuerpo de la petición: lista de valores de volatilidad histórica."""
    lags: list[float]

    @field_validator("lags")
    @classmethod
    def validar_lags(cls, v):
        if len(v) == 0:
            raise ValueError("La lista 'lags' no puede estar vacía.")
        return v


class PredictionResponse(BaseModel):
    """Respuesta de la API."""
    prediction: list[float]
    lag_usado: int
    horizonte: int
    descripcion: str
