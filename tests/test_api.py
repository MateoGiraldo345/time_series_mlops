"""Pruebas unitarias de la API. Ejecutar con: pytest tests/test_api.py -v"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)
LAG = 60

def test_health_check():
    """Verifica que el servidor levante y el modelo esté en memoria."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["lag_esperado"] == LAG
    assert data["modelo_cargado"] is True

def test_predict_endpoint():
    """Prueba una predicción exitosa con datos de volatilidad simulados."""
    # Generamos 60 valores aleatorios pequeños simulando volatilidad (ej. 0.0005)
    lags_input = list(np.random.rand(LAG) * 0.001)
    response   = client.post("/predict", json={"lags": lags_input})
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == 7

def test_predict_lags_incorrectos():
    """Prueba el escudo de Pydantic enviando 3 valores en vez de 60."""
    response = client.post("/predict", json={"lags": [0.0001, 0.0002, 0.0003]})
    assert response.status_code == 422 # Unprocessable Entity

def test_predict_lista_vacia():
    """Prueba que no se caiga si mandan una lista vacía."""
    response = client.post("/predict", json={"lags": []})
    assert response.status_code == 422

def test_predict_valores_finitos():
    """Verifica que la API no devuelva NaNs ni Infinitos."""
    lags_input = list(np.random.rand(LAG) * 0.001)
    response   = client.post("/predict", json={"lags": lags_input})
    
    if response.status_code == 200:
        preds = response.json()["prediction"]
        assert all(np.isfinite(p) for p in preds)