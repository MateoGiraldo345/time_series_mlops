"""
API de predicción de volatilidad BTC.
Recibe los últimos `lag` valores de volatilidad histórica y devuelve 7 predicciones de volatilidad.
"""

from fastapi import FastAPI, HTTPException
from app.schemas import InputData, PredictionResponse
import joblib
import numpy as np
import os
import tensorflow as tf  # <-- IMPORTANTE para cargar el modelo

app = FastAPI(
    title="BTC Volatility Forecast API",
    description="Predice la volatilidad futura del BTC en un horizonte de 7 pasos.",
    version="1.0.0",
)

# ── Cargar modelo y scalers al iniciar ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROC_PATH = os.getenv(
    "PREPROC_PATH", os.path.join(BASE_DIR, "preprocesamiento.joblib")
)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "mejor_modelo.keras"))

try:
    # 1. Cargar metadata y scalers
    bundle = joblib.load(PREPROC_PATH)
    scaler_X = bundle["scaler_X"]
    scaler_Y = bundle["scaler_Y"]
    LAG = bundle["lag"]
    HORIZONTE = bundle["horizonte"]

    # 2. Cargar el modelo de Keras
    modelo = tf.keras.models.load_model(MODEL_PATH)

    print(f"✅ Modelo cargado exitosamente: lag={LAG}, horizonte={HORIZONTE}")
except Exception as e:
    modelo = scaler_X = scaler_Y = None
    LAG = HORIZONTE = None
    print(f"ADVERTENCIA: Error al cargar los artefactos. Error: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "modelo_cargado": modelo is not None,
        "lag_esperado": LAG,
        "horizonte": HORIZONTE,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Corre primero 2_model_training.ipynb.",
        )
    if len(data.lags) != LAG:
        raise HTTPException(
            status_code=422,
            detail=f"Se esperan {LAG} valores en 'lags', se recibieron {len(data.lags)}.",
        )

    # Preparar datos y predecir
    features = np.array(data.lags).reshape(1, -1)
    X_scaled = scaler_X.transform(features)

    # verbose=0 evita que TensorFlow imprima barras de carga en la consola de la API
    y_scaled = modelo.predict(X_scaled, verbose=0)

    prediction = scaler_Y.inverse_transform(y_scaled)[0].tolist()

    return PredictionResponse(
        prediction=[round(v, 8) for v in prediction],
        lag_usado=LAG,
        horizonte=HORIZONTE,
        descripcion=(
            f"Volatilidad predicha para los próximos {HORIZONTE} minutos "
            f"usando los últimos {LAG} valores de volatilidad histórica."
        ),
    )
