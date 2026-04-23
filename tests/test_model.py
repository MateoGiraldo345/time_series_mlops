"""Validación de la red neuronal. Ejecutar con: pytest tests/test_model.py -v"""
import os
import joblib
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "app", "mejor_modelo.keras")
PREPROC_PATH = os.path.join(BASE_DIR, "app", "preprocesamiento.joblib")


def test_artefactos_existen():
    """Verifica que los archivos necesarios existan en app/"""
    assert os.path.exists(MODEL_PATH), "Falta mejor_modelo.keras"
    assert os.path.exists(PREPROC_PATH), "Falta preprocesamiento.joblib"


def test_arquitectura_modelo():
    """Verifica que el modelo tenga dimensiones coherentes con el bundle."""
    modelo = tf.keras.models.load_model(MODEL_PATH)
    bundle = joblib.load(PREPROC_PATH)

    LAG = bundle["lag"]
    HORIZONTE = bundle["horizonte"]

    # Verificar entrada
    assert modelo.input_shape == (None, LAG), (
        f"El modelo espera {modelo.input_shape}, no {(None, LAG)}"
    )

    # Verificar salida
    assert modelo.output_shape == (None, HORIZONTE), (
        f"El modelo produce {modelo.output_shape}, no {(None, HORIZONTE)}"
    )


def test_inferencia_matematica():
    """Prueba pipeline completo: escalar -> modelo -> desescalar."""
    modelo = tf.keras.models.load_model(MODEL_PATH)
    bundle = joblib.load(PREPROC_PATH)

    scaler_X = bundle["scaler_X"]
    scaler_Y = bundle["scaler_Y"]
    LAG = bundle["lag"]
    HORIZONTE = bundle["horizonte"]

    # Datos dummy
    dummy_data = np.random.rand(1, LAG) * 0.001

    # 1. Escalar
    scaled_input = scaler_X.transform(dummy_data)
    assert not np.isnan(scaled_input).any()

    # 2. Predecir
    scaled_output = modelo.predict(scaled_input, verbose=0)
    assert scaled_output.shape == (1, HORIZONTE)

    # 3. Desescalar
    final_output = scaler_Y.inverse_transform(scaled_output)

    # Validaciones
    assert final_output.shape == (1, HORIZONTE)
    assert not np.isnan(final_output).any()
    assert np.all(np.isfinite(final_output))
