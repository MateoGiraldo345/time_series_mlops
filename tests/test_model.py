"""Validación de la red neuronal. Ejecutar con: pytest tests/test_model.py -v"""
import pytest
import os
import joblib
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, "app", "mejor_modelo.keras")
SCALER_X_PATH = os.path.join(BASE_DIR, "app", "mejor_modelo_scaler_x.joblib")
SCALER_Y_PATH = os.path.join(BASE_DIR, "app", "mejor_modelo_scaler_y.joblib")

LAG = 60
HORIZONTE = 7

def test_artefactos_existen():
    """Verifica que los archivos generados en el notebook estén en la carpeta app/"""
    assert os.path.exists(MODEL_PATH), "Falta mejor_modelo.keras"
    assert os.path.exists(SCALER_X_PATH), "Falta el scaler de entrada"
    assert os.path.exists(SCALER_Y_PATH), "Falta el scaler de salida"

def test_arquitectura_modelo():
    """Verifica que el modelo acepte 60 inputs y bote 7 outputs."""
    modelo = tf.keras.models.load_model(MODEL_PATH)
    
    # Verificamos la capa de entrada
    input_shape = modelo.input_shape
    assert input_shape == (None, LAG), f"El modelo espera {input_shape}, no 60."
    
    # Verificamos la capa de salida
    output_shape = modelo.output_shape
    assert output_shape == (None, HORIZONTE), f"El modelo bota {output_shape}, no 7."

def test_inferencia_matematica():
    """Prueba que el pipeline de escalado -> modelo -> desescalado funcione."""
    modelo   = tf.keras.models.load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_Y = joblib.load(SCALER_Y_PATH)
    
    # Simulamos un vector de 60 minutos de volatilidad
    dummy_data = np.random.rand(1, LAG) * 0.001
    
    # 1. Escalar
    scaled_input = scaler_X.transform(dummy_data)
    assert not np.isnan(scaled_input).any()
    
    # 2. Predecir
    scaled_output = modelo.predict(scaled_input, verbose=0)
    assert scaled_output.shape == (1, HORIZONTE)
    
    # 3. Desescalar
    final_output = scaler_Y.inverse_transform(scaled_output)
    
    # Verificar que los números sean reales y tengan sentido lógico
    assert not np.isnan(final_output).any()
    assert final_output.shape == (1, HORIZONTE)