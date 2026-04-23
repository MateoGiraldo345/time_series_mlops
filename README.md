# BTC Volatility Forecast — Proyecto Integrador Deep Learning

Este proyecto busca predecir la variable temporal de interés: volatilidad del bitcoin utilizando exclusivamente su histórico temporal. Para ello, se emplea la librería timeseries-cv, junto con estrategias de validación temporal, una búsqueda exhausta de hiperparámetros, una red neuronal tipo MLP y un análisis riguroso de residuos.
---

## Estructura del proyecto

```
time-series-mlops/
├── data/
│   └── btc_1m_2021_2026.xls          # Serie temporal BTC
├── notebooks/
│   ├── 1_eda.ipynb                    # Exploración y análisis
│   ├── 2_model_training.ipynb         # Entrenamiento MLP + CV + métricas
│   └── 3_residual_analysis.ipynb      # Diagnóstico de residuos (BDS)
├── app/
│   ├── api.py                         # API FastAPI
│   ├── schemas.py                     # Modelos Pydantic
│   ├── preprocesamiento.joblib        # Preprocesamiento del input
│   └── mejor_modelo.keras             # Modelo entrenado
├── tests/
│   ├── test_model.py                  # Tests del modelo
│   └── test_api.py                    # Tests de la API
├── .github/workflows/
│   └── ci.yml                         # CI/CD GitHub Actions
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
git clone https://github.com/MateoGiraldo345/time_series_mlops.git
cd time-series-mlops
pip install -r requirements.txt
```

---

## Orden de ejecución

```
1. notebooks/1_eda.ipynb              # EDA
2. notebooks/2_model_training.ipynb   # Entrenamiento → genera app/model.joblib
3. notebooks/3_residual_analysis.ipynb # Diagnóstico de residuos
```

---

## API

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Documentación interactiva: http://localhost:8000/docs

### Ejemplo curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"lags": [0.00051202, 0.00047919, 0.00047937, 0.00048238, 0.00048578, 0.0004863, 0.00048431, 0.00047884, 0.00049063, 0.00051498, 0.00050464, 0.00048664, 0.00048114, 0.000469, 0.0005117, 0.0005312, 0.00053499, 0.00053697, 0.00052974, 0.00054057, 0.00055262, 0.0005623, 0.00057475, 0.00058323, 0.00058529, 0.00057929, 0.00060833, 0.00060755, 0.00060583, 0.0006051, 0.00053978, 0.00054612, 0.00054604, 0.00056647, 0.00056053, 0.00056447, 0.00057126, 0.00056221, 0.00061953, 0.00062107, 0.00063231, 0.00063238, 0.00063361, 0.00064025, 0.00061182, 0.00059227, 0.00060903, 0.00063561, 0.00063718, 0.0006211, 0.00060026, 0.00059621, 0.00057202, 0.00060984, 0.00062174, 0.00061985, 0.00060872, 0.00060892, 0.00061482, 0.00060058]}'
```

### Salida esperada

```json
{
  "prediction": [
    0.00063506,
    0.0006351,
    0.00064427,
    0.0006424,
    0.00064905,
    0.00063267,
    0.00063411
  ],
  "lag_usado": 60,
  "horizonte": 7,
  "descripcion": "Volatilidad predicha para los próximos 7 minutos usando los últimos 60 valores de volatilidad histórica."
}
```

---

## Docker

```bash
docker build -t btc-volatility-api:latest .
docker run -p 8000:8000 btc-volatility-api:latest
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Descripción del modelo

| Componente | Detalle |
|---|---|
| Arquitectura | MLP (256, 256) |
| Activación | ReLU + Dropout 0.3 |
| Optimizador | Adam |
| Variable objetivo | Volatilidad rolling (std retornos log, ventana=30) |
| Features | Lags del precio de cierre (15, 30, 60, 90 min) |
| Horizonte | 7 pasos futuros (multi-output) |
| Validación | split_train_val_test_groupKFold (tsxv) |
| Diagnóstico | Test BDS sobre residuos h=1 |

---

## Métricas

Para cada tamaño de lag y fold: RMSE, MAE, MAPE, MSE por horizonte h=1…7 y BDS p-value.

- **BDS p > 0.05** → residuos independientes ✓
- **BDS p ≤ 0.05** → estructura residual sin capturar ✗
