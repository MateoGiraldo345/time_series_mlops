# BTC Volatility Forecast — Proyecto Integrador Deep Learning

Predicción multistep de la volatilidad del precio de Bitcoin usando MLP,
validación cruzada temporal sin data leakage, diagnóstico de residuos con BDS,
y despliegue como API REST con FastAPI.

---

## Estructura del proyecto

```
time-series-mlops/
├── data/
│   └── btc_1m_2021_2026.xls          # Serie temporal BTC
├── notebooks/
│   ├── figs/                          # Gráficas generadas
│   ├── 1_eda.ipynb                    # Exploración y análisis
│   ├── 2_model_training.ipynb         # Entrenamiento MLP + CV + métricas
│   └── 3_residual_analysis.ipynb      # Diagnóstico de residuos (BDS)
├── results/                           # Tablas CSV con métricas
├── app/
│   ├── api.py                         # API FastAPI
│   ├── schemas.py                     # Modelos Pydantic
│   └── model.joblib                   # Modelo entrenado
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
git clone https://github.com/tu-usuario/time-series-mlops.git
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
  -d '{"lags": [0.00045, 0.00048, ...]}'
```

### Salida esperada

```json
{
  "prediction": [0.00046, 0.00047, 0.00045, 0.00048, 0.00046, 0.00047, 0.00045],
  "lag_usado": 15,
  "horizonte": 7,
  "descripcion": "Volatilidad predicha para los próximos 7 minutos..."
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
| Arquitectura | MLP (128 → 64 → 32 → 7) |
| Activación | ReLU + Dropout 0.2 |
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
