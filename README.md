# Telco Churn MLOps Pipeline

A complete MLOps pipeline for Telco Customer Churn Prediction with experiment tracking, model registry, serving, and drift monitoring.

## Architecture

```mermaid
graph TD
    A[Raw Data] --> B[Data Prep & Drift Injection]
    B --> C[Reference Dataset 70%]
    B --> D[Current Dataset 30% with Drift]
    C --> E[Training Pipeline]
    E --> F[MLflow Experiment Tracking]
    F --> G[Model Registry]
    G --> H[Staging]
    H --> I[Production]
    I --> J[FastAPI Model Serving]
    C --> K[Drift Monitoring]
    D --> K
    K --> L[Evidently Reports]
    L --> F
    J --> M[/predict API]
    J --> N[/predict/batch API]
```

## Quickstart

### Prerequisites
- Python 3.10+
- `uv` package manager (`pip install uv`)

### Setup
```bash
# Clone and enter project
cd nemo

# Install dependencies with uv
uv sync

# Verify installation
uv run python -c "import mlflow; import evidently; import fastapi; print('OK')"
```

### 1. Prepare Data (with Drift Injection)
```bash
uv run python -m src.data.split
```
This creates:
- `data/drift_injected/reference.csv` — 70% training reference data
- `data/drift_injected/current_drifted.csv` — 30% current data with synthetic drift:
  - MonthlyCharges shifted +$10
  - Contract_Month-to-month oversampled 2x
  - 5% label flip (Churn)

### 2. Train Models & Register Best
```bash
uv run train
```
This runs the training pipeline:
- Trains 4 models: RandomForest_Shallow, RandomForest_Deep, LogisticRegression_L2, XGBoost
- Logs params, metrics (accuracy, precision, recall, F1, ROC-AUC), and artifacts (model, confusion matrix, ROC curve) to MLflow
- Registers best model (by F1) to MLflow Model Registry as `telco-churn-model`
- Transitions: `Staging` → `Production`

### 3. Serve Model via FastAPI
```bash
# Terminal 1: Start server
uv run serve

# Terminal 2: Test predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender_Male": 1,
    "SeniorCitizen": 0,
    "Partner_Yes": 1,
    "Dependents_Yes": 0,
    "PhoneService_Yes": 1,
    "MultipleLines_No_phone_service": 0,
    "MultipleLines_Yes": 1,
    "InternetService_Fiber_optic": 1,
    "InternetService_No": 0,
    "OnlineSecurity_No_internet_service": 0,
    "OnlineSecurity_Yes": 1,
    "OnlineBackup_No_internet_service": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No_internet_service": 0,
    "DeviceProtection_Yes": 1,
    "TechSupport_No_internet_service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No_internet_service": 0,
    "StreamingTV_Yes": 1,
    "StreamingMovies_No_internet_service": 0,
    "StreamingMovies_Yes": 1,
    "Contract_One_year": 0,
    "Contract_Two_year": 0,
    "PaperlessBilling_Yes": 1,
    "PaymentMethod_Credit_card_automatic": 0,
    "PaymentMethod_Electronic_check": 1,
    "PaymentMethod_Mailed_check": 0,
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2
  }'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"customers": [...]}'

# Health check
curl http://localhost:8000/health

### 4. Run Drift Monitoring
```bash
uv run drift-check
```
This:
- Loads reference and current (drifted) datasets
- Generates Evidently data drift report (dataset + column-level)
- Generates target drift report
- Calculates 3 custom metrics:
  1. Mean MonthlyCharges difference
  2. Churn rate shift for Month-to-month contracts
  3. Mean tenure difference
- Saves HTML reports to `reports/` and logs to MLflow as artifacts
- Logs custom metrics to MLflow

### 5. View MLflow UI
```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```
Open http://localhost:5000 to view:
- **Experiments** → `telco-churn-experiment`: Compare 4 model runs (metrics, params, artifacts)
- **Models** → `telco-churn-model`: Registry with version stages (Staging, Production)
- **Artifacts**: Confusion matrices, ROC curves, drift HTML reports

---

## MLflow Experiment Tracking

### Tracked Metrics (per model run)
| Metric | Description |
|--------|-------------|
| `accuracy` | Overall accuracy |
| `precision` | Precision (positive class) |
| `recall` | Recall (positive class) |
| `f1` | F1 score (primary selection metric) |
| `roc_auc` | Area under ROC curve |

### Tracked Artifacts (per model run)
- `model/` — Serialized sklearn/XGBoost model with signature
- `confusion_matrix.png` — Confusion matrix heatmap
- `roc_curve.png` — ROC curve plot
- `feature_importance.png` — Feature importance (tree models)

### Model Registry Stages
```
Version 1 → Staging → Production (archives previous)
```

---

## Model Serving API

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + model version |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch predictions |

### Request Schema (Single)
```json
{
  "gender_Male": 1,
  "SeniorCitizen": 0,
  "Partner_Yes": 1,
  "Dependents_Yes": 0,
  "PhoneService_Yes": 1,
  "MultipleLines_No_phone_service": 0,
  "MultipleLines_Yes": 1,
  "InternetService_Fiber_optic": 1,
  "InternetService_No": 0,
  "OnlineSecurity_No_internet_service": 0,
  "OnlineSecurity_Yes": 1,
  "OnlineBackup_No_internet_service": 0,
  "OnlineBackup_Yes": 0,
  "DeviceProtection_No_internet_service": 0,
  "DeviceProtection_Yes": 1,
  "TechSupport_No_internet_service": 0,
  "TechSupport_Yes": 0,
  "StreamingTV_No_internet_service": 0,
  "StreamingTV_Yes": 1,
  "StreamingMovies_No_internet_service": 0,
  "StreamingMovies_Yes": 1,
  "Contract_One_year": 0,
  "Contract_Two_year": 0,
  "PaperlessBilling_Yes": 1,
  "PaymentMethod_Credit_card_automatic": 0,
  "PaymentMethod_Electronic_check": 1,
  "PaymentMethod_Mailed_check": 0,
  "tenure": 12,
  "MonthlyCharges": 70.35,
  "TotalCharges": 844.2
}
```

### Response Schema
```json
{
  "churn_probability": 0.73,
  "churn_prediction": 1,
  "model_version": "1"
}
```

---

## Drift Monitoring

### Injected Drift (for demonstration)
| Feature | Drift Type | Magnitude |
|---------|------------|-----------|
| MonthlyCharges | Distribution shift | +$10 mean increase |
| Contract_Month-to-month | Covariate shift | 2x oversample |
| Churn (target) | Label drift | 5% random flip |

### Evidently Reports
- **Data Drift Report**: Dataset drift + column-level drift scores (PSI, KS test)
- **Target Drift Report**: Churn distribution shift

### Custom Metrics (logged to MLflow)
| Metric | Description | Threshold |
|--------|-------------|-----------|
| `drift_monthly_charges_mean_diff` | |mean(cur) - mean(ref)| | > 5.0 |
| `drift_churn_rate_contract_shift` | |churn_rate(cur_mtm) - churn_rate(ref_mtm)| | > 0.05 |
| `drift_tenure_mean_diff` | |mean(cur) - mean(ref)| | > 2.0 |

### HTML Reports in MLflow
Drift check runs create a nested MLflow run named `drift_monitoring` with artifacts:
- `drift_reports/data_drift_report.html`
- `drift_reports/target_drift_report.html`

---

## Project Structure

```
nemo/
├── pyproject.toml          # uv dependencies & project config
├── uv.lock                 # Locked dependencies
├── README.md               # This file
├── src/
│   ├── config.py           # Paths, thresholds, MLflow config
│   ├── data/
│   │   ├── load.py         # Load processed train/test splits
│   │   └── split.py        # Raw → processed + drift injection
│   ├── train/
│   │   ├── train.py        # Training pipeline + MLflow logging
│   │   ├── evaluate.py     # Metrics + plotting
│   │   └── models.py       # Model definitions & hyperparams
│   ├── serve/
│   │   ├── app.py          # FastAPI app + model loading
│   │   └── schemas.py      # Pydantic request/response models
│   └── monitor/
│       ├── drift_check.py  # Evidently drift detection pipeline
│       ├── custom_metrics.py # 3 custom drift metrics
│       └── __init__.py     # Exports
├── tests/                  # Unit tests
├── data/
│   ├── raw/                # Original Telco CSV
│   ├── processed/          # Train/test splits (reference)
│   └── drift_injected/     # Reference + drifted current
├── reports/                # Generated HTML drift reports
└── mlruns/                 # MLflow tracking artifacts
```

---

## Configuration (src/config.py)

Key settings:
```python
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "telco-churn-experiment"
MLFLOW_MODEL_NAME = "telco-churn-model"
DRIFT_THRESHOLDS = {
    "monthly_charges_mean_diff": 5.0,
    "churn_rate_contract_shift": 0.05,
    "tenure_mean_diff": 2.0,
}
```

---

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_custom_metrics.py -v
```

---

## Example MLflow UI Screenshots

> **Note**: Run `mlflow ui` locally to view actual screenshots.

### Experiment Comparison Table
![Experiment Comparison](docs/mlflow_experiment_comparison.png)

### Model Registry Stages
![Model Registry](docs/mlflow_model_registry.png)

### Drift Monitoring Artifacts
![Drift Reports](docs/mlflow_drift_artifacts.png)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `uv sync` |
| MLflow UI empty | Ensure `mlruns/` exists, run training first |
| Model not loading in API | Check model is in Production stage: `mlflow models list` |
| Drift data not found | Run `uv run python -m src.data.split` first |
| Port 8000 in use | Change `API_PORT` in `src/config.py` |

---

## License

MIT License - Feel free to use for learning and production.
