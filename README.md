# Customer Churn Prediction — Production ML Pipeline

End-to-end **Machine Learning Engineering project** for predicting customer churn using the Telco Customer Churn dataset.

This project demonstrates how a typical **data science notebook project can be refactored into a production-ready ML system**, including:

- Modular Python package
- Reproducible training pipeline
- Versioned models
- Batch predictions
- FastAPI inference API
- Input validation with Pydantic
- CI/CD with GitHub Actions
- Docker containerization
- Logging, configuration, and testing

The project is structured as a **3-stage ML engineering portfolio project**:

| Stage | Focus | Outcome |
|-----|------|------|
| Stage 1 | Data Science | Exploratory analysis and modeling in notebooks |
| Stage 2 | ML Engineering | Refactor notebook code into modular pipeline |
| Stage 3 | Deployment | API, validation, testing, CI/CD, and Docker |

---

# Project Architecture

The repository follows a **production-style ML project layout**.

```
.
├── .github/workflows/ci.yml
├── Dockerfile
├── README.md
├── config.yaml
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
├── api
│   ├── main.py
│   └── schemas.py
├── data
│   ├── examples
│   │   └── example_dict.json
│   └── raw
│       └── telco_churn.csv
├── logs
├── models
│   ├── current
│   │   ├── config.json
│   │   ├── metrics.json
│   │   ├── model.pkl
│   │   ├── preprocessor.pkl
│   │   └── proxy_input.csv
│   └── v1
├── notebooks
├── output
├── scripts
│   ├── predict.py
│   ├── promote.py
│   └── train.py
├── churn
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── logger.py
│   ├── model_loader.py
│   ├── predict.py
│   ├── predict_single.py
│   ├── preprocessing.py
│   └── train_model.py
└── tests
    ├── test_api.py
    ├── test_config.py
    ├── test_data_loader.py
    ├── test_model.py
    ├── test_predict.py
    ├── test_preprocessing.py
    ├── test_promote.py
    └── test_train_model.py
```

---

# Project Workflow

The pipeline follows a **typical ML lifecycle**:

```
Raw Data
   │
   ▼
Data Loader
   │
   ▼
Preprocessing Pipeline
   │
   ▼
Model Training
   │
   ▼
Evaluation
   │
   ▼
Model Versioning
   │
   ▼
Prediction (Batch or API)
```

---

# Installation

Clone the repository:

```
git clone https://github.com/andreslunagodoy/ml_data_churn.git
cd ml_data_churn
```

Create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements-dev.txt
pip install -e .
```

`requirements.txt` holds the pinned runtime dependencies (this is what the Docker image installs). `requirements-dev.txt` includes it and adds the test and lint tools. The second command installs the project in editable mode, so `churn` can be imported from anywhere.

---

# Configuration

Training parameters are defined in `config.yaml` at the project root:

```yaml
target_column: "Churn"
target_map:
  "Yes": 1
  "No": 0
test_size: 0.2
random_state: 42
model_type: "logistic_regression"
model_version: "v1"
```

To change the model type or training parameters, edit this file — no code changes needed. You can also pass an override file when training:

```
python -m scripts.train --config path/to/override.yaml
```

The override file only needs the keys you want to change; they are merged over `config.yaml`. Unknown keys and invalid values (such as an unknown `model_type` or a `test_size` outside 0–1) are rejected with an error.

---

# Dataset

The project uses the **Telco Customer Churn dataset**.

Location:

```
data/raw/telco_churn.csv
```

Target variable: `Churn`

Typical features include:

- Contract type
- Monthly charges
- Tenure
- Internet service
- Payment method
- Demographics

---

# Training the Model

Training runs the full pipeline:

1. Load data
2. Apply preprocessing (cleaning, feature engineering, scaling/encoding)
3. Train model
4. Evaluate performance
5. Save model artifacts

Run training:

```
python -m scripts.train
```

Artifacts are saved to `models/<model_version>/<timestamp>/` (for example `models/v1/261106_141423/`):

| File | Description |
|----|----|
| model.pkl | Trained ML model |
| preprocessor.pkl | Feature preprocessing pipeline |
| metrics.json | Model evaluation metrics |
| config.json | Training configuration snapshot |
| proxy_input.csv | Test-split features, a stand-in for new data in batch predictions |

---

# Model Versioning

Models are stored in the `models/` directory.

```
models/
├── current/          # The model the API and batch predictions use (tracked in git)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── metrics.json
│   ├── config.json
│   └── proxy_input.csv
└── v1/<timestamp>/   # Every training run (git-ignored)
```

- Training never touches `current/`. Promoting a model is an explicit step:

  ```
  python scripts/promote.py models/v1/<timestamp>
  ```

  It copies the five artifacts into `models/current/`, and refuses to copy anything if one is missing.
- `current/` is tracked in git so the API works immediately after cloning. Training runs are git-ignored to avoid bloating the repo.
- Set the `CHURN_MODEL_DIR` environment variable to make the API and batch predictions load a different model folder. CI uses this to test the model it just trained.
- The model loader validates that model and preprocessor come from the same directory and that a `config.json` is present.

---

# Batch Predictions

Run batch predictions with:

```
python -m scripts.predict
```

By default, uses the model in `models/current/`, reads that folder's `proxy_input.csv` (the test split saved by training, used as a stand-in for new data) and writes to a timestamped directory under `output/`. You can override each of these:

```
python -m scripts.predict --model-dir models/v1/<timestamp> --input path/to/data.csv --output path/to/results.csv
```

If `--model-dir` is given without `--input`, the input defaults to that folder's own `proxy_input.csv`.

Input data is validated against a required column list before preprocessing.

---

# API Deployment

A **FastAPI service** exposes the trained model for real-time predictions.

Start the API:

```
uvicorn api.main:app --reload
```

Endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single customer prediction |
| `/docs` | GET | Interactive API documentation |

The model and preprocessor are loaded once at startup via FastAPI's lifespan pattern and stored on `app.state`.

### Running with Docker

```
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

# Example API Request

Example input JSON:

```json
{
  "customerID": "3524-WQDSG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 43,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 99.3,
  "TotalCharges": "4209.95"
}
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.31
}
```

### Input Validation

All API inputs are validated with Pydantic using `Literal` types for categorical fields and `Field` constraints for numeric fields. Invalid requests return a 422 with details on what's wrong. For example:

- `gender` only accepts `"Male"` or `"Female"`
- `Contract` only accepts `"Month-to-month"`, `"One year"`, or `"Two year"`
- `tenure` must be >= 0
- `MonthlyCharges` must be >= 0
- `TotalCharges` is a string, as in the raw data. It must be a finite number >= 0, or blank (the dataset has blanks for brand-new customers)

---

# Testing

The test suite (39 tests) covers configuration, model loading and promotion, prediction logic, API endpoints, data loading, and input validation.

| Test File | Tests | What it covers |
|-----------|-------|---------------|
| test_api.py | 14 | Health check, prediction, invalid inputs (gender, contract, tenure, charges, `TotalCharges` format), blank `TotalCharges` |
| test_config.py | 9 | `model_type` validation, config overrides, unknown keys, invalid values, model path, `CHURN_MODEL_DIR` |
| test_data_loader.py | 2 | Successful load, missing file error |
| test_model.py | 3 | Model/preprocessor loading, required methods |
| test_predict.py | 4 | Single prediction, output columns, missing columns, probability range |
| test_preprocessing.py | 2 | `tenure_group` covers every tenure value |
| test_promote.py | 3 | Promotion copies all artifacts, overwrites, refuses an incomplete folder |
| test_train_model.py | 2 | Seeded training is reproducible, a different seed changes the model |

Run all tests:

```
pytest
```

The tests use the model in `models/current/`, or the folder named by `CHURN_MODEL_DIR`.

---

# CI/CD

A **GitHub Actions** pipeline runs automatically on every push to `main` and on pull requests.

The pipeline runs two jobs:

### Lint and Test

1. Install dependencies (`requirements-dev.txt` and the package itself)
2. **Lint** with `ruff` — catches style issues, unused imports, common mistakes
3. **Type check** with `mypy` — verifies type hint consistency
4. **Train model** — trains from scratch into `models/v1/<timestamp>/`
5. **Run tests** — full test suite, pointed at the freshly trained model through `CHURN_MODEL_DIR`, so CI tests the model it just built rather than the committed one

### Docker Build

1. **Build Docker image** — verifies the Dockerfile and dependencies work

Configuration for `ruff` and `mypy` lives in `pyproject.toml`.

---

# Logging

All pipeline runs generate logs stored in `logs/`. Each training run creates a timestamped log file.

Logging captures:

- Training start/end
- Data loading
- Preprocessing steps
- Model evaluation metrics

---

# Notebooks

Exploratory analysis and early modeling were done in `notebooks/`:

| Notebook | Stage |
|----------|-------|
| Churn_DA_v0.ipynb | Initial data science exploration |
| Churn_DA_v1.ipynb | First refactoring pass |
| Churn_DA_v2.ipynb | Final refactoring into modular pipeline |

These represent the progression from notebook prototype to production code.

---

# Core Source Modules

The `churn/` package contains the **modular ML pipeline components**. All functions include type hints.

| Module | Purpose |
|------|------|
| config.py | Project paths, `load_config` (defaults + optional override file, validated), `get_model_path` |
| data_loader.py | Data ingestion with file existence and empty data validation |
| preprocessing.py | Feature engineering, cleaning, and sklearn preprocessing pipelines |
| train_model.py | Model training (logistic regression, random forest, gradient boosting) |
| evaluate.py | Model evaluation (accuracy, ROC-AUC, precision, recall, F1) |
| model_loader.py | Loading saved artifacts with compatibility validation |
| predict.py | Batch prediction with column validation |
| predict_single.py | Single record prediction |
| logger.py | Logging utilities |

---

# Tech Stack

**Machine Learning:** scikit-learn, pandas, numpy

**API:** FastAPI, Uvicorn, Pydantic

**Testing:** pytest

**CI/CD:** GitHub Actions, ruff, mypy

**Deployment:** Docker

---

# Future Improvements

Potential next steps:

- Model monitoring and data drift detection
- Experiment tracking (MLflow)
- Feature store integration
- Automated retraining triggers
- CD pipeline (push Docker image to registry, deploy to cloud)

---

# Author

Andres Luna

https://github.com/andreslunagodoy
https://www.linkedin.com/in/andres-luna-06a31b101/

Machine Learning Portfolio Project

Focus: **Applied ML Engineering and Production Pipelines**
