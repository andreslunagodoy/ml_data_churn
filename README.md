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
├── requirements.txt
├── api
│   ├── main.py
│   └── schemas.py
├── data
│   ├── examples
│   │   └── example_dict.json
│   ├── external
│   ├── processed
│   └── raw
│       └── telco_churn.csv
├── input
│   └── new_data.csv
├── logs
├── models
│   ├── current
│   │   ├── config.json
│   │   ├── metrics.json
│   │   ├── model.pkl
│   │   └── preprocessor.pkl
│   └── v1
├── notebooks
├── output
├── scripts
│   ├── predict.py
│   └── train.py
├── src
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── logger.py
│   ├── model_loader.py
│   ├── predict.py
│   ├── predict_single.py
│   ├── preprocessing.py
│   └── train_model.py
├── tests
│   ├── test_api.py
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_predict.py
└── tools
    └── aux_generatedata.py
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
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

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

To change the model type or training parameters, edit this file — no code changes needed. You can also pass an alternative config file when training:

```
python -m scripts.train --config path/to/other_config.yaml
```

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

Artifacts are saved to `models/current/`:

| File | Description |
|----|----|
| model.pkl | Trained ML model |
| preprocessor.pkl | Feature preprocessing pipeline |
| metrics.json | Model evaluation metrics |
| config.json | Training configuration snapshot |

---

# Model Versioning

Models are stored in the `models/` directory.

```
models/
├── current/          # Latest model (tracked in git)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── metrics.json
│   └── config.json
└── v1/               # Historical versions (git-ignored)
```

- `current/` holds the latest model and is tracked in git so the API works immediately after cloning
- Older versions are archived under `v1/`, `v2/`, etc. and are git-ignored to avoid bloating the repo
- The model loader validates that model and preprocessor come from the same directory and that a `config.json` is present

---

# Batch Predictions

Run batch predictions with:

```
python -m scripts.predict
```

By default, reads from `input/new_data.csv` and writes to a timestamped directory under `output/`. You can override paths:

```
python -m scripts.predict --input path/to/data.csv --output path/to/results.csv
```

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

---

# Testing

The test suite covers model loading, prediction logic, API endpoints, data loading, and input validation.

| Test File | Tests | What it covers |
|-----------|-------|---------------|
| test_api.py | 9 | Health check, prediction, invalid inputs (bad gender, contract, negative tenure, etc.) |
| test_data_loader.py | 2 | Successful load, missing file error |
| test_model.py | 3 | Model/preprocessor loading, required methods |
| test_predict.py | 4 | Single prediction, output columns, missing columns, probability range |

Run all tests:

```
pytest tests/ -v
```

---

# CI/CD

A **GitHub Actions** pipeline runs automatically on every push to `main` and on pull requests.

The pipeline runs two jobs:

### Lint and Test

1. Install dependencies
2. **Lint** with `ruff` — catches style issues, unused imports, common mistakes
3. **Type check** with `mypy` — verifies type hint consistency
4. **Train model** — trains from scratch so integration tests can run
5. **Run tests** — full test suite (18 tests)

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

The `src/` package contains the **modular ML pipeline components**. All functions include type hints.

| Module | Purpose |
|------|------|
| config.py | Loads configuration from `config.yaml`, defines project paths |
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
