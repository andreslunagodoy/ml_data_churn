# Customer Churn Prediction — Production ML Pipeline

End-to-end **Machine Learning Engineering project** for predicting customer churn using the Telco Customer Churn dataset.

This project demonstrates how a typical **data science notebook project can be refactored into a production-ready ML system**, including:

- Modular Python package
- Reproducible training pipeline
- Versioned models
- Batch predictions
- FastAPI inference API
- Logging, configuration, and testing

The project is structured as a **3-stage ML engineering portfolio project**:

| Stage | Focus | Outcome |
|-----|------|------|
| Stage 1 | Data Science | Exploratory analysis and modeling in notebooks |
| Stage 2 | ML Engineering | Refactor notebook code into modular pipeline |
| Stage 3 | Deployment | API, validation, testing, and documentation |

---

# Project Architecture

The repository follows a **production-style ML project layout**.

```
.
├── README.md
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
│   └── train_pipeline_timestamp.log
├── models
│   ├── current
│   │   ├── config.json
│   │   ├── metrics.json
│   │   ├── model.pkl
│   │   └── preprocessor.pkl
│   └── v1
├── notebooks
├── output
├── requirements.txt
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
git clone <repo_url>
cd churn-ml-pipeline
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

# Dataset

The project uses the **Telco Customer Churn dataset**.

Location:

```
data/raw/telco_churn.csv
```

Target variable:

```
Churn
```

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
2. Apply preprocessing
3. Train model
4. Evaluate performance
5. Save model artifacts

Run training:

```
python scripts/train.py
```

Artifacts are saved to:

```
models/current/
```

Files generated:

| File | Description |
|----|----|
| model.pkl | Trained ML model |
| preprocessor.pkl | Feature preprocessing pipeline |
| metrics.json | Model evaluation metrics |
| config.json | Training configuration |

---

# Model Versioning

Models are stored in the `models/` directory.

Example:

```
models/
├── current/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── metrics.json
│   └── config.json
└── v1/
```

Workflow:

- Each training run produces artifacts
- `current/` holds the latest model
- Older versions can be archived as `v1`, `v2`, etc.

This mimics **basic experiment tracking and model registry behavior**.

---

# Batch Predictions

Batch predictions are run using the prediction script.

Input data example:

```
input/new_data.csv
```

Run prediction:

```
python scripts/predict.py --input input/new_data.csv
```

Predictions are stored in a timestamped directory:

```
output/
└── 260310_170204/
    └── predictions.csv
```

---

# API Deployment

A **FastAPI service** exposes the trained model for real-time predictions.

Start the API:

```
uvicorn api.main:app --reload
```

Default endpoint:

```
http://127.0.0.1:8000
```

Interactive documentation:

```
http://127.0.0.1:8000/docs
```

---

# Example API Request

Example input JSON:

```
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1025.0
}
```

Example response:

```
{
  "churn_probability": 0.73,
  "prediction": "Yes"
}
```

---

# Logging

All pipeline runs generate logs stored in:

```
logs/
```

Logging captures:

- Training start/end
- Data loading
- Model training
- Evaluation metrics
- Prediction runs
- API events

This mimics **production observability practices**.

---

# Testing

Basic unit tests are provided for key modules.

Test coverage includes:

- Model loading
- Prediction logic
- API endpoints

Run tests with:

```
pytest tests/
```

---

# Notebooks

Exploratory analysis and early modeling were done in:

```
notebooks/
```

Examples:

- Churn_DA_v0.ipynb

This notebook represents the **Week 1 Data Science phase** before refactoring.

- Churn_DA_v1.ipynb
- Churn_DA_v2.ipynb

These notebooks represents different stages of refactoring.

---

# Core Source Modules

The `src/` package contains the **modular ML pipeline components**.

| Module | Purpose |
|------|------|
| config.py | Project configuration |
| data_loader.py | Data ingestion |
| preprocessing.py | Feature preprocessing |
| train_model.py | Model training |
| evaluate.py | Model evaluation |
| model_loader.py | Loading saved artifacts |
| predict.py | Batch prediction logic |
| predict_single.py | Single record prediction |
| logger.py | Logging utilities |

---

# Scripts

Scripts provide **CLI entry points** for pipeline operations.

| Script | Purpose |
|------|------|
| scripts/train.py | Train and save a model |
| scripts/predict.py | Run batch predictions |
| scripts/aux_generatedata.py | Generate example data |

---

# Example Input Data

Example JSON input for API testing:

```
data/examples/example_dict.json
```

---

# Future Improvements

Potential next steps to extend the project:

- Docker containerization
- CI/CD pipeline
- Model monitoring
- Data drift detection
- Experiment tracking (MLflow)
- Feature store integration
- Automated retraining

---

# Tech Stack

Machine Learning

- scikit-learn
- pandas
- numpy

API

- FastAPI
- Uvicorn
- Pydantic

Engineering

- Python packaging
- Logging
- Modular pipelines
- Pytest

---

# Learning Goals

This project demonstrates **ML Engineering best practices**:

- Refactoring notebooks into production code
- Modular pipeline design
- Reproducible training
- Model artifact management
- Serving ML models through APIs
- Testing ML systems

---

# Portfolio-Level Enhancements

To further demonstrate **production-grade ML engineering practices**, the project includes several features typically found in real-world ML systems.

These additions make the repository closer to a **deployable ML service rather than a simple model project**.

---

### 1. Reproducible Training via Configuration

All model training parameters are stored in a configuration file.

Example configuration:

```
models/current/config.json
```

Example contents:

```
{
    "target_column": "Churn",
    "target_map": {
        "Yes": 1,
        "No": 0
    },
    "test_size": 0.2,
    "random_state": 42,
    "model_type": "logistic_regression",
    "timestamp": "260305_160127"
}
```

Benefits:

- Ensures experiments are reproducible
- Allows easy hyperparameter changes
- Separates **code from configuration**

---

### 2. Model Artifact Management

Training produces a **complete model artifact bundle** rather than only a single model file.

Each model version contains:

```
models/current/
├── model.pkl
├── preprocessor.pkl
├── metrics.json
└── config.json
```

Artifact descriptions:

| Artifact | Purpose |
|--------|--------|
| model.pkl | Trained ML model |
| preprocessor.pkl | Feature engineering pipeline |
| metrics.json | Evaluation metrics |
| config.json | Training configuration |

Why this matters:

Production ML systems must keep **all components required to reproduce predictions**, not just the model weights.

This ensures:

- reproducibility
- rollback capability
- consistent preprocessing during inference

---

### 3. Timestamped Prediction Outputs

Batch predictions automatically generate **timestamped output directories**.

Example:

```
output/
├── 260305_171456/
│   └── predictions.csv
├── 260310_163059/
│   └── predictions.csv
```

Benefits:

- Every prediction run is preserved
- Results are easy to audit
- No risk of overwriting previous outputs

This mirrors how **production inference jobs** are logged in real ML pipelines.

---

### 4. Structured Logging

The project uses centralized logging utilities located in:

```
src/logger.py
```

Logs are stored in:

```
logs/
```

Example logged events:

- Training start and completion
- Dataset loading
- Model evaluation results

Logging is essential for:

- debugging pipelines
- monitoring model performance
- production observability

---

### 5. API Input Validation

The FastAPI service validates all incoming prediction requests using **Pydantic schemas**.

Defined in:

```
api/schemas.py
```

Benefits:

- prevents malformed inputs
- ensures feature types are correct
- improves API reliability

Example validation features:

- type checking
- required fields
- structured request format

---

### 6. Automated Testing

Unit tests are included for key components of the system.

Location:

```
tests/
```

Example tests:

| Test File | Coverage |
|---------|---------|
| test_model.py | model loading and prediction |
| test_predict.py | batch prediction pipeline |
| test_api.py | API endpoints |

Run all tests with:

```
pytest tests/
```

Testing ensures:

- prediction logic works
- API endpoints remain stable
- future refactoring does not break the pipeline

---

## Why These Features Matter

Many ML portfolio projects stop at:

- training a model
- saving a `.pkl`
- showing accuracy

This project demonstrates **ML Engineering practices used in real production systems**:

- modular ML pipelines
- model artifact management
- API-based inference
- reproducible experiments
- structured logging
- automated tests

These practices bridge the gap between **Data Science prototypes and deployable ML systems**.

# Author
Andres Luna

https://github.com/andreslunagodoy
https://www.linkedin.com/in/andres-luna-06a31b101/

Machine Learning Portfolio Project

Focus: **Applied ML Engineering and Production Pipelines**