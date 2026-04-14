from sklearn.model_selection import train_test_split
import argparse
import joblib
import json

from src.config import CONFIG, MODEL_PATH, MODELS_FAMILY, TIMESTAMP
from src.logger import setup_logger
from src.data_loader import load_raw_data
from src.preprocessing import get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES
from src.train_model import train_model
from src.evaluate import evaluate_model


logger = setup_logger(f"train_pipeline_{TIMESTAMP}")

def main():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file (default: config.yaml)")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            CONFIG.update(yaml.safe_load(f))
    # Start logger
    logger.info("Starting training pipeline")

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Load data
    df_raw = load_raw_data()

    logger.info(f"Loaded data with shape {df_raw.shape}")

    # Split features and target
    X_raw = df_raw.drop(columns=CONFIG['target_column'])
    y = df_raw[CONFIG['target_column']].map(CONFIG['target_map'])

    logger.info(f"Split features and target {CONFIG['target_column']}")

    # Split train and test sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=CONFIG['test_size'], stratify=y, random_state=CONFIG['random_state']
    )

    logger.info(f"Split train and test set of size {CONFIG['test_size']}")

    # Preprocessing
    preprocessor = get_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES, CONFIG['model_type'])

    preprocessor.fit(X_train_raw, y_train)
    X_train = preprocessor.transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    logger.info(f"Finished preprocessing for a {MODELS_FAMILY[CONFIG['model_type']]}-type model")

    # Train model
    model = train_model(X_train, y_train, CONFIG['model_type'])
    logger.info(f"Completed training a {CONFIG['model_type']} model")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Metrics: {metrics}")

    # Save artifacts
    joblib.dump(model, MODEL_PATH / "model.pkl")
    joblib.dump(preprocessor, MODEL_PATH / "preprocessor.pkl")
    with open(MODEL_PATH / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    with open(MODEL_PATH / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=4)

    logger.info(f"Saved model and preprocessor to {MODEL_PATH}")
    logger.info("Training pipeline finished successfully")

if __name__ == "__main__":
    main()
