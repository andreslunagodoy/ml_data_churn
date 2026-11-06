from sklearn.model_selection import train_test_split
import argparse
import joblib
import json

from churn.config import load_config, get_model_path, MODELS_FAMILY, TIMESTAMP
from churn.logger import setup_logger
from churn.data_loader import load_raw_data
from churn.preprocessing import get_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES
from churn.train_model import train_model
from churn.evaluate import evaluate_model


logger = setup_logger(f"train_pipeline_{TIMESTAMP}")

def main():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--config", type=str, default=None, help="YAML file with values that override config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = get_model_path(config)

    # Start logger
    logger.info("Starting training pipeline")

    model_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df_raw = load_raw_data()

    logger.info(f"Loaded data with shape {df_raw.shape}")

    # Split features and target
    X_raw = df_raw.drop(columns=config['target_column'])
    y = df_raw[config['target_column']].map(config['target_map'])

    logger.info(f"Split features and target {config['target_column']}")

    # Split train and test sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=config['test_size'], stratify=y, random_state=config['random_state']
    )

    logger.info(f"Split train and test set of size {config['test_size']}")

    # Preprocessing
    preprocessor = get_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES, config['model_type'])

    preprocessor.fit(X_train_raw, y_train)
    X_train = preprocessor.transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    logger.info(f"Finished preprocessing for a {MODELS_FAMILY[config['model_type']]}-type model")

    # Train model
    model = train_model(X_train, y_train, config['model_type'], config['random_state'])
    logger.info(f"Completed training a {config['model_type']} model")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Metrics: {metrics}")

    # Save artifacts
    joblib.dump(model, model_path / "model.pkl")
    joblib.dump(preprocessor, model_path / "preprocessor.pkl")
    with open(model_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    with open(model_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    # Test split (raw features) saved as a stand-in for new data in batch predictions
    X_test_raw.to_csv(model_path / "proxy_input.csv", index=False)

    logger.info(f"Saved model and preprocessor to {model_path}")
    logger.info("Training pipeline finished successfully")

if __name__ == "__main__":
    main()
